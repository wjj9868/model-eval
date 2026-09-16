# @author: ztwz
"""LoRA 蒸馏训练：用线上 teacher 输出做 SFT，产出可直接被 vLLM 托管的合并权重。

数据：{prompt, output} 的 JSONL（prompt 已是线上渲染好的完整输入）。
口径（已对照线上源码 UserChatAnalysisAI / OpenRouterTaskCaller 核实）：线上为单条 user
消息（无 system，任务指令嵌在 prompt 模板头部）+ response_format strict + temperature=1.0，
训练样本构造与此同构；留出集评测同样 user-only、temperature 对齐线上。
链路：clean_train_data.py 清洗 → split_dataset.py 划分 → 本脚本训练（默认输入 data/split/train.jsonl）。

目标是「给定 prompt 产出六字段 JSON」，因此：
- 走 chat 模板拼成 user/assistant 两段，**只在 assistant 段算 loss**（train_on_responses_only），
  避免把算力浪费在复述超长 prompt 上；
- 训练后默认合并权重落盘（save_pretrained_merged），评测端直接用 vLLM 起服务，
  不必依赖 vLLM 对该混合架构的 LoRA 支持；
- 训练结束自动在留出集上跑一次真实评测（生成 + 复用仓库评分器），
  避免只看 loss 就交付（loss 低但结构化输出崩掉是 SFT 的典型失败模式）。

关键配置的依据（按论文/实践选型）：
- **NEFTune**（Jain et al., ICLR 2024, arXiv:2310.05914）：训练时给 embedding 加噪，
  指令跟随显著提升（LLaMA-2-7B AlpacaEval 胜率 29.79% → 64.69%）；7B 量级推荐 α=5。
- **rsLoRA**（Kalajdzievski, 2023）：秩稳定缩放，rank ≥ 16 时比经典 LoRA 更稳。
- **cosine + warmup**：长序列 SFT 的标准调度，比 linear 更平滑（配合单 epoch）。
- **宽覆盖 target modules**（Qwen 系默认 q/k/v/o/gate/up/down 全线性层）：
  "LoRA Learns Less and Forgets Less"(TMLR 2024) 指出领域适配需要更宽的适配面。
- **长度过滤而非静默截断**：被截断到看不见答案的样本一律剔除并记录，
  避免全 -100 标签引发的 NaN/无效样本（max_seq_len 从 8192 提到 12288，剔除率 8% → ~1%）。
- **batch × 累积 + group_by_length**：按长度分桶减少 padding 浪费。

依赖：.venv-train（unsloth + trl + peft + transformers<=5.5.0，复用系统 torch）。
用法：
  .venv-train/bin/python scripts/train_lora.py \
      --model Qwen/Qwen3.5-2B --train-file data/split/train.jsonl \
      --eval-file data/split/test_1k.jsonl --eval-limit 30 \
      --out-dir models/lora_2b --merged-dir models/merged_2b --epochs 1
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

# 允许直接 `python scripts/train_lora.py` 运行：把仓库根加入 sys.path（供训练后评测复用 app 评分器）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA 蒸馏训练（Unsloth + trl）")
    # 数据与产物
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--train-file", default="data/split/train.jsonl",
                   help="训练数据（链路：clean_train_data → split_dataset → 本脚本）")
    p.add_argument("--out-dir", default="models/lora_run")
    p.add_argument("--merged-dir", default=None, help="给定则训练后合并权重落盘（供 vLLM 直接加载）")
    p.add_argument("--limit", type=int, default=0, help="只用前 N 条（0=全部）")
    # 训练超参
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1, help=">0 时覆盖 epochs（冒烟测试用）")
    p.add_argument("--max-seq-len", type=int, default=12288)
    p.add_argument("--lr", type=float, default=2e-4, help="LoRA 可比全参微调高一个量级")
    p.add_argument("--batch-size", type=int, default=8,
                   help="每设备 batch：A10 23G 下 2B 的 LoRA 用 8 仍有余量，且能喂满算力（旧默认 2=显存只用到 ~20%）")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="梯度累积步数，与 --batch-size 乘积 = global batch（默认 = 8 × 1）")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--neftune-alpha", type=float, default=5.0,
                   help="NEFTune embedding 噪声强度（ICLR 2024；0=关闭）")
    p.add_argument("--lr-scheduler", default="cosine")
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--group-by-length", action="store_true", default=True,
                   help="按长度分桶，减少 batch 内 padding 浪费")
    p.add_argument("--no-group-by-length", dest="group_by_length", action="store_false")
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    # 训练后留出集快速评测（生成 + 复用仓库评分器）
    p.add_argument("--eval-file", default=None, help="留出集 JSONL（不传则跳过训练后评测）")
    p.add_argument("--eval-limit", type=int, default=30)
    p.add_argument("--eval-temperature", type=float, default=1.0,
                   help="留出集生成温度（线上硬编码 1.0，默认对齐）")
    p.add_argument("--eval-max-new-tokens", type=int, default=2048)
    return p.parse_args()


def load_records(path: str, limit: int = 0) -> list[dict]:
    """读入 {prompt, output} 训练对"""
    records = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def build_texts(records: list[dict], tokenizer) -> list[dict]:
    """按 chat 模板拼 user(prompt)+assistant(output) 两段文本"""
    return [
        {
            "text": tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": record["prompt"]},
                    {"role": "assistant", "content": record["output"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        }
        for record in records
    ]


def filter_by_length(dataset_list: list[dict], tokenizer, max_seq_len: int,
                     batch_size: int = 512) -> tuple[list[dict], dict]:
    """按长度过滤：超长样本直接剔除（不截断答案），返回过滤后数据与统计。

    批量 tokenize（逐条调用在 7 万条上要 ~17 分钟，批量后约 1 分钟）。
    """
    kept, lengths = [], []
    dropped = 0
    for start in range(0, len(dataset_list), batch_size):
        batch = dataset_list[start:start + batch_size]
        enc = tokenizer([item["text"] for item in batch], add_special_tokens=False)
        for item, ids in zip(batch, enc["input_ids"]):
            n = len(ids)
            lengths.append(n)
            if n > max_seq_len:
                dropped += 1
                continue
            kept.append(item)
        if (start // batch_size) % 16 == 0:
            print(f"  长度过滤进度 {min(start + batch_size, len(dataset_list))}/{len(dataset_list)}", flush=True)
    lengths.sort()
    stats = {
        "total": len(dataset_list),
        "kept": len(kept),
        "dropped_over_length": dropped,
        "seq_len_median": lengths[len(lengths) // 2] if lengths else 0,
        "seq_len_p95": lengths[int(len(lengths) * 0.95)] if lengths else 0,
        "seq_len_max": lengths[-1] if lengths else 0,
        "max_seq_len": max_seq_len,
    }
    return kept, stats


def eval_on_holdout(model, tokenizer, args) -> dict | None:
    """训练后留出集评测：生成 + 复用仓库评分器（只输出分数，不打印样本内容）。

    口径与线上一致：单条 user 消息、temperature 对齐（本地 generate 无结构化输出约束）。
    """
    if not args.eval_file:
        return None

    import torch  # 延迟导入，脚本主体不强制依赖

    from app.evaluation.embeddings import EmbeddingClient  # noqa: E402
    from app.evaluation.scoring.aggregator import evaluate_sample  # noqa: E402
    from unsloth import FastLanguageModel  # noqa: E402

    records = load_records(args.eval_file, args.eval_limit)
    FastLanguageModel.for_inference(model)
    torch.manual_seed(args.seed)  # 评测采样可复现
    student_texts = []
    started = time.time()
    for record in records:
        messages = [{"role": "user", "content": record["prompt"]}]
        # return_dict=True 一并拿到 attention_mask（pad 与 eos 同 token 时不传会告警且行为不可靠）
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=args.eval_max_new_tokens,
            do_sample=True, temperature=args.eval_temperature, top_p=0.9,
        )
        row = out[0][inputs["input_ids"].shape[1]:]
        # 截断（打满 max_new_tokens 未停）按调用失败降级：记空文本 → 评分全 0（与线上语义一致）
        if bool((row == tokenizer.eos_token_id).any()):
            student_texts.append(tokenizer.decode(row, skip_special_tokens=True))
        else:
            student_texts.append("")
    gen_seconds = (time.time() - started) / max(len(records), 1)

    embedder = EmbeddingClient(device="cpu")  # 打分只走 CPU，避免与推理抢显存
    keys = ("total_score", "json_valid", "memory_precision", "memory_recall",
            "speaker_attribution", "hallucination_penalty", "summary_score", "intent_score", "other")
    sums = {k: [0.0, 0] for k in keys}  # [累计, 适用条数]
    for i, (record, student) in enumerate(zip(records, student_texts)):
        payload = evaluate_sample(str(i), record["prompt"], record["output"], student, embedder).to_dict()
        for k in keys:
            if payload[k] is not None:
                sums[k][0] += payload[k]
                sums[k][1] += 1

    print(f"\n=== 训练后留出集评测（n={len(records)}，{gen_seconds:.1f}s/条生成）===", flush=True)
    truncated = sum(1 for s in student_texts if not s)
    print(f"  截断失败（打满 max_new_tokens，计 0 分）: {truncated}/{len(records)}", flush=True)
    result = {}
    for k in keys:
        total, usable = sums[k]
        value = round(total / usable, 4) if usable else None
        result[k] = value
        print(f"  {k}: {value if value is not None else 'N/A'}  (适用 {usable}/{len(records)})", flush=True)
    return result


def main() -> None:
    args = parse_args()
    started_all = time.time()

    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    records = load_records(args.train_file, args.limit)
    print(f"训练样本 {len(records)} 条（{args.train_file}）", flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,          # 自动（A10 走 bf16）
        load_in_4bit=False,  # 单卡 23G 下 2B/4B 的 LoRA 用 bf16 更稳、更快
    )

    lora_kwargs = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 省显存的关键
        random_state=args.seed,
        use_rslora=True,                       # 秩稳定缩放（rank>=16 推荐）
        # 不指定 target_modules：unsloth 按架构自动选全部线性层（q/k/v/o/gate/up/down）
    )
    use_rslora = True
    try:
        model = FastLanguageModel.get_peft_model(model, **lora_kwargs)
    except TypeError:
        # 老版本 unsloth 不认 use_rslora，退回经典缩放并明确告知
        lora_kwargs.pop("use_rslora")
        use_rslora = False
        print("提示：当前 unsloth 不支持 use_rslora，回退经典 LoRA 缩放", flush=True)
        model = FastLanguageModel.get_peft_model(model, **lora_kwargs)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Qwen3.5 是多模态模型：from_pretrained 拿到的是 processor，纯文本会被当成图片源解析，
    # 训练只用文本，故取底层文本 tokenizer 统一用于统计与 collator。
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    dataset_list, seq_stats = filter_by_length(build_texts(records, text_tokenizer),
                                               text_tokenizer, args.max_seq_len)
    print(f"序列长度过滤：保留 {seq_stats['kept']}/{seq_stats['total']}，"
          f"超长剔除 {seq_stats['dropped_over_length']}；"
          f"中位 {seq_stats['seq_len_median']} P95 {seq_stats['seq_len_p95']} 最大 {seq_stats['seq_len_max']}", flush=True)
    print(f"可训练参数 {trainable:,} / {sum(p.numel() for p in model.parameters()):,} "
          f"({trainable / max(sum(p.numel() for p in model.parameters()), 1):.2%})", flush=True)

    from datasets import Dataset

    dataset = Dataset.from_list(dataset_list)

    # 只对 assistant 段算 loss（按当前 tokenizer 的模板取锚点串）
    instruction_part = response_part = None
    if "<|im_start|>" in (text_tokenizer.chat_template or ""):
        instruction_part, response_part = "<|im_start|>user\n", "<|im_start|>assistant\n"
    else:
        print("警告：模板不是 Qwen 风格，跳过 response-only 掩码（全序列算 loss）", flush=True)

    # 兼容不同 trl 版本的 SFTConfig 参数名
    supported = set(inspect.signature(SFTConfig.__init__).parameters)
    config_kwargs = {
        "output_dir": args.out_dir,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "group_by_length": args.group_by_length,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.lr,
        "bf16": True,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "optim": "adamw_8bit",
        "lr_scheduler_type": args.lr_scheduler,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": 0.0,
        "seed": args.seed,
        "data_seed": args.seed,
        "report_to": [],
        "dataset_text_field": "text",
        "max_seq_length": args.max_seq_len,
        "max_length": args.max_seq_len,
        "packing": False,  # 未启用 FA2 varlen（本机只有 xformers），packing 会引入跨样本注意力污染
        "neftune_noise_alpha": args.neftune_alpha or None,
    }
    config_kwargs = {k: v for k, v in config_kwargs.items() if k in supported}

    trainer = SFTTrainer(model=model, train_dataset=dataset, args=SFTConfig(**config_kwargs))
    if instruction_part:
        try:
            from trl import train_on_responses_only as mask_helper
        except ImportError:  # 老版本在 unsloth 命名空间下
            from unsloth.chat_templates import train_on_responses_only as mask_helper
        trainer = mask_helper(trainer, instruction_part=instruction_part, response_part=response_part)

    trainer.train()
    train_minutes = (time.time() - started_all) / 60
    print(f"训练完成，用时 {train_minutes:.1f} 分钟", flush=True)

    model.save_pretrained(args.out_dir)
    text_tokenizer.save_pretrained(args.out_dir)
    print(f"LoRA 适配器已保存：{args.out_dir}", flush=True)

    holdout = eval_on_holdout(model, text_tokenizer, args)

    # 训练配置与统计落盘，保证可复现（含数据规模、序列分布、超参、留出集分数）
    run_info = {
        "model": args.model,
        "train_file": args.train_file,
        "n_records": len(records),
        "seq_stats": seq_stats,
        "trainable_params": trainable,
        "hyperparams": {
            "epochs": args.epochs, "max_steps": args.max_steps, "lr": args.lr,
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "use_rslora": use_rslora,
            "neftune_alpha": args.neftune_alpha, "lr_scheduler": args.lr_scheduler,
            "warmup_ratio": args.warmup_ratio, "group_by_length": args.group_by_length,
            "seed": args.seed,
        },
        "train_minutes": round(train_minutes, 1),
        "holdout_scores": holdout,
        # 评测口径一并落盘，保证分数可解释（对齐线上：user-only、temperature=1.0）
        "holdout_eval_config": {
            "file": args.eval_file,
            "limit": args.eval_limit,
            "temperature": args.eval_temperature,
            "max_new_tokens": args.eval_max_new_tokens,
            "system_prompt": False,
            "response_schema": None,
            "seed": args.seed,
        },
    }
    Path(args.out_dir, "train_run.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"训练记录已保存：{args.out_dir}/train_run.json", flush=True)

    if args.merged_dir:
        model.save_pretrained_merged(args.merged_dir, text_tokenizer, save_method="merged_16bit")
        print(f"合并权重已保存：{args.merged_dir}", flush=True)


if __name__ == "__main__":
    main()
