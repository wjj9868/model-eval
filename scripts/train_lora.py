# @author: ztwz
"""LoRA 蒸馏训练：用线上 teacher 输出做 SFT，产出可直接被 vLLM 托管的合并权重。

数据：{prompt, output} 的 JSONL（prompt 已是线上渲染好的完整输入，无 system prompt）。
目标是"给定 prompt 产出六字段 JSON"，因此：
- 走 chat 模板拼成 user/assistant 两段，**只在 assistant 段算 loss**（train_on_responses_only），
  避免把算力浪费在复述超长 prompt 上；
- 训练后默认合并权重落盘（save_pretrained_merged），评测端直接用 vLLM 起服务，
  不必依赖 vLLM 对该混合架构的 LoRA 支持。

依赖：.venv-train（unsloth + trl + peft + transformers<=5.5.0，复用系统 torch）。
用法：
  .venv-train/bin/python scripts/train_lora.py \
      --model Qwen/Qwen3.5-2B --train-file data/train_lora_2k.jsonl \
      --out-dir models/lora_2b --merged-dir models/merged_2b \
      --epochs 1 --max-seq-len 8192
"""
from __future__ import annotations

import argparse
import inspect
import json
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA 蒸馏训练（Unsloth + trl）")
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--train-file", default="data/train_lora_2k.jsonl")
    p.add_argument("--out-dir", default="models/lora_run")
    p.add_argument("--merged-dir", default=None, help="给定则训练后合并权重落盘（供 vLLM 直接加载）")
    p.add_argument("--limit", type=int, default=0, help="只用前 N 条（0=全部），便于先跑小样")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1, help=">0 时覆盖 epochs（冒烟测试用）")
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_records(args) -> list[dict]:
    """读入 {prompt, output} 训练对"""
    records = []
    with open(args.train_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if args.limit and len(records) >= args.limit:
                break
    return records


def build_texts(records: list[dict], tokenizer) -> list[dict]:
    """按 chat 模板拼 user(prompt)+assistant(output) 两段文本"""
    texts = []
    for record in records:
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": record["prompt"]},
                {"role": "assistant", "content": record["output"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append({"text": text})
    return texts


def main() -> None:
    args = parse_args()

    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    records = load_records(args)
    print(f"训练样本 {len(records)} 条（{args.train_file}）", flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,          # 自动（A10 走 bf16）
        load_in_4bit=False,  # 单卡 23G 下 2B/4B 的 LoRA 用 bf16 更稳、更快
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 省显存的关键
        random_state=args.seed,
        # 不指定 target_modules：混合架构（GatedDeltaNet + Full Attention）交给 unsloth 自动选择更稳
    )

    # Qwen3.5 是多模态模型：from_pretrained 拿到的是 processor，直接喂纯文本会被当成图片源解析。
    # 训练只用文本，取底层文本 tokenizer（processor.tokenizer）统一用于统计与 collator。
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    dataset_list = build_texts(records, text_tokenizer)
    lengths = [len(text_tokenizer(t["text"], add_special_tokens=False)["input_ids"]) for t in dataset_list]
    over = sum(1 for n in lengths if n > args.max_seq_len)
    print(f"序列长度：中位={sorted(lengths)[len(lengths) // 2]} 最大={max(lengths)} "
          f"超 {args.max_seq_len} 被截断={over}/{len(lengths)}", flush=True)

    from datasets import Dataset

    dataset = Dataset.from_list(dataset_list)

    # 只对 assistant 段算 loss（按当前 tokenizer 的模板取锚点串）
    instruction_part = "<|im_start|>user\n"
    response_part = "<|im_start|>assistant\n"
    if "<|im_start|>" not in tokenizer.chat_template:
        instruction_part, response_part = None, None
        print("警告：模板不是 Qwen 风格，跳过 response-only 掩码（全序列算 loss）", flush=True)

    # 兼容不同 trl 版本的 SFTConfig 参数名（max_seq_length / max_length 等）
    supported = set(inspect.signature(SFTConfig.__init__).parameters)
    config_kwargs = {
        "output_dir": args.out_dir,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.lr,
        "bf16": True,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "optim": "adamw_8bit",
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "seed": args.seed,
        "report_to": [],
        "dataset_text_field": "text",
        "max_seq_length": args.max_seq_len,
        "max_length": args.max_seq_len,
        "packing": False,
    }
    config_kwargs = {k: v for k, v in config_kwargs.items() if k in supported}
    print(f"SFTConfig 生效参数：{sorted(config_kwargs)}", flush=True)

    trainer_kwargs = {"model": model, "train_dataset": dataset, "args": SFTConfig(**config_kwargs)}
    trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = text_tokenizer
    else:
        trainer_kwargs["tokenizer"] = text_tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    if instruction_part:
        try:
            from trl import train_on_responses_only as mask_helper
        except ImportError:  # 老版本在 unsloth 命名空间下
            from unsloth.chat_templates import train_on_responses_only as mask_helper
        trainer = mask_helper(trainer, instruction_part=instruction_part, response_part=response_part)

    started = time.time()
    trainer.train()
    print(f"训练完成，用时 {(time.time() - started) / 60:.1f} 分钟", flush=True)

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"LoRA 适配器已保存：{args.out_dir}", flush=True)

    if args.merged_dir:
        # 合并回 16bit 全量权重：评测端 vLLM 直接加载，绕开 vLLM 对混合架构 LoRA 的支持问题
        model.save_pretrained_merged(args.merged_dir, text_tokenizer, save_method="merged_16bit")
        print(f"合并权重已保存：{args.merged_dir}", flush=True)


if __name__ == "__main__":
    main()
