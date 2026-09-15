# @author: ztwz
"""真实数据打分：模型推理 N 条样本 + 评分器打分。

- 后端：--backend hf（本地 transformers）或 openai（vLLM 等 OpenAI 兼容服务）
- 模型：--model 指定（Qwen/Qwen3.5-2B 或 Qwen/Qwen3.5-4B）
- 数据：data/user_chat_analysis.jsonl；--require-schema 只取 prompt 里带完整六字段
  schema 说明的样本（旧模板样本没有 rolling_summary/memory_snapshot 要求，分数不可比）
- 口径：线上输入 = prompt(单条 user 消息) + response_format(json_schema, strict)，
  默认按此还原（--no-response-schema / --no-system-prompt 可做消融）；
  --prompt-mode raw 走 base 模型原始续写
- 输出：仅分数与耗时统计；模型原始输出默认不落盘，调试时加 --dump-outputs（涉密）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

# 允许直接 `python scripts/run_real_eval.py` 运行：把仓库根加入 sys.path（无需手动 PYTHONPATH）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 允许联网下载（本地缓存优先）；纯离线环境可自行 export HF_HUB_OFFLINE=1
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from app.evaluation.embeddings import EmbeddingClient  # noqa: E402
from app.evaluation.scoring.aggregator import evaluate_sample  # noqa: E402

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"
# prompt 中同时出现这六个字段名 → 带完整输出 schema 的新模板样本（旧模板缺后三项要求）
SCHEMA_FIELDS = ("analysis_summary", "user_intent", "paid_reaction", "ai_performance",
                 "rolling_summary", "memory_snapshot")

# 线上调用是 system prompt（输出协议）+ prompt（业务输入），但导出的 jsonl 只含 user 部分，
# 真实 system prompt 不可得，这里按 app/evaluation/output_parser.py 的权威 schema 重建一份，
# 2B/4B 共用同一份以保证横向可比；拿到线上原文后用 --system-prompt-file 覆盖即可。
SYSTEM_PROMPT = (
    "你是对话分析与记忆管理专家。请只输出一个 JSON 对象，不要输出任何解释文字、前言或 Markdown 代码围栏。\n"
    "顶层字段必须齐全：analysis_summary, user_intent, paid_reaction, ai_performance, "
    "rolling_summary, memory_snapshot。\n"
    "memory_snapshot 必须是对象，包含五类数组：facts, preferences, avoid_topics, active_events, open_loops；"
    "条数上限分别为 10, 10, 5, 5, 5。\n"
    "所有字段的值必须严格依据输入中的用户基本信息、旧摘要/旧记忆与增量聊天记录，不得编造。"
)
# 控制台逐条打印的分项（grounding_mean 在 details 内，单独处理）
DIMENSIONS = ("total_score", "json_valid", "memory_precision", "memory_recall",
              "speaker_attribution", "summary_score", "intent_score", "other",
              "hallucination_penalty")


def _fmt(value) -> str:
    """可选分项的展示：None（不适用）显示 N/A"""
    return "N/A" if value is None else f"{value:.2f}"


def _mean(values: list) -> tuple[float, int]:
    """均值与适用条数：None 视为不适用，剔除后再求均值（避免把不适用当 0 拉低）"""
    usable = [v for v in values if v is not None]
    return (sum(usable) / len(usable) if usable else 0.0, len(usable))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="模型跑真实数据并用评分器打分")
    p.add_argument("--backend", choices=("hf", "openai"), default="hf",
                   help="hf: 本地 transformers；openai: 调 OpenAI 兼容服务（vLLM）")
    p.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF 模型 id（2B/4B 自行切换）")
    p.add_argument("--prompt-mode", choices=("chat", "raw"), default="chat",
                   help="chat: 套对话模板（instruct 模型吐 JSON）；raw: base 模型原始续写")
    p.add_argument("--load-4bit", action="store_true", help="用 bitsandbytes 4bit 加载（显存紧张时）")
    p.add_argument("--batch-size", type=int, default=4, help="批量推理条数（左 padding 对齐）")
    p.add_argument("--input", default="data/user_chat_analysis.jsonl")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--out", default="data/real_eval_scores.jsonl")
    p.add_argument("--dump-outputs", default=None, help="模型原始输出另存 JSONL（涉密，默认关闭）")
    p.add_argument("--require-schema", action="store_true",
                   help="只评测 prompt 含完整六字段 schema 的样本（口径可比）")
    p.add_argument("--system-prompt", default=SYSTEM_PROMPT, help="system 消息内容（输出协议）")
    p.add_argument("--system-prompt-file", default=None, help="从文件读取 system prompt（优先于 --system-prompt）")
    p.add_argument("--no-system-prompt", action="store_true", help="不发送 system 消息（A/B 对照用）")
    p.add_argument("--response-schema", default="app/schemas/user_chat_response_schema.json",
                   help="线上结构化输出 JSON Schema（response_format 强约束）")
    p.add_argument("--no-response-schema", action="store_true", help="不发送结构化输出约束（A/B 对照用）")
    p.add_argument("--base-url", default="http://127.0.0.1:8000", help="openai 后端服务根地址")
    p.add_argument("--served-model-name", default=None, help="服务端模型名（默认取 --model）")
    p.add_argument("--api-key", default=None)
    p.add_argument("--timeout", type=float, default=600.0, help="openai 后端读取超时（秒）")
    p.add_argument("--concurrency", type=int, default=8, help="openai 后端并发请求数")
    p.add_argument("--stage", choices=("both", "generate", "score"), default="both",
                   help="generate: 只生成并落盘；score: 只读落盘输出打分；both: 生成+打分")
    p.add_argument("--workers", type=int, default=1, help="打分并行进程数（>1 走多进程）")
    return p.parse_args()


def load_model(model_id: str, load_4bit: bool = False):
    """加载模型并全量驻留 GPU（bf16 默认；--load-4bit 走 bitsandbytes 4bit）"""
    if load_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        kwargs = {"quantization_config": quant, "device_map": {"": 0}}
    else:
        kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0}}
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_batch(tokenizer, model, prompts: list[str], max_new_tokens: int, prompt_mode: str,
                   system_prompt: str | None = None) -> list[str]:
    """批量推理：chat 模式套对话模板（system + user，与线上口径一致），raw 模式原文续写。

    只保留续写部分（去掉 prompt 前缀）；左 padding 对齐批量长度。
    """
    if prompt_mode == "chat":
        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    else:
        texts = prompts

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = outputs[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def resolve_system_prompt(args) -> str | None:
    """取 system prompt 内容：--no-system-prompt 关闭；--system-prompt-file 优先"""
    if args.no_system_prompt:
        return None
    if args.system_prompt_file:
        with open(args.system_prompt_file, "r", encoding="utf-8") as fin:
            return fin.read().strip()
    return args.system_prompt


def resolve_response_schema(args) -> dict | None:
    """取线上结构化输出约束（response_format 用的 JSON Schema）；--no-response-schema 关闭"""
    if args.no_response_schema or not args.response_schema:
        return None
    try:
        with open(args.response_schema, "r", encoding="utf-8") as fin:
            return json.load(fin)
    except FileNotFoundError as exc:
        raise SystemExit(f"未找到 response schema 文件：{args.response_schema}") from exc


def build_openai_config(args, system_prompt: str | None, response_schema: dict | None = None):
    """构造 call_model 所需的配置对象（复用线上同一条 OpenAI 兼容调用路径）"""
    from types import SimpleNamespace

    from app.enums import Provider

    return SimpleNamespace(
        provider=Provider.OPENAI,
        base_url=args.base_url,
        model_name=args.served_model_name or args.model,
        temperature=0.1,
        max_tokens=args.max_new_tokens,
        timeout_seconds=int(args.timeout),
        api_key=args.api_key,
        system_prompt=system_prompt,
        response_schema=response_schema,
    )


def chat_payload(config, prompt: str) -> dict:
    """按线上口径构造请求体。

    线上（UserChatAnalysisAI.java）为：messages=[user: prompt]，
    并额外传 UserChatSchema.schema() → response_format(json_schema, strict=true)。
    """
    messages = [{"role": "user", "content": prompt}]
    if config.system_prompt:
        messages.insert(0, {"role": "system", "content": config.system_prompt})
    payload = {
        "model": config.model_name,
        "messages": messages,
        "temperature": config.temperature,
        "top_p": 0.9,
        "max_tokens": config.max_tokens,
    }
    if getattr(config, "response_schema", None) is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "user_chat_analysis",
                "strict": True,
                "schema": config.response_schema,
            },
        }
    return payload


def generate_remote(config, prompts: list[str], concurrency: int) -> list[str]:
    """并发调用 OpenAI 兼容服务（vLLM），返回与 prompts 顺序对齐的输出"""
    import requests

    url = config.base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else None

    def one(prompt: str) -> str:
        response = requests.post(
            url, headers=headers, json=chat_payload(config, prompt),
            timeout=(10, config.timeout_seconds),
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(one, prompts))


def load_samples(args) -> tuple[list[dict], int]:
    """读入评测样本；--require-schema 时跳过旧模板样本（缺后三项要求，分数不可比）"""
    samples, scanned = [], 0
    with open(args.input, "r", encoding="utf-8") as fin:
        for line in fin:
            scanned += 1
            record = json.loads(line)
            if args.require_schema and not all(key in record["prompt"] for key in SCHEMA_FIELDS):
                continue
            samples.append(record)
            if len(samples) >= args.limit:
                break
    return samples, scanned


def run_generation(args, samples: list[dict], system_prompt: str | None,
                   response_schema: dict | None) -> list[str]:
    """生成阶段：批量/并发推理，逐条落盘模型原始输出，返回与 samples 对齐的输出列表。

    与打分解耦后，可先集中压满 GPU 出数（stage=generate），再单独并行打分（stage=score）。
    """
    if not args.dump_outputs:
        raise SystemExit("stage=generate/both 需要 --dump-outputs 指定模型输出落盘路径")

    tokenizer = model = remote_config = None
    if args.backend == "hf":
        tokenizer, model = load_model(args.model, args.load_4bit)
    else:
        remote_config = build_openai_config(args, system_prompt, response_schema)

    # openai 后端按“并发数的若干倍”切子批：既让服务端连续批处理跑满，又能逐批落盘看进度；
    # 本地 hf 后端按 batch-size 分批（左 padding 对齐）
    if args.backend == "openai":
        span = max(1, args.concurrency * 4)
        chunks = [samples[i:i + span] for i in range(0, len(samples), span)]
    else:
        chunks = [samples[i:i + args.batch_size] for i in range(0, len(samples), args.batch_size)]

    students: list[str] = []
    started_all = time.time()
    with open(args.dump_outputs, "w", encoding="utf-8") as dump:
        for chunk in chunks:
            print(f"生成 #{len(students)}..{len(students) + len(chunk) - 1} ...", flush=True)
            started = time.time()
            prompts = [record["prompt"] for record in chunk]
            if args.backend == "hf":
                texts = generate_batch(
                    tokenizer, model, prompts, args.max_new_tokens, args.prompt_mode, system_prompt
                )
            else:
                texts = generate_remote(remote_config, prompts, args.concurrency)
            seconds = (time.time() - started) / len(chunk)
            for text in texts:
                dump.write(json.dumps({"id": len(students), "student": text}, ensure_ascii=False) + "\n")
                students.append(text)
            dump.flush()
            print(f"  已生成 {len(students)}/{len(samples)} 条，本批 {seconds:.1f}s/条，"
                  f"累计 {(time.time() - started_all):.0f}s", flush=True)
    return students


def read_students(path: str | None, limit: int) -> list[str]:
    """读取落盘的模型输出（供 stage=score 复用，避免重复推理）"""
    if not path:
        raise SystemExit("stage=score 需要 --dump-outputs 指向已有的模型输出文件")
    with open(path, "r", encoding="utf-8") as fin:
        texts = [json.loads(line)["student"] for line in fin]
    if len(texts) < limit:
        raise SystemExit(f"模型输出条数不足：{len(texts)} < {limit}")
    return texts[:limit]


# 每个打分进程内复用的 embedder（懒加载，避免重复载入模型）
_WORKER_EMBEDDER = None


def _new_embedder() -> EmbeddingClient:
    """打分用的 embedder：强制 CPU。

    多进程打分时若让 SentenceTransformer 自动选设备，每个 worker 都会去占 GPU，
    几十个进程会直接把显存打爆，且与 vLLM 抢资源；打分是纯 CPU 活，固定 CPU 最稳。
    """
    return EmbeddingClient(device="cpu")


def _score_one(task: tuple[str, str, str, str]) -> dict:
    """单样本打分（供多进程池调用，必须是模块级函数才可 pickle）"""
    global _WORKER_EMBEDDER
    if _WORKER_EMBEDDER is None:
        _WORKER_EMBEDDER = _new_embedder()
    sample_id, prompt, teacher, student = task
    return evaluate_sample(sample_id, prompt, teacher, student, _WORKER_EMBEDDER).to_dict()


def run_scoring(args, samples: list[dict], students: list[str], tag: str) -> list[dict]:
    """打分阶段：单进程或按 --workers 多进程并行，逐条落盘并打印"""
    tasks = [(str(i), record["prompt"], record["output"], student)
             for i, (record, student) in enumerate(zip(samples, students))]

    payloads: list[dict] = []
    if args.workers > 1:
        print(f"并行打分：workers={args.workers}", flush=True)
        with Pool(processes=args.workers) as pool:
            for payload in pool.imap(_score_one, tasks, chunksize=2):
                payloads.append(payload)
                if len(payloads) % 50 == 0:
                    print(f"  已打分 {len(payloads)}/{len(tasks)}", flush=True)
    else:
        embedder = _new_embedder()
        for sample_id, prompt, teacher, student in tasks:
            payloads.append(evaluate_sample(sample_id, prompt, teacher, student, embedder).to_dict())

    with open(args.out, "w", encoding="utf-8") as fout:
        for i, (payload, student) in enumerate(zip(payloads, students)):
            payload["model"] = tag
            payload["backend"] = args.backend
            payload["student_chars"] = len(student)
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            dims = " ".join(f"{key}={_fmt(payload[key])}" for key in DIMENSIONS if key != "total_score")
            print(f"#{i} total={payload['total_score']:.3f} {dims} "
                  f"grounding={_fmt(payload['details']['grounding_mean'])}", flush=True)
    return payloads


def print_report(results: list[dict], tag: str, args) -> None:
    """汇总均值与通过率：不适用（None）的分项剔除后再求均值，并标注适用条数"""
    full_pass = sum(1 for r in results if r["json_valid"] == 1.0)
    print(f"\n=== {len(results)} 条均值（{tag} / {args.backend}）===")
    for key in DIMENSIONS:
        value, usable = _mean([r[key] for r in results])
        print(f"{key}: {value:.4f}  (适用 {usable}/{len(results)})")
    grounding, g_usable = _mean([r["details"]["grounding_mean"] for r in results])
    print(f"grounding_mean: {grounding:.4f}  (适用 {g_usable}/{len(results)})")
    print(f"json 全项通过条数: {full_pass}/{len(results)}")


def main() -> None:
    args = parse_args()
    system_prompt = resolve_system_prompt(args)
    response_schema = resolve_response_schema(args)
    tag = args.served_model_name or args.model

    samples, scanned = load_samples(args)
    print(f"样本加载：扫描 {scanned} 行，取 {len(samples)} 条（require_schema={args.require_schema}）", flush=True)
    print(f"system_prompt：{'关闭' if system_prompt is None else str(len(system_prompt)) + ' 字符'}", flush=True)
    print(f"response_format：{'关闭' if response_schema is None else 'json_schema(strict) 已启用'}", flush=True)
    print(f"stage={args.stage} backend={args.backend} tag={tag}", flush=True)

    students = None
    if args.stage in ("generate", "both"):
        students = run_generation(args, samples, system_prompt, response_schema)
    if args.stage in ("score", "both"):
        if students is None:
            students = read_students(args.dump_outputs, len(samples))
        results = run_scoring(args, samples, students, tag)
        print_report(results, tag, args)


if __name__ == "__main__":
    main()
