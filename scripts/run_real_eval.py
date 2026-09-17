# @author: ztwz
"""真实数据打分：模型推理 N 条样本 + 评分器打分。

- 后端：--backend hf（本地 transformers）或 openai（vLLM 等 OpenAI 兼容服务）
- 模型：--model 指定（Qwen/Qwen3.5-2B 或 Qwen/Qwen3.5-4B）
- 数据：data/user_chat_analysis.jsonl；--require-schema 只取 prompt 里带完整六字段
  schema 说明的样本（旧模板样本没有 rolling_summary/memory_snapshot 要求，分数不可比）
- 口径（已对照线上源码 UserChatAnalysisAI / OpenRouterTaskCaller 核实）：
  线上 = 单条 user 消息（无 system，任务指令嵌在 prompt 模板头部）+ response_format(json_schema, strict)
  + temperature=1.0（硬编码），默认按此还原；--system-prompt / --no-response-schema 可做消融；
  --prompt-mode raw 走 base 模型原始续写
- 截断降级：finish_reason=length（token 打满仍未停）按调用失败处理（与线上
  callResult.success=false 语义一致），该样本记空文本 → 评分全 0，报告单列截断率；
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

# 线上（UserChatAnalysisAI.java）为单条 user 消息 + response_format(json_schema, strict)，
# 无 system 消息——任务指令嵌在 user prompt 模板头部，导出数据已包含；评测默认对齐，
# 如需 A/B 消融用 --system-prompt / --system-prompt-file 显式注入。

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
    p.add_argument("--dump-observations", default=None,
                   help="观测文件 JSONL：只落盘高风险样本（json_valid<1.0 / intent<0.6 / 截断）的"
                        "关键信息（scores + checks 定位 + teacher/student 对照 + prompt 前300字符），涉密，默认关闭")
    p.add_argument("--require-schema", action="store_true",
                   help="只评测 prompt 含完整六字段 schema 的样本（口径可比）")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="采样温度（线上 OpenRouterTaskCaller 硬编码 1.0，默认对齐）")
    p.add_argument("--seed", type=int, default=42, help="采样种子（A/B 对比可复现）")
    p.add_argument("--system-prompt", default=None,
                   help="A/B 消融：注入 system 消息（线上不存在，默认不发送）")
    p.add_argument("--system-prompt-file", default=None, help="从文件读取 system prompt（优先于 --system-prompt）")
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
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"),
                   help="打分 embedder 设备：默认 cpu（避免与 vLLM 抢显存）；评测生成结束后可传 cuda 加速")
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
                   temperature: float, system_prompt: str | None = None) -> list[str]:
    """批量推理：chat 模式套对话模板（默认与线上同构：无 system），raw 模式原文续写。

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
        temperature=temperature,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = outputs[:, inputs["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    # 截断判定：行内既无 eos 也无 padding = 打满 max_new_tokens 仍未停止
    eos_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id
    results = []
    for row, text in zip(generated, texts):
        stopped = bool((row == eos_id).any()) or (pad_id is not None and bool((row == pad_id).any()))
        results.append({"content": text, "finish_reason": None if stopped else "length"})
    return results


def resolve_system_prompt(args) -> str | None:
    """取 system prompt 内容：默认 None（对齐线上无 system 消息）；
    --system-prompt-file 优先于 --system-prompt，仅供 A/B 消融。"""
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
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        timeout_seconds=int(args.timeout),
        api_key=args.api_key,
        system_prompt=system_prompt,
        response_schema=response_schema,
        seed=args.seed,
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
        "seed": config.seed,  # vLLM 按请求级种子复现采样
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


def generate_remote(config, prompts: list[str], concurrency: int) -> list[dict]:
    """并发调用 OpenAI 兼容服务（vLLM），返回与 prompts 顺序对齐的 {content, finish_reason}"""
    import requests

    url = config.base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else None

    def one(prompt: str) -> dict:
        response = requests.post(
            url, headers=headers, json=chat_payload(config, prompt),
            timeout=(10, config.timeout_seconds),
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        return {"content": choice["message"]["content"], "finish_reason": choice.get("finish_reason")}

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
                   response_schema: dict | None) -> tuple[list[str], list[str | None]]:
    """生成阶段：批量/并发推理，逐条落盘模型原始输出，返回与 samples 对齐的输出与截断标记。

    与打分解耦后，可先集中压满 GPU 出数（stage=generate），再单独并行打分（stage=score）。
    finish_reason=length 按调用失败降级：样本记空文本（评分自然全 0），dump 保留原始截断文本。
    """
    if not args.dump_outputs:
        raise SystemExit("stage=generate/both 需要 --dump-outputs 指定模型输出落盘路径")

    tokenizer = model = remote_config = None
    if args.backend == "hf":
        torch.manual_seed(args.seed)  # A/B 对比可复现（openai 后端由请求级 seed 承担）
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
    finish_reasons: list[str | None] = []
    started_all = time.time()
    with open(args.dump_outputs, "w", encoding="utf-8") as dump:
        for chunk in chunks:
            print(f"生成 #{len(students)}..{len(students) + len(chunk) - 1} ...", flush=True)
            started = time.time()
            prompts = [record["prompt"] for record in chunk]
            if args.backend == "hf":
                results = generate_batch(
                    tokenizer, model, prompts, args.max_new_tokens, args.prompt_mode,
                    args.temperature, system_prompt
                )
            else:
                results = generate_remote(remote_config, prompts, args.concurrency)
            seconds = (time.time() - started) / len(chunk)
            for item in results:
                dump.write(json.dumps({"id": len(students), "student": item["content"],
                                       "finish_reason": item["finish_reason"]},
                                      ensure_ascii=False) + "\n")
                students.append("" if item["finish_reason"] == "length" else item["content"])
                finish_reasons.append(item["finish_reason"])
            dump.flush()
            print(f"  已生成 {len(students)}/{len(samples)} 条，本批 {seconds:.1f}s/条，"
                  f"累计 {(time.time() - started_all):.0f}s", flush=True)
    return students, finish_reasons


def read_students(path: str | None, limit: int) -> tuple[list[str], list[str | None]]:
    """读取落盘的模型输出（供 stage=score 复用，避免重复推理）；截断行同样按失败降级为空文本"""
    students, finish_reasons, _ = load_dump(path, limit)
    return students, finish_reasons


def load_dump(path: str | None, limit: int) -> tuple[list[str], list[str | None], list[str]]:
    """读 dump 文件：返回 (降级 student, finish_reason, 原始输出原文)。

    原始输出保留截断前的完整文本（供观测排查"没输出"vs"被截断"两类失败）；
    降级 student 用于评分（截断按线上语义记空文本 → 全 0）。
    """
    if not path:
        raise SystemExit("stage=score 需要 --dump-outputs 指向已有的模型输出文件")
    students: list[str] = []
    finish_reasons: list[str | None] = []
    raw_students: list[str] = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            row = json.loads(line)
            finish = row.get("finish_reason")
            finish_reasons.append(finish)
            raw_students.append(row["student"])
            students.append("" if finish == "length" else row["student"])
    if len(students) < limit:
        raise SystemExit(f"模型输出条数不足：{len(students)} < {limit}")
    return students[:limit], finish_reasons[:limit], raw_students[:limit]


# 每个打分进程内复用的 embedder（懒加载，避免重复载入模型）
_WORKER_EMBEDDER = None
_EMBED_DEVICE = "cpu"


def _init_worker(device: str) -> None:
    """多进程打分 worker 初始化：接收 embedder 设备参数（spawn 方式下全局变量不继承）"""
    global _EMBED_DEVICE
    _EMBED_DEVICE = device


def _new_embedder() -> EmbeddingClient:
    """打分用的 embedder：默认 CPU。

    默认固定 CPU 的原因：多进程打分时若让 SentenceTransformer 自动选设备，
    每个 worker 都会去占 GPU，几十个进程会直接把显存打爆，且与 vLLM 抢资源。
    需要 GPU 打分时用 --device cuda 显式开启（评测生成结束后 vLLM 空闲，可安全复用）。
    """
    return EmbeddingClient(device=_EMBED_DEVICE)


def _score_one(task: tuple[str, str, str, str]) -> dict:
    """单样本打分（供多进程池调用，必须是模块级函数才可 pickle）"""
    global _WORKER_EMBEDDER
    if _WORKER_EMBEDDER is None:
        _WORKER_EMBEDDER = _new_embedder()
    sample_id, prompt, teacher, student = task
    return evaluate_sample(sample_id, prompt, teacher, student, _WORKER_EMBEDDER).to_dict()


def run_scoring(args, samples: list[dict], students: list[str], tag: str,
                finish_reasons: list[str | None]) -> list[dict]:
    """打分阶段：单进程或按 --workers 多进程并行，逐条落盘并打印"""
    tasks = [(str(i), record["prompt"], record["output"], student)
             for i, (record, student) in enumerate(zip(samples, students))]

    payloads: list[dict] = []
    if args.workers > 1:
        print(f"并行打分：workers={args.workers}", flush=True)
        with Pool(processes=args.workers, initializer=_init_worker, initargs=(args.device,)) as pool:
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
            payload["finish_reason"] = finish_reasons[i]
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
    truncated = sum(1 for r in results if r.get("finish_reason") == "length")
    print(f"截断失败（finish_reason=length，按线上语义计 0 分）: {truncated}/{len(results)}")


def write_observations(path: str, samples: list[dict], students: list[str],
                       finish_reasons: list[str | None], results: list[dict],
                       raw_students: list[str] | None = None) -> None:
    """落盘"任一分项低分或异常"样本的关键信息，供人工逐条排查失败模式。

    风险判定覆盖所有维度（不只 json/intent/截断）：
      json_valid<1 / intent<0.6 / memory P或R<0.6 / speaker<0.6 / summary<0.6 /
      other<0.6 / hallucination_penalty>0.2 / 截断。
    每行含：why（失败原因标记）、scores、checks（json 失败时的 L1 检查明细）、
    prompt_excerpt（前 300 字符）、teacher / student 原文对照，
    以及 student_raw（**原始输出**——截断样本的原文，区分"没输出"vs"被截断"）。
    """
    n_json = n_intent = n_trunc = n_other = 0
    with open(path, "w", encoding="utf-8") as fout:
        for i, (record, student, payload) in enumerate(zip(samples, students, results)):
            json_fail = payload["json_valid"] != 1.0
            intent_low = payload["intent_score"] is not None and payload["intent_score"] < 0.6
            truncated = finish_reasons[i] == "length"
            # 其它分项低分（None 不适用忽略）；幻觉 >0.2 视为高发
            low_other = [
                key for key in ("memory_precision", "memory_recall", "speaker_attribution",
                                "summary_score", "other")
                if payload.get(key) is not None and payload[key] < 0.6
            ] + (["hallucination_high"] if payload.get("hallucination_penalty") is not None
                 and payload["hallucination_penalty"] > 0.2 else [])
            if not (json_fail or intent_low or truncated or low_other):
                continue
            n_json += 1 if json_fail else 0
            n_intent += 1 if intent_low else 0
            n_trunc += 1 if truncated else 0
            n_other += 1 if low_other else 0
            why = [k for k, flag in (("json_invalid", json_fail), ("intent_low", intent_low),
                                     ("truncated", truncated)) if flag] + low_other
            row = {
                "sample_id": i,
                "why": why,
                "scores": {k: payload[k] for k in DIMENSIONS if payload[k] is not None},
                "finish_reason": finish_reasons[i],
                # json 丢分时最有用的定位：具体哪个 L1 结构检查没通过
                "checks": (payload.get("details", {}).get("rule", {}).get("checks")
                           if json_fail else None),
                "prompt_excerpt": record["prompt"][:300],
                "teacher": record["output"],
                "student": student,
                # 原始模型输出（截断样本此处保留截断前全文，评分用 student 为空）
                "student_raw": raw_students[i] if raw_students else student,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"观测文件已保存：{path}（高风险共 {n_json + n_intent + n_trunc + n_other} 条："
          f"json_valid<1: {n_json}，intent<0.6: {n_intent}，截断: {n_trunc}，"
          f"其它低分: {n_other}）", flush=True)


def main() -> None:
    args = parse_args()
    global _EMBED_DEVICE
    _EMBED_DEVICE = args.device
    system_prompt = resolve_system_prompt(args)
    response_schema = resolve_response_schema(args)
    tag = args.served_model_name or args.model

    samples, scanned = load_samples(args)
    print(f"样本加载：扫描 {scanned} 行，取 {len(samples)} 条（require_schema={args.require_schema}）", flush=True)
    print(f"system_prompt：{'关闭（对齐线上）' if system_prompt is None else str(len(system_prompt)) + ' 字符'}", flush=True)
    print(f"response_format：{'关闭' if response_schema is None else 'json_schema(strict) 已启用'}", flush=True)
    print(f"temperature={args.temperature} seed={args.seed}", flush=True)
    print(f"stage={args.stage} backend={args.backend} tag={tag}", flush=True)

    students = None
    finish_reasons: list[str | None] = []
    if args.stage in ("generate", "both"):
        students, finish_reasons = run_generation(args, samples, system_prompt, response_schema)
    if args.stage in ("score", "both"):
        raw_students: list[str] | None = None
        if students is None:
            students, finish_reasons, raw_students = load_dump(args.dump_outputs, len(samples))
        else:
            # 生成后直接打分：dump 已含原始输出，重新读一份供观测使用
            if args.dump_observations and args.dump_outputs:
                _, _, raw_students = load_dump(args.dump_outputs, len(samples))
        results = run_scoring(args, samples, students, tag, finish_reasons)
        print_report(results, tag, args)
        if args.dump_observations:
            write_observations(args.dump_observations, samples, students, finish_reasons, results,
                               raw_students)


if __name__ == "__main__":
    main()
