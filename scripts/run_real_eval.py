# @author: ztwz
"""真实数据打分：本地模型推理 N 条样本 + 修复后的评分器打分。

- 模型：unsloth/Qwen3.5-2B（微调底座，作为微调前基线；4bit 量化适配 4GB 显存）
- 数据：data/sample.jsonl 前 N 条（线上真实 prompt + teacher 输出）
- 输出：仅分数与耗时统计，不输出任何样本内容（涉密）
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 强制走本地缓存，避免线上 HEAD 重试

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.evaluation.embeddings import EmbeddingClient
from app.evaluation.scoring.aggregator import evaluate_sample

MODEL_ID = "unsloth/Qwen3.5-2B"
# 控制台逐条打印的分项（grounding_mean 在 details 内，单独处理）
DIMENSIONS = ("total_score", "json_valid", "memory_precision", "memory_recall",
              "speaker_attribution", "summary_score", "intent_score", "other",
              "hallucination_penalty")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="本地模型跑真实数据并用评分器打分")
    p.add_argument("--input", default="data/sample.jsonl")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--out", default="data/real_eval_scores.jsonl")
    return p.parse_args()


def load_model():
    """4bit 量化加载底座模型（4GB 显存约束；全量驻留 GPU，禁止 CPU offload 拖慢推理）"""
    from transformers import BitsAndBytesConfig

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant, device_map={"": 0}
    )
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    """Base 模型原始续写：prompt 原文直接作为前缀，不套对话模板"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    # 只保留续写部分（去掉 prompt 前缀）
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    embedder = EmbeddingClient()
    tokenizer, model = load_model()

    samples = []
    with open(args.input, "r", encoding="utf-8") as fin:
        for line in fin:
            samples.append(json.loads(line))
            if len(samples) >= args.limit:
                break

    results = []
    with open(args.out, "w", encoding="utf-8") as fout:
        for line_no, record in enumerate(samples):
            print(f"#{line_no} generating...", flush=True)
            started = time.time()
            student_text = generate(tokenizer, model, record["prompt"], args.max_new_tokens)
            gen_seconds = time.time() - started
            score = evaluate_sample(str(line_no), record["prompt"], record["output"], student_text, embedder)
            payload = score.to_dict()
            payload["gen_seconds"] = round(gen_seconds, 1)
            payload["student_chars"] = len(student_text)
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            results.append(payload)
            dims = " ".join(f"{key}={payload[key]:.2f}" for key in DIMENSIONS if key != "total_score")
            print(f"#{line_no} total={payload['total_score']:.3f} {dims} "
                  f"grounding={payload['details']['grounding_mean']:.2f} gen={gen_seconds:.0f}s", flush=True)

    print(f"\n=== {len(results)} 条均值 ===")
    for key in DIMENSIONS:
        print(f"{key}: {sum(r[key] for r in results) / len(results):.4f}")
    grounding = [r["details"]["grounding_mean"] for r in results]
    print(f"grounding_mean: {sum(grounding) / len(grounding):.4f}")


if __name__ == "__main__":
    main()
