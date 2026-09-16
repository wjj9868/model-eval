# @author: ztwz
"""序列长度统计：流式 tokenize {prompt, output}，只输出聚合数字，不打印任何数据内容。

用于验证 train_lora 的 --max-seq-len 取值：给出总量在候选截断下的剔除率，
并顺带给出推理侧（vLLM 上下文 = prompt + 生成）的 prompt 长度分布。
模板开销用哑内容实测一次（与训练端 chat 渲染同构），逐条长度 = prompt + output + 开销。

用法：
  .venv/Scripts/python.exe scripts/stat_seq_len.py --input data/user_chat_analysis.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 本地缓存优先，避免联网往返

# 候选截断点（含 8192 旧值与 12288 现值，向上探到 32k）
CUTOFFS = (8192, 10240, 12288, 14336, 16384, 20480, 24576, 32768)


def percentile(sorted_vals: list[int], q: float) -> int:
    """q ∈ [0,1]，取就近分位"""
    idx = min(int(len(sorted_vals) * q), len(sorted_vals) - 1)
    return sorted_vals[idx]


def describe(name: str, vals: list[int]) -> str:
    s = sorted(vals)
    return (f"{name}: n={len(s)} mean={sum(s) / len(s):.0f} "
            f"p50={percentile(s, 0.50)} p90={percentile(s, 0.90)} p95={percentile(s, 0.95)} "
            f"p98={percentile(s, 0.98)} p99={percentile(s, 0.99)} max={s[-1]}")


def main() -> None:
    p = argparse.ArgumentParser(description="序列长度统计（只输出聚合数字，不打印内容）")
    p.add_argument("--input", default="data/user_chat_analysis.jsonl")
    p.add_argument("--model", default="unsloth/Qwen3.5-2B", help="tokenizer 来源（训练同源）")
    p.add_argument("--chunk", type=int, default=1000, help="批编码条数")
    args = p.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # 模板开销：哑内容实测一次，与训练端 chat 渲染同构（Qwen 系 BPE 预切分下边界不跨段合并）
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        tokenize=False, add_generation_prompt=False,
    )
    overhead = (
        len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        - len(tokenizer("x", add_special_tokens=False)["input_ids"])
        - len(tokenizer("y", add_special_tokens=False)["input_ids"])
    )

    prompt_lens: list[int] = []
    output_lens: list[int] = []
    total_lens: list[int] = []
    n_bad = 0
    buf_p: list[str] = []
    buf_o: list[str] = []
    started = time.time()

    def flush() -> None:
        p_ids = tokenizer(buf_p, add_special_tokens=False)["input_ids"]
        o_ids = tokenizer(buf_o, add_special_tokens=False)["input_ids"]
        for pi, oi in zip(p_ids, o_ids):
            pl, ol = len(pi), len(oi)
            prompt_lens.append(pl)
            output_lens.append(ol)
            total_lens.append(pl + ol + overhead)

    with open(args.input, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            buf_p.append(rec["prompt"])
            buf_o.append(rec["output"])
            if len(buf_p) >= args.chunk:
                flush()
                buf_p, buf_o = [], []
                done = len(total_lens)
                if done % 10000 == 0:
                    print(f"  已统计 {done} 条（{time.time() - started:.0f}s）", flush=True)
        if buf_p:
            flush()

    n = len(total_lens)
    print(f"\n=== 序列长度统计（{args.input}，n={n}，坏行 {n_bad}，"
          f"{time.time() - started:.0f}s，模板开销 ≈{overhead} tokens）===")
    print(describe("total", total_lens))
    print(describe("prompt", prompt_lens))
    print(describe("output", output_lens))

    print("\n各截断下 total 超限剔除：")
    for c in CUTOFFS:
        over = sum(1 for t in total_lens if t > c)
        print(f"  >{c:>6}: {over:>6} ({over / n:.2%})")

    print(f"\n全量 tokens ≈ {sum(total_lens):,}（≈{sum(total_lens) / 1e9:.2f}B）")


if __name__ == "__main__":
    main()
