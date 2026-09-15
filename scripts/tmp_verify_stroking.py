# @author: ztwz
"""实证 Stroking 记录在 sample.jsonl 中的原始来源。"""
import json

lines = open("data/sample.jsonl", "r", encoding="utf-8").read().splitlines()
hit = None
for line in lines:
    r = json.loads(line)
    if "Stroking" in r.get("output", ""):
        hit = r
        break

if hit is None:
    print("sample.jsonl 中未找到含 Stroking 的记录")
    raise SystemExit(0)

# prompt 里 Stroking 相关上下文（用户信息段 + 会话行 + 聊天行）
print("=== prompt 中与 Stroking 相关的上下文 ===")
for ln in hit["prompt"].splitlines():
    if "Stroking" in ln or ln.startswith("- Name:") or ln.startswith("会话"):
        print(ln)

print("\n=== teacher 输出的 memory facts ===")
out = json.loads(hit["output"])
print(json.dumps(out.get("memory_snapshot", {}).get("facts", []), ensure_ascii=False))

print("\n=== 该条在 sample.jsonl 的总体积/行号 ===")
idx = next(i for i, line in enumerate(lines) if json.loads(line) == hit)
print(f"行号: {idx}, prompt字符数: {len(hit['prompt'])}, output字符数: {len(hit['output'])}")