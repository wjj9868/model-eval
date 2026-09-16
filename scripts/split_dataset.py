# @author: ztwz
"""训练/开发/测试集划分：内容聚类（防同会话跨集泄漏）+ 分层抽样。

为什么不能简单按行随机切分：
- 线上 prompt 是「固定头部 + 唯一内容 + 固定尾部」模板（实测 prefix512 仅 32 个唯一值、
  tail512 仅 50 个唯一值），同一用户的多条记录会共享中段的记忆块/上下文。
- 按行随机切分会让同一用户的记忆块同时出现在训练集与测试集，测试分数被"记忆化"虚高。

做法：
1. **按文档频率剔除模板窗口**：对全文取重叠窗口哈希，出现频率超过 --max-df-ratio 的窗口
   视为模板/样板文字（实测模板窗口 df≈74k），不参与聚类；否则整份数据会并成一个巨簇。
2. 剩余窗口命中相同哈希的记录并查集聚为一簇，**同簇整体落入同一划分**。
3. 分层键取自 teacher（参照）输出：memory 条目数分桶 × 输出字符长度分桶，
   保证测试集"维度适用性"与全量一致（memory 维度仅在有条目时适用）。
4. --force-test-file 中的样本（及其所在簇）强制进测试集，并按原顺序排在测试集最前，
   从而与既有基线分数（同一批样本、同顺序）逐条可比。

只输出统计与校验结果，不打印样本内容。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.output_parser import memory_items, parse_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按内容聚类 + 分层划分训练/开发/测试集")
    p.add_argument("--input", default="data/user_chat_analysis.jsonl", help="全量数据（每行 {prompt, output}）")
    p.add_argument("--out-dir", default="data/split", help="输出目录")
    p.add_argument("--test-size", type=int, default=1000, help="测试集目标条数（含强制样本）")
    p.add_argument("--dev-size", type=int, default=500, help="开发集目标条数（迭代调参用，不参与训练）")
    p.add_argument("--seed", type=int, default=42)
    # 窗口参数：全文重叠窗口 + 文档频率过滤
    p.add_argument("--window", type=int, default=256, help="窗口字符数")
    p.add_argument("--stride", type=int, default=192, help="窗口步长")
    p.add_argument("--max-df-ratio", type=float, default=0.01,
                   help="模板窗口判定阈值（文档频率超过该比例视为模板样板文字）")
    p.add_argument("--rare-df-max", type=int, default=100,
                   help="高特异窗口的文档频率上限：只有出现于 ≤ 该条数的窗口才参与配对")
    p.add_argument("--min-shared-windows", type=int, default=2,
                   help="判定强近重复所需的最小共享高特异窗口数（设 1 会让全部数据传递闭包成巨簇）")
    p.add_argument("--max-key-rows", type=int, default=500,
                   help="单个窗口涉及行数超过该值时跳过配对数（防 O(n²)）")
    p.add_argument("--force-test-file", default="data/eval_holdout_100.jsonl",
                   help="强制进测试集的文件（保持原顺序排最前）；留空则跳过")
    p.add_argument("--report-only", action="store_true", help="只打印聚类统计，不写文件")
    return p.parse_args()


def prompt_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def window_keys(text: str, lo: int, hi: int, window: int, stride: int) -> list[str]:
    """[lo, hi) 区间内的重叠窗口哈希"""
    if hi - lo < window:
        lo, hi = 0, len(text)
    out = []
    for start in range(lo, max(hi - window, lo) + 1, stride):
        seg = text[start:start + window]
        if len(seg) >= window // 2:
            out.append(hashlib.sha1(seg.encode("utf-8")).hexdigest()[:16])
    return out


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # 路径压缩
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def stratum_of(record: dict) -> str:
    """分层键：memory 条目数分桶（决定 memory 维度是否适用）× 输出长度分桶"""
    items = memory_items(parse_output(record["output"]))
    total = sum(len(v) for v in items.values())
    mem = "mem0" if total == 0 else "mem1-2" if total <= 2 else "mem3-5" if total <= 5 else "mem6+"
    n = len(record["output"])
    length = "len-s" if n < 1200 else "len-m" if n < 1800 else "len-l"
    return f"{mem}|{length}"


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    records: list[dict] = []
    with open(args.input, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                records.append(json.loads(line))
    n = len(records)
    print(f"载入 {n} 条（{args.input}）", flush=True)

    # ---------- 1. 窗口 + 文档频率 ----------
    per_row_keys: list[set[str]] = []
    freq: Counter[str] = Counter()
    for record in records:
        keys = set(window_keys(record["prompt"], 0, len(record["prompt"]), args.window, args.stride))
        per_row_keys.append(keys)
        freq.update(keys)

    max_df = max(int(n * args.max_df_ratio), 2)
    template_keys = [k for k, c in freq.items() if c > max_df]
    rare = {k for k, c in freq.items() if 2 <= c <= args.rare_df_max}
    n_rare = sum(1 for keys in per_row_keys if keys & rare)
    print(f"窗口总数 {sum(len(k) for k in per_row_keys)}，唯一 {len(freq)}；"
          f"模板窗口 {len(template_keys)} 个（df 最大 {sorted((freq[k] for k in template_keys), reverse=True)[:2]}，阈值 {max_df}）；"
          f"高特异窗口（df≤{args.rare_df_max}）{len(rare)} 个，覆盖 {n_rare} 条（{n_rare / n:.1%}）", flush=True)

    # ---------- 2. 强近重复配对 → 并查集 ----------
    # 单窗口连边过于宽松（半通用文本会把全量传递闭包成一个巨簇），
    # 故要求一行对至少共享 --min-shared-windows 个高特异窗口才算同源，必须同侧。
    inverted: dict[str, list[int]] = defaultdict(list)
    for i, keys in enumerate(per_row_keys):
        for key in keys & rare:
            inverted[key].append(i)

    pair_hits: Counter[tuple[int, int]] = Counter()
    skipped_keys = 0
    for ids in inverted.values():
        if len(ids) > args.max_key_rows:
            skipped_keys += 1
            continue
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                pair_hits[(ids[a], ids[b])] += 1

    uf = UnionFind(n)
    edges = 0
    for (a, b), shared in pair_hits.items():
        if shared >= args.min_shared_windows:
            uf.union(a, b)
            edges += 1
    max_shared = max(pair_hits.values()) if pair_hits else 0
    print(f"候选对 {len(pair_hits)}，强近重复对（共享≥{args.min_shared_windows}）{edges} 对"
          f"（最大共享窗口数 {max_shared}；跳过超大窗口 {skipped_keys} 个）", flush=True)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)
    sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    multi_rows = sum(s for s in sizes if s > 1)
    buckets = Counter()
    for s in sizes:
        buckets["1" if s == 1 else "2-10" if s <= 10 else "11-100" if s <= 100
                else "101-1000" if s <= 1000 else ">1000"] += 1
    print(f"簇 {len(clusters)} 个（单样本簇 {buckets['1']}）；规模分布 {dict(buckets)}；"
          f"最大簇 {sizes[:5]}", flush=True)
    print(f"落在多成员簇内 {multi_rows} 条（{multi_rows / n:.1%}）", flush=True)

    if args.report_only:
        print("（--report-only：未写文件）", flush=True)
        return

    # ---------- 3. 强制进测试集的样本 ----------
    forced: list[int] = []
    if args.force_test_file and Path(args.force_test_file).exists():
        idx_of_hash = {prompt_hash(r["prompt"]): i for i, r in enumerate(records)}
        with open(args.force_test_file, "r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                h = prompt_hash(json.loads(line)["prompt"])
                if h not in idx_of_hash:
                    raise SystemExit(f"强制样本不在全量数据中：{h[:12]}…")
                forced.append(idx_of_hash[h])
        print(f"强制进测试集 {len(forced)} 条（{args.force_test_file}）", flush=True)

    # ---------- 4. 分层 + 按簇分配 ----------
    cluster_ids = sorted(clusters, key=lambda c: (min(clusters[c]), len(clusters[c])))
    rank = {c: k for k, c in enumerate(cluster_ids)}
    forced_clusters = {uf.find(i) for i in forced}

    strata: dict[str, list[int]] = defaultdict(list)
    for c in cluster_ids:
        if c in forced_clusters:
            continue
        dominant = Counter(stratum_of(records[i]) for i in clusters[c]).most_common(1)[0][0]
        strata[dominant].append(c)
    for v in strata.values():
        v.sort(key=lambda c: rank[c])
        rng.shuffle(v)

    assigned: dict[str, list[int]] = {"test": [], "dev": [], "train": []}
    for c in forced_clusters:
        assigned["test"].extend(clusters[c])
    remaining = n - len(assigned["test"])  # 强制样本之外待分配条数
    ratio_test = max(args.test_size - len(assigned["test"]), 0) / max(remaining, 1)
    ratio_dev = args.dev_size / max(remaining, 1)

    # 每个分层独立按比例分配（若共用全局额度，先遍历到的层会抢光额度，后段层分布失衡）
    for key, cls in sorted(strata.items()):
        total_rows = sum(len(clusters[c]) for c in cls)
        want_test, want_dev = total_rows * ratio_test, total_rows * ratio_dev
        got_test = got_dev = 0
        for c in cls:
            size = len(clusters[c])
            if got_test < want_test:
                assigned["test"].extend(clusters[c])
                got_test += size
            elif got_dev < want_dev:
                assigned["dev"].extend(clusters[c])
                got_dev += size
            else:
                assigned["train"].extend(clusters[c])

    # 测试集顺序：强制样本在前（保持原文件顺序），其余按簇序；train/dev 打散便于 --limit 取样
    forced_set = set(forced)
    test_idx = forced + sorted((i for i in assigned["test"] if i not in forced_set),
                               key=lambda i: rank[uf.find(i)])
    dev_idx = sorted(assigned["dev"], key=lambda i: rank[uf.find(i)])
    train_idx = sorted(assigned["train"], key=lambda i: rank[uf.find(i)])
    rng.shuffle(dev_idx)
    rng.shuffle(train_idx)

    # ---------- 5. 校验：三集互斥（行级 + 簇级） ----------
    sets = {"train": train_idx, "test": test_idx, "dev": dev_idx}
    hashes = {k: {prompt_hash(records[i]["prompt"]) for i in v} for k, v in sets.items()}
    for a, b in (("train", "test"), ("train", "dev"), ("test", "dev")):
        inter = hashes[a] & hashes[b]
        assert not inter, f"{a} 与 {b} 存在重合 {len(inter)} 条"
    assert sum(len(v) for v in sets.values()) == n, "划分后总条数不等于全量"
    cl_of = {k: {uf.find(i) for i in v} for k, v in sets.items()}
    for a, b in (("train", "test"), ("train", "dev"), ("test", "dev")):
        assert not (cl_of[a] & cl_of[b]), f"{a} 与 {b} 存在同簇样本（跨集泄漏）"

    # ---------- 6. 落盘 ----------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_map = {"train": "train.jsonl", "test": "test_1k.jsonl", "dev": "dev.jsonl"}
    for name, idx in sets.items():
        path = out_dir / file_map[name]
        with open(path, "w", encoding="utf-8") as fout:
            for i in idx:
                fout.write(json.dumps(records[i], ensure_ascii=False) + "\n")
        print(f"{name:5s} {len(idx):6d} 条 -> {path}", flush=True)

    # ---------- 7. 分层一致性对比 ----------
    all_counts = Counter(stratum_of(r) for r in records)
    print("\n=== 分层分布对比（该层占各自集合的比例）===")
    print(f"{'分层':16s} {'全量':>8s} {'训练':>8s} {'测试':>8s} {'开发':>8s}")
    for key in sorted(all_counts):
        cells = [f"{sum(1 for i in sets[name] if stratum_of(records[i]) == key) / max(len(sets[name]), 1):.1%}"
                 for name in ("train", "test", "dev")]
        print(f"{key:16s} {all_counts[key] / n:>8.1%} {cells[0]:>8s} {cells[1]:>8s} {cells[2]:>8s}")

    manifest = {
        "input": args.input, "seed": args.seed,
        "sizes": {k: len(v) for k, v in sets.items()},
        "window_params": {"window": args.window, "stride": args.stride,
                          "max_df_ratio": args.max_df_ratio, "max_df": max_df},
        "clusters": {"n_clusters": len(clusters), "size_buckets": dict(buckets),
                     "largest": sizes[:10], "rows_in_multi_member_clusters": multi_rows,
                     "rare_windows": len(rare), "rare_window_rows": n_rare,
                     "strong_pairs": edges, "max_shared_windows": max_shared,
                     "template_windows": len(template_keys)},
        "forced_test": {"file": args.force_test_file, "n": len(forced),
                        "position": "测试集前 N 条，顺序与原文件一致"},
        "stratum_distribution": {
            key: {name: round(sum(1 for i in sets[name] if stratum_of(records[i]) == key)
                              / max(len(sets[name]), 1), 4)
                  for name in ("train", "test", "dev")}
            for key in sorted(all_counts)
        },
        "file_sha1": {file_map[k]: hashlib.sha1((out_dir / file_map[k]).read_bytes()).hexdigest()
                      for k in sets},
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n清单已保存：{out_dir / 'split_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
