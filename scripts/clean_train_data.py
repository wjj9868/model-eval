# @author: ztwz
"""清洗训练数据：自评过滤 speaker 归因错误样本。

对 data/user_chat_analysis.jsonl（每行 {prompt, output}）逐条做 teacher==student 自评，
仅保留 speaker_attribution == 1.0（无归因错误）且 json_valid == 1.0（结构完整）的样本。
归因错误 = teacher 把人设（AI）发言记成了用户事实，不应让被训练模型学到的坏模式。

- 报告只写行号与分数，不含样本内容（涉密不外泄）。
- --out/--report 追加写；--offset/--limit 支持分批续跑；重跑先加 --reset 清空旧输出。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 允许直接 `python scripts/clean_train_data.py` 运行：把仓库根加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.embeddings import EmbeddingClient  # noqa: E402
from app.evaluation.scoring.aggregator import evaluate_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="自评清洗：过滤 speaker 归因错误训练样本")
    p.add_argument("--input", default="data/user_chat_analysis.jsonl", help="训练数据输入")
    p.add_argument("--out", default="data/train_clean.jsonl", help="保留样本输出（追加，除非 --reset）")
    p.add_argument("--report", default="data/train_clean_report.jsonl", help="每行分数报告（追加，除非 --reset）")
    p.add_argument("--offset", type=int, default=0, help="跳过前 N 行（续跑用）")
    p.add_argument("--limit", type=int, default=0, help="最多处理行数，0=处理到文件尾")
    p.add_argument("--reset", action="store_true", help="重跑：先清空 --out/--report 旧内容")
    p.add_argument("--progress-every", type=int, default=200, help="每 N 行打印一次进度")
    return p.parse_args()


def _round_opt(value: float | None) -> float | None:
    """可选分项保留 None（不适用），仅对数值四舍五入"""
    return None if value is None else round(value, 4)


def _stat_counts(score) -> tuple[int, int, int]:
    """统计 memory 明细中证据说话人分布：用户 / 人设 / 无法判定"""
    n_true = n_false = n_none = 0
    for detail in score.details.get("memory_details", []):
        if detail.get("evidence_is_user") is True:
            n_true += 1
        elif detail.get("evidence_is_user") is False:
            n_false += 1
        else:
            n_none += 1
    return n_true, n_false, n_none


def main() -> None:
    args = parse_args()
    embedder = EmbeddingClient(device="cpu")  # 自评只走 CPU，避免与推理服务抢显存

    open_mode = "w" if args.reset else "a"
    total = kept = dropped_speaker = dropped_parseable = 0
    bad_json = 0
    start = time.time()

    with open(args.input, "r", encoding="utf-8") as fin, \
            open(args.out, open_mode, encoding="utf-8") as fout, \
            open(args.report, open_mode, encoding="utf-8") as frep:
        for line_no, line in enumerate(fin):
            if line_no < args.offset or (args.limit and total >= args.limit):
                continue
            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                frep.write(json.dumps({"line_no": line_no, "error": "bad_json"}, ensure_ascii=False) + "\n")
                continue

            score = evaluate_sample(str(line_no), record["prompt"], record["output"], record["output"], embedder)
            n_true, n_false, n_none = _stat_counts(score)

            # speaker/幻觉可能为 None（该维度不适用：无可判定条目 / 无记忆条目）——
            # 不适用不等于有错，按"无归因错误"处理，避免误杀样本
            speaker_ok = score.speaker_attribution is None or score.speaker_attribution >= 1.0
            parseable_ok = score.json_valid >= 1.0
            keep = speaker_ok and parseable_ok
            if not speaker_ok:
                dropped_speaker += 1
            if not parseable_ok:
                dropped_parseable += 1

            frep.write(json.dumps({
                "line_no": line_no,
                "speaker_attribution": _round_opt(score.speaker_attribution),
                "hallucination_penalty": _round_opt(score.hallucination_penalty),
                "json_valid": score.json_valid,
                "evidence_user": n_true,
                "evidence_persona": n_false,
                "evidence_unknown": n_none,
                "kept": keep,
            }, ensure_ascii=False) + "\n")

            if keep:
                kept += 1
                fout.write(line)

            if total % args.progress_every == 0:
                elapsed = time.time() - start
                print(f"[{total}] elapsed={elapsed:.0f}s speed={total / elapsed:.1f}rows/s "
                      f"kept={kept} dropped_speaker={dropped_speaker} dropped_parseable={dropped_parseable}",
                      flush=True)

    print("=== 汇总 ===")
    print(f"处理: {total}（坏 JSON {bad_json}）")
    print(f"归因错误剔除: {dropped_speaker}")
    print(f"结构缺陷剔除: {dropped_parseable}")
    print(f"保留: {kept} -> {args.out}")


if __name__ == "__main__":
    main()
