# @author: ztwz
"""清洗训练数据：自评过滤 speaker 归因错误样本。

对 data/user_chat_analysis.jsonl（每行 {prompt, output}）逐条做 teacher==student 自评，
仅保留 speaker_attribution == 1.0（无归因错误）且 json_valid == 1.0（结构完整）的样本。
归因错误 = teacher 把人设（AI）发言记成了用户事实，不应让被训练模型学到的坏模式。

- 并行：--workers 进程池分批评分，主进程单写者按原顺序落盘，结果与单进程完全一致。
  单进程逐行喂 embedder 时 GPU 利用率仅 ~5%（等 Python），多进程聚合吞吐约 3-5 倍。
- 报告只写行号与分数，不含样本内容（涉密不外泄）。
- --out/--report 追加写；--offset/--limit 支持分批续跑；重跑先加 --reset 清空旧输出。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

# 允许直接 `python scripts/clean_train_data.py` 运行：把仓库根加入 sys.path
# （Windows spawn 的子进程会重新 import 本模块，此处插入对子进程同样生效）
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
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                   help="embedder 设备：GPU 空闲的训练机用 cuda 提速（默认 cpu，与推理服务共存时用）")
    p.add_argument("--workers", type=int, default=8, help="并行打分进程数（1=进程内直调，无 pickle 开销）")
    p.add_argument("--batch", type=int, default=2000, help="多进程每批喂给池的行数")
    return p.parse_args()


def _round_opt(value: float | None) -> float | None:
    """可选分项保留 None（不适用），仅对数值四舍五入"""
    return None if value is None else round(value, 4)


# worker 内复用的 embedder（懒加载；多进程时每进程各一份）
_WORKER_EMBEDDER: EmbeddingClient | None = None


def _init_worker(device: str) -> None:
    """进程池 initializer：为每个 worker 建一份 embedder。

    父进程不初始化 embedder/CUDA：fork 启动方式下父进程先碰 GPU 会破坏子进程上下文。
    """
    global _WORKER_EMBEDDER
    _WORKER_EMBEDDER = EmbeddingClient(device=device)


def _eval_one(task: tuple[int, str, str]) -> dict:
    """单行自评（teacher==student）：返回报告字段与保留判定，不含样本内容"""
    line_no, prompt, output = task
    if _WORKER_EMBEDDER is None:
        raise RuntimeError("embedder 未初始化（应经 _init_worker 构造）")
    score = evaluate_sample(str(line_no), prompt, output, output, _WORKER_EMBEDDER)
    n_true = n_false = n_none = 0
    for detail in score.details.get("memory_details", []):
        if detail.get("evidence_is_user") is True:
            n_true += 1
        elif detail.get("evidence_is_user") is False:
            n_false += 1
        else:
            n_none += 1
    # speaker/幻觉可能为 None（该维度不适用：无可判定条目 / 无记忆条目）——
    # 不适用不等于有错，按"无归因错误"处理，避免误杀样本
    return {
        "line_no": line_no,
        "speaker_attribution": _round_opt(score.speaker_attribution),
        "hallucination_penalty": _round_opt(score.hallucination_penalty),
        "json_valid": score.json_valid,
        "evidence_user": n_true,
        "evidence_persona": n_false,
        "evidence_unknown": n_none,
        "speaker_ok": score.speaker_attribution is None or score.speaker_attribution >= 1.0,
        "parseable_ok": score.json_valid >= 1.0,
    }


def main() -> None:
    args = parse_args()

    open_mode = "w" if args.reset else "a"
    total = kept = dropped_speaker = dropped_parseable = 0
    bad_json = 0
    done = 0  # 已完成行数（含坏行，进度打印用）
    start = time.time()

    if args.workers > 1:
        pool = Pool(processes=args.workers, initializer=_init_worker, initargs=(args.device,))
    else:
        _init_worker(args.device)
        pool = None

    def flush(batch: list[tuple[int, str, dict]]) -> None:
        """按批评分并按原顺序落盘（主进程单写者，多 worker 不碰文件）"""
        nonlocal kept, dropped_speaker, dropped_parseable, done
        results = (pool.map(_eval_one, [(no, rec["prompt"], rec["output"])
                                        for no, _, rec in batch], chunksize=16)
                   if pool is not None
                   else [_eval_one((no, rec["prompt"], rec["output"])) for no, _, rec in batch])
        for (line_no, line, _), payload in zip(batch, results):
            speaker_ok, parseable_ok = payload["speaker_ok"], payload["parseable_ok"]
            keep = speaker_ok and parseable_ok
            if not speaker_ok:
                dropped_speaker += 1
            if not parseable_ok:
                dropped_parseable += 1
            frep.write(json.dumps({
                "line_no": line_no,
                "speaker_attribution": payload["speaker_attribution"],
                "hallucination_penalty": payload["hallucination_penalty"],
                "json_valid": payload["json_valid"],
                "evidence_user": payload["evidence_user"],
                "evidence_persona": payload["evidence_persona"],
                "evidence_unknown": payload["evidence_unknown"],
                "kept": keep,
            }, ensure_ascii=False) + "\n")
            if keep:
                kept += 1
                fout.write(line)
            done += 1
            if done % args.progress_every == 0:
                elapsed = time.time() - start
                print(f"[{done}] elapsed={elapsed:.0f}s speed={done / elapsed:.1f}rows/s "
                      f"kept={kept} dropped_speaker={dropped_speaker} "
                      f"dropped_parseable={dropped_parseable}", flush=True)

    try:
        with open(args.input, "r", encoding="utf-8") as fin, \
                open(args.out, open_mode, encoding="utf-8") as fout, \
                open(args.report, open_mode, encoding="utf-8") as frep:
            batch: list[tuple[int, str, dict]] = []
            for line_no, line in enumerate(fin):
                if line_no < args.offset:
                    continue
                if args.limit and total >= args.limit:
                    break
                total += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    bad_json += 1
                    done += 1
                    frep.write(json.dumps({"line_no": line_no, "error": "bad_json"},
                                          ensure_ascii=False) + "\n")
                    continue
                batch.append((line_no, line, record))
                if len(batch) >= args.batch:
                    flush(batch)
                    batch = []
            if batch:
                flush(batch)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print("=== 汇总 ===")
    print(f"处理: {total}（坏 JSON {bad_json}；workers={args.workers} device={args.device}）")
    print(f"归因错误剔除: {dropped_speaker}")
    print(f"结构缺陷剔除: {dropped_parseable}")
    print(f"保留: {kept} -> {args.out}")


if __name__ == "__main__":
    main()
