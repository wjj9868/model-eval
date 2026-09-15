# @author: ztwz
"""评测对比图：把两个模型（如 2B / 4B）在同一评测集上的打分渲染成一张 PNG。

输入为 run_real_eval.py 的 --out 产物（逐条分数 JSONL，含 details.rule.checks）。
输出：多面板对比图 + 控制台关键统计（均值/中位数/分位/通过率）。

用法：
  python scripts/plot_eval_compare.py \
      --scores-a data/scores_2b_1000.jsonl --label-a "Qwen3.5-2B" \
      --scores-b data/scores_4b_1000.jsonl --label-b "Qwen3.5-4B" \
      --out data/eval_2b_vs_4b.png
"""
from __future__ import annotations

import argparse
import json
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")  # 无显示环境必须显式指定后端，故 import 顺序后置
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 中文字体：matplotlib 对 Noto CJK 的 .ttc 只注册首个 face（名字为 "Noto Sans CJK JP"），
# 该 face 含 CJK 统一汉字，可正常渲染简体中文；均缺失时退回默认字体
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Serif CJK JP", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 维度顺序与展示名；note 标明越大越好的方向
DIMENSIONS = [
    ("total_score", "总分"),
    ("json_valid", "JSON 合法"),
    ("memory_precision", "记忆准确率"),
    ("memory_recall", "记忆召回"),
    ("speaker_attribution", "说话人归属"),
    ("summary_score", "摘要质量"),
    ("intent_score", "意图识别"),
    ("other", "其它"),
    ("hallucination_penalty", "幻觉惩罚(越低越好)"),
]
CHECK_NAMES = [
    ("parseable", "可解析"),
    ("fields_complete", "六字段齐全"),
    ("memory_categories_valid", "记忆五分类合法"),
    ("rolling_summary_length_ok", "摘要长度合法"),
    ("memory_no_duplicates", "记忆无重复"),
]


def load_scores(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fin:
        return [json.loads(line) for line in fin]


def dim_value(row: dict, key: str) -> float | None:
    """单样本维度值；None = 该维度不适用（无判定依据），统计时剔除"""
    value = row["details"]["grounding_mean"] if key == "grounding_mean" else row[key]
    return None if value is None else float(value)


def mean_over_applicable(values: list[float | None]) -> float:
    """剔除不适用（None）后的均值；全不适用返回 0"""
    usable = [v for v in values if v is not None]
    return sum(usable) / len(usable) if usable else 0.0


def check_rate(rows: list[dict], check: str) -> float:
    """某项断言通过率；样本连解析都失败时该样本整体计为未通过"""
    passed = 0
    for row in rows:
        checks = row["details"]["rule"].get("checks", {})
        if checks.get(check) is True:
            passed += 1
    return passed / len(rows) if rows else 0.0


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def summarize(rows: list[dict], label: str) -> None:
    """打印单模型关键统计（不适用维度标注适用条数，避免空集按满分/0 混入均值）"""
    totals = [r["total_score"] for r in rows]
    valid = [r["json_valid"] for r in rows]
    print(f"\n=== {label}（n={len(rows)}）===")
    print(f"  总分均值: {mean(totals):.4f}  中位数: {median(totals):.4f}  "
          f"P10: {pct(totals, 10):.4f}  P90: {pct(totals, 90):.4f}")
    print(f"  JSON全通过占比: {sum(1 for v in valid if v == 1.0) / len(valid):.4f}  "
          f"JSON完全无效占比: {sum(1 for v in valid if v == 0.0) / len(valid):.4f}")
    for key, _ in DIMENSIONS:
        values = [dim_value(r, key) for r in rows]
        usable = sum(1 for v in values if v is not None)
        print(f"  维度 {key}: {mean_over_applicable(values):.4f}  (适用 {usable}/{len(rows)})")
    for check, _ in CHECK_NAMES:
        print(f"  断言 {check}: {check_rate(rows, check):.4f}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="渲染两个模型的评测对比图")
    parser.add_argument("--scores-a", required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--scores-b", required=True)
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--out", default="data/eval_compare.png")
    parser.add_argument("--title", default="Qwen3.5 2B vs 4B 评测对比")
    args = parser.parse_args()

    rows_a, rows_b = load_scores(args.scores_a), load_scores(args.scores_b)
    labels = (args.label_a, args.label_b)
    summarize(rows_a, args.label_a)
    summarize(rows_b, args.label_b)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"{args.title}（n={len(rows_a)} vs {len(rows_b)}，线上口径：prompt + response_format json_schema）",
                 fontsize=15)

    # 1) 各维度均值对比
    ax1 = fig.add_subplot(2, 3, 1)
    x = np.arange(len(DIMENSIONS))
    width = 0.38
    means_a = [mean_over_applicable([dim_value(r, k) for r in rows_a]) for k, _ in DIMENSIONS]
    means_b = [mean_over_applicable([dim_value(r, k) for r in rows_b]) for k, _ in DIMENSIONS]
    ax1.bar(x - width / 2, means_a, width, label=labels[0], color="#4C78A8")
    ax1.bar(x + width / 2, means_b, width, label=labels[1], color="#F58518")
    ax1.set_xticks(x)
    ax1.set_xticklabels([name for _, name in DIMENSIONS], rotation=40, ha="right", fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("均值")
    ax1.set_title("各维度均值对比（不适用维度已剔除）")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # 2) 总分分布
    ax2 = fig.add_subplot(2, 3, 2)
    bins = np.linspace(0, 1, 41)
    ax2.hist([r["total_score"] for r in rows_a], bins=bins, alpha=0.6, label=labels[0], color="#4C78A8")
    ax2.hist([r["total_score"] for r in rows_b], bins=bins, alpha=0.6, label=labels[1], color="#F58518")
    ax2.set_xlabel("总分")
    ax2.set_ylabel("样本数")
    ax2.set_title("总分分布")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # 3) json_valid 打分档位占比（四档：0 / 0.25 字段不全 / 0.5 字段全但有缺陷 / 1.0 全通过）
    ax3 = fig.add_subplot(2, 3, 3)
    levels = [0.0, 0.25, 0.5, 1.0]
    shares = []
    for rows in (rows_a, rows_b):
        total = len(rows)
        shares.append([sum(1 for r in rows if r["json_valid"] == lv) / total for lv in levels])
    x3 = np.arange(len(levels))
    ax3.bar(x3 - width / 2, shares[0], width, label=labels[0], color="#4C78A8")
    ax3.bar(x3 + width / 2, shares[1], width, label=labels[1], color="#F58518")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(["0 无效", "0.25 字段不全", "0.5 有缺陷", "1.0 全通过"])
    ax3.set_ylabel("占比")
    ax3.set_title("JSON 结构分档占比")
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    # 4) 规则断言通过率
    ax4 = fig.add_subplot(2, 3, 4)
    x4 = np.arange(len(CHECK_NAMES))
    rates_a = [check_rate(rows_a, k) for k, _ in CHECK_NAMES]
    rates_b = [check_rate(rows_b, k) for k, _ in CHECK_NAMES]
    ax4.bar(x4 - width / 2, rates_a, width, label=labels[0], color="#4C78A8")
    ax4.bar(x4 + width / 2, rates_b, width, label=labels[1], color="#F58518")
    ax4.set_xticks(x4)
    ax4.set_xticklabels([n for _, n in CHECK_NAMES], rotation=30, ha="right", fontsize=8)
    ax4.set_ylim(0, 1.05)
    ax4.set_ylabel("通过率")
    ax4.set_title("L1 规则断言通过率")
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.3)

    # 5) 同一样本两点对比散点（对角线为完全一致）
    ax5 = fig.add_subplot(2, 3, 5)
    n = min(len(rows_a), len(rows_b))
    xs = [rows_a[i]["total_score"] for i in range(n)]
    ys = [rows_b[i]["total_score"] for i in range(n)]
    ax5.scatter(xs, ys, s=10, alpha=0.35, color="#54A24B")
    ax5.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax5.set_xlabel(f"{labels[0]} 总分")
    ax5.set_ylabel(f"{labels[1]} 总分")
    ax5.set_title("同一样本逐条对比（对角线=y 更好）")
    ax5.grid(alpha=0.3)

    # 6) 关键指标条形（均值）
    ax6 = fig.add_subplot(2, 3, 6)
    keys = ["总分", "JSON全通过占比", "JSON完全无效占比"]
    vals_a = [mean([r["total_score"] for r in rows_a]),
              sum(1 for r in rows_a if r["json_valid"] == 1.0) / len(rows_a),
              sum(1 for r in rows_a if r["json_valid"] == 0.0) / len(rows_a)]
    vals_b = [mean([r["total_score"] for r in rows_b]),
              sum(1 for r in rows_b if r["json_valid"] == 1.0) / len(rows_b),
              sum(1 for r in rows_b if r["json_valid"] == 0.0) / len(rows_b)]
    x6 = np.arange(len(keys))
    ax6.bar(x6 - width / 2, vals_a, width, label=labels[0], color="#4C78A8")
    ax6.bar(x6 + width / 2, vals_b, width, label=labels[1], color="#F58518")
    ax6.set_xticks(x6)
    ax6.set_xticklabels(keys)
    ax6.set_ylim(0, 1.05)
    ax6.set_title("关键指标")
    ax6.legend(fontsize=8)
    for xi, (va, vb) in enumerate(zip(vals_a, vals_b)):
        ax6.text(xi - width / 2, va + 0.02, f"{va:.3f}", ha="center", fontsize=8)
        ax6.text(xi + width / 2, vb + 0.02, f"{vb:.3f}", ha="center", fontsize=8)
    ax6.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=150)
    print(f"\n图已保存：{args.out}")


if __name__ == "__main__":
    main()
