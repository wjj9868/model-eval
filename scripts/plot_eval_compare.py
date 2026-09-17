# @author: ztwz
"""评测对比图：把任意多个模型在同一评测集上的打分渲染成一张综合分析 PNG。

输入为 run_real_eval.py 的 --out 产物（逐条分数 JSONL，含 details.rule.checks）。
输出：多面板对比图 + 控制台关键统计（均值/中位数/分位/通过率/逐条胜率）。

**样本量对齐（重要）**：不同模型可能只在不同条数上评过（如基础模型 100 条、
训练后模型 1000 条）。逐条可比的前提是"同一评测集、同一评分器版本、同样的行序"，
且短序列必须是长序列的前缀（`split_dataset.py --force-test-file` 已保证
评测集前 N 条即旧基线样本）。脚本默认 `--align common`：图内所有面板只取各曲线的
公共前缀（n = min），避免"100 条曲线"和"1000 条曲线"混在一张图里比较；
控制台的"全量口径"表仍按各自全部样本输出，供单独看规模效应。

用法：
  # 旧接口（2~3 条曲线，保持向后兼容）
  python scripts/plot_eval_compare.py \
      --scores-a data/scores_2b_100_v4.jsonl --label-a "Qwen3.5-2B" \
      --scores-b data/scores_4b_100_v4.jsonl --label-b "Qwen3.5-4B" \
      --out data/eval_2b_vs_4b_v4.png

  # 新接口：任意多条曲线（2B/4B 基础模型 + smoke + 10k 训练模型 + teacher 上限）
  python scripts/plot_eval_compare.py \
      --series "Qwen3.5-2B=data/scores_2b_100_v4.jsonl" \
      --series "Qwen3.5-4B=data/scores_4b_100_v4.jsonl" \
      --series "训练后 smoke=data/real_eval_scores.jsonl" \
      --series "训练后 10k=data/real_eval_10k_scores.jsonl" \
      --series "teacher 上限=data/teacher_oracle_scores.jsonl" \
      --baseline 0 \
      --out data/eval_full_compare.png

注意：多条曲线必须来自同一评测集与同一评分器版本，否则分数不可比（逐条面板按行号对齐）。
"""
from __future__ import annotations

import argparse
import json
import math
from statistics import mean, median, stdev

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

# 曲线配色（按顺序循环取用）：蓝 / 橙 / 绿 / 红 / 青 / 紫
SERIES_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]

# 维度顺序与展示名；hallucination_penalty 为惩罚项（越低越好），Grounding 证据均值越高越好
DIMENSIONS = [
    ("total_score", "总分"),
    ("json_valid", "JSON 合法"),
    ("memory_precision", "记忆准确率"),
    ("memory_recall", "记忆召回"),
    ("speaker_attribution", "说话人归属"),
    ("summary_score", "摘要质量"),
    ("intent_score", "意图匹配"),
    ("other", "其它"),
    ("hallucination_penalty", "幻觉惩罚↓"),
]
CHECK_NAMES = [
    ("parseable", "可解析"),
    ("fields_complete", "六字段齐全"),
    ("memory_categories_valid", "记忆五分类合法"),
    ("rolling_summary_length_ok", "摘要长度合法"),
    ("memory_no_duplicates", "记忆无重复"),
]
# 惩罚项：越大越差（画图与差值面板需反向理解）
PENALTY_DIMS = {"hallucination_penalty"}
# 可选维度（None = 不适用）：用于"适用率"热力图，暴露"空记忆白拿分"的覆盖面问题
OPTIONAL_DIMS = ["memory_precision", "memory_recall", "speaker_attribution",
                 "hallucination_penalty", "grounding_mean"]


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


def ci95(values: list[float]) -> float:
    """均值的 95% 置信半宽（正态近似）；n<2 无法估计返回 0"""
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def trunc_rate(rows: list[dict]) -> float:
    """截断失败率：finish_reason=length 按线上语义计调用失败（评分已降级为 0）"""
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.get("finish_reason") == "length") / len(rows)


def json_share(rows: list[dict], level: float) -> float:
    return sum(1 for r in rows if r["json_valid"] == level) / len(rows) if rows else 0.0


def oriented(key: str, value: float) -> float:
    """统一成"越大越好"方向（惩罚项取补），供差值/胜率类面板使用"""
    return 1.0 - value if key in PENALTY_DIMS else value


def summarize(rows: list[dict], label: str) -> None:
    """打印单模型关键统计（不适用维度标注适用条数，避免空集按满分/0 混入均值）"""
    totals = [r["total_score"] for r in rows]
    valid = [r["json_valid"] for r in rows]
    print(f"\n=== {label}（n={len(rows)}）===")
    print(f"  总分均值: {mean(totals):.4f} ±{ci95(totals):.4f}(95%CI)  中位数: {median(totals):.4f}  "
          f"P10: {pct(totals, 10):.4f}  P90: {pct(totals, 90):.4f}")
    print(f"  JSON全通过占比: {sum(1 for v in valid if v == 1.0) / len(valid):.4f}  "
          f"JSON完全无效占比: {sum(1 for v in valid if v == 0.0) / len(valid):.4f}  "
          f"截断率: {trunc_rate(rows):.4f}")
    for key, _ in DIMENSIONS:
        values = [dim_value(r, key) for r in rows]
        usable = sum(1 for v in values if v is not None)
        print(f"  维度 {key}: {mean_over_applicable(values):.4f}  (适用 {usable}/{len(rows)})")
    for check, _ in CHECK_NAMES:
        print(f"  断言 {check}: {check_rate(rows, check):.4f}")


def grouped_bars(ax, x, series_values: list[list[float]], labels: list[str],
                 ylim: float | None = None) -> None:
    """多曲线分组柱：每组 N 根，围绕刻度居中排布"""
    n = len(series_values)
    width = 0.8 / n
    for i, (values, label) in enumerate(zip(series_values, labels)):
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, values, width, label=label,
               color=SERIES_COLORS[i % len(SERIES_COLORS)])
    if ylim is not None:
        ax.set_ylim(0, ylim)


def win_rate_matrix(rows_list: list[list[dict]]) -> np.ndarray:
    """两两逐条胜率矩阵：m[i][j] = 曲线 i 在公共前缀内总分严格高于 j 的样本占比"""
    n = len(rows_list)
    matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a = [r["total_score"] for r in rows_list[i]]
            b = [r["total_score"] for r in rows_list[j]]
            k = min(len(a), len(b))
            wins = sum(1 for t in range(k) if a[t] > b[t])
            matrix[i, j] = wins / k if k else 0.0
    return matrix


def render(series: list[dict], labels: list[str], aligned: list[list[dict]],
           baseline: int, args) -> None:
    """渲染综合分析图：1~14 面板用公共前缀对齐口径，15 表用各自全量，16 为读图说明"""
    n_common = len(aligned[0])
    colors = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(labels))]
    sizes = " / ".join(f"{s['label']}={len(s['rows'])}" for s in series)

    fig = plt.figure(figsize=(24, 18))
    fig.suptitle(
        f"{args.title}\n各曲线样本量：{sizes}；图内面板统一取公共前缀 n={n_common} 对齐",
        fontsize=16,
    )

    # 1) 各维度均值对比
    ax1 = fig.add_subplot(4, 4, 1)
    x = np.arange(len(DIMENSIONS))
    means = [[mean_over_applicable([dim_value(r, k) for r in rows]) for k, _ in DIMENSIONS]
             for rows in aligned]
    grouped_bars(ax1, x, means, labels, ylim=1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels([name for _, name in DIMENSIONS], rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("均值")
    ax1.set_title("① 各维度均值（不适用维度已剔除）")
    ax1.legend(fontsize=7)
    ax1.grid(axis="y", alpha=0.3)

    # 2) 总分均值 ± 95% CI
    ax2 = fig.add_subplot(4, 4, 2)
    totals = [[r["total_score"] for r in rows] for rows in aligned]
    vals = [mean(t) for t in totals]
    errs = [ci95(t) for t in totals]
    ax2.bar(range(len(labels)), vals, yerr=errs, capsize=4, color=colors)
    for i, (value, err) in enumerate(zip(vals, errs)):
        ax2.text(i, value + err + 0.015, f"{value:.3f}", ha="center", fontsize=8)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("总分均值")
    ax2.set_title(f"② 总分均值±95%CI（n={n_common}）")
    ax2.grid(axis="y", alpha=0.3)

    # 3) 总分 ECDF（比直方图更易看尾部与超越点）
    ax3 = fig.add_subplot(4, 4, 3)
    for i, (t, label) in enumerate(zip(totals, labels)):
        ordered = np.sort(t)
        ax3.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
                 where="post", label=label, color=colors[i])
    ax3.set_xlabel("总分")
    ax3.set_ylabel("累积占比")
    ax3.set_title("③ 总分 ECDF")
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)

    # 4) 总分箱线图（看中位数与离散度）
    ax4 = fig.add_subplot(4, 4, 4)
    # 不传 labels=（matplotlib 3.9 起已弃用），刻度标签单独设置，兼容新旧版本
    bp = ax4.boxplot(totals, patch_artist=True, showfliers=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median_line in bp["medians"]:
        median_line.set_color("black")
    ax4.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax4.set_ylabel("总分")
    ax4.set_title("④ 总分分布箱线")
    ax4.grid(axis="y", alpha=0.3)

    # 5) json_valid 四档占比（0 / 0.25 字段不全 / 0.5 有缺陷 / 1.0 全通过）
    ax5 = fig.add_subplot(4, 4, 5)
    levels = [0.0, 0.25, 0.5, 1.0]
    level_names = ["0 无效", "0.25 字段不全", "0.5 有缺陷", "1.0 全通过"]
    level_colors = ["#E45756", "#F58518", "#F2CF5B", "#54A24B"]
    bottom = np.zeros(len(labels))
    xs = np.arange(len(labels))
    for level, name, color in zip(levels, level_names, level_colors):
        shares = np.array([json_share(rows, level) for rows in aligned])
        ax5.bar(xs, shares, bottom=bottom, label=name, color=color)
        bottom += shares
    ax5.set_xticks(xs)
    ax5.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax5.set_ylim(0, 1.0)
    ax5.set_ylabel("占比")
    ax5.set_title("⑤ JSON 结构分档占比")
    ax5.legend(fontsize=7, loc="lower right")
    ax5.grid(axis="y", alpha=0.3)

    # 6) 规则断言通过率
    ax6 = fig.add_subplot(4, 4, 6)
    x6 = np.arange(len(CHECK_NAMES))
    rates = [[check_rate(rows, k) for k, _ in CHECK_NAMES] for rows in aligned]
    grouped_bars(ax6, x6, rates, labels, ylim=1.05)
    ax6.set_xticks(x6)
    ax6.set_xticklabels([n for _, n in CHECK_NAMES], rotation=30, ha="right", fontsize=8)
    ax6.set_ylabel("通过率")
    ax6.set_title("⑥ L1 规则断言通过率")
    ax6.legend(fontsize=7)
    ax6.grid(axis="y", alpha=0.3)

    # 7) 关键指标：总分均值 / JSON 全通过 / 截断率
    ax7 = fig.add_subplot(4, 4, 7)
    keys = ["总分均值", "JSON 全通过", "截断率"]
    kvals = [[mean([r["total_score"] for r in rows]),
              json_share(rows, 1.0),
              trunc_rate(rows)] for rows in aligned]
    x7 = np.arange(len(keys))
    grouped_bars(ax7, x7, kvals, labels, ylim=1.12)
    ax7.set_xticks(x7)
    ax7.set_xticklabels(keys)
    ax7.set_title("⑦ 关键指标")
    ax7.legend(fontsize=7)
    width = 0.8 / len(labels)
    for i, series_vals in enumerate(kvals):
        offset = (i - (len(labels) - 1) / 2) * width
        for xi, value in enumerate(series_vals):
            ax7.text(xi + offset, value + 0.02, f"{value:.2f}", ha="center", fontsize=7)
    ax7.grid(axis="y", alpha=0.3)

    # 8) 幻觉惩罚（越低越好）与 grounding 证据均值（越高越好）
    ax8 = fig.add_subplot(4, 4, 8)
    penalty = [mean_over_applicable([dim_value(r, "hallucination_penalty") for r in rows])
               for rows in aligned]
    grounding = [mean_over_applicable([dim_value(r, "grounding_mean") for r in rows])
                 for rows in aligned]
    x8 = np.arange(2)
    # 分组柱：每组 2 根（幻觉惩罚率 / grounding 均值），曲线作为颜色
    width8 = 0.8 / len(labels)
    for i, label in enumerate(labels):
        offset = (i - (len(labels) - 1) / 2) * width8
        ax8.bar(x8 + offset, [penalty[i], grounding[i]], width8, label=label, color=colors[i])
    for i in range(len(labels)):
        offset = (i - (len(labels) - 1) / 2) * width8
        ax8.text(0 + offset, penalty[i] + 0.02, f"{penalty[i]:.2f}", ha="center", fontsize=7)
        ax8.text(1 + offset, grounding[i] + 0.02, f"{grounding[i]:.2f}", ha="center", fontsize=7)
    ax8.set_xticks(x8)
    ax8.set_xticklabels(["幻觉惩罚↓", "grounding↑"])
    ax8.set_ylim(0, 1.12)
    ax8.set_title("⑧ 幻觉惩罚与证据均值")
    ax8.legend(fontsize=7)
    ax8.grid(axis="y", alpha=0.3)

    # 9) 同一样本逐条对比散点：以基线为横轴，对角线=持平，点在对角线上方=该曲线更好
    ax9 = fig.add_subplot(4, 4, 9)
    base_totals = totals[baseline]
    for i, label in enumerate(labels):
        if i == baseline:
            continue
        ax9.scatter(base_totals, totals[i], s=10, alpha=0.35, label=f"{label} vs {labels[baseline]}",
                    color=colors[i])
    ax9.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax9.set_xlabel(f"{labels[baseline]} 总分")
    ax9.set_ylabel("各模型总分")
    ax9.set_title(f"⑨ 同一样本逐条对比（相对 {labels[baseline]}）")
    ax9.legend(fontsize=7)
    ax9.grid(alpha=0.3)

    # 10) 两两逐条胜率热力图
    ax10 = fig.add_subplot(4, 4, 10)
    matrix = win_rate_matrix(aligned)
    im = ax10.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax10.set_xticks(range(len(labels)))
    ax10.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax10.set_yticks(range(len(labels)))
    ax10.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                ax10.text(j, i, "—", ha="center", va="center", fontsize=9)
            else:
                ax10.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax10.set_title("⑩ 逐条胜率（行>列 的样本占比）")
    fig.colorbar(im, ax=ax10, fraction=0.046)

    # 11) 各维度相对基线的差值（已统一为"正=更好"）
    ax11 = fig.add_subplot(4, 4, 11)
    delta_dims = [k for k, _ in DIMENSIONS]
    base_means = [mean_over_applicable([oriented(k, v) for v in
                                        [dim_value(r, k) for r in aligned[baseline]] if v is not None])
                  for k in delta_dims]
    x11 = np.arange(len(delta_dims))
    width11 = 0.8 / max(1, len(labels) - 1)
    others = [i for i in range(len(labels)) if i != baseline]
    for slot, i in enumerate(others):
        diffs = []
        for k in delta_dims:
            values = [dim_value(r, k) for r in aligned[i]]
            m = mean_over_applicable([oriented(k, v) for v in values if v is not None])
            diffs.append(m - base_means[delta_dims.index(k)])
        offset = (slot - (len(others) - 1) / 2) * width11
        ax11.bar(x11 + offset, diffs, width11, label=f"{labels[i]} - {labels[baseline]}", color=colors[i])
    ax11.axhline(0, color="black", linewidth=0.8)
    ax11.set_xticks(x11)
    ax11.set_xticklabels([name for _, name in DIMENSIONS], rotation=40, ha="right", fontsize=8)
    ax11.set_ylabel("差值（正=更好）")
    ax11.set_title("⑪ 各维度相对基线之差（幻觉已取补）")
    ax11.legend(fontsize=7)
    ax11.grid(axis="y", alpha=0.3)

    # 12) 相对基线的逐条总分差值分布
    ax12 = fig.add_subplot(4, 4, 12)
    bins12 = np.linspace(-1, 1, 41)
    for i in others:
        deltas = [totals[i][t] - base_totals[t] for t in range(n_common)]
        ax12.hist(deltas, bins=bins12, alpha=0.6, label=f"{labels[i]} - {labels[baseline]}",
                  color=colors[i])
    ax12.axvline(0, color="black", linewidth=0.8)
    ax12.set_xlabel("Δ总分")
    ax12.set_ylabel("样本数")
    ax12.set_title(f"⑫ Δ总分分布（正=优于 {labels[baseline]}）")
    ax12.legend(fontsize=7)
    ax12.grid(axis="y", alpha=0.3)

    # 13) 仅结构合格样本（json_valid≥0.5）的语义维度对比：剥离"格式崩掉"的影响
    ax13 = fig.add_subplot(4, 4, 13)
    valid_rows = [[r for r in rows if r["json_valid"] >= 0.5] for rows in aligned]
    sem_keys = ["memory_precision", "memory_recall", "speaker_attribution",
                "summary_score", "intent_score", "other"]
    x13 = np.arange(len(sem_keys))
    sem_means = [[mean_over_applicable([dim_value(r, k) for r in rows]) for k in sem_keys]
                 for rows in valid_rows]
    grouped_bars(ax13, x13, sem_means, labels, ylim=1.05)
    ax13.set_xticks(x13)
    ax13.set_xticklabels([name for k, name in DIMENSIONS if k in sem_keys],
                         rotation=30, ha="right", fontsize=8)
    ax13.set_ylabel("均值")
    counts13 = "/".join(str(len(rows)) for rows in valid_rows)
    ax13.set_title(f"⑬ 结构合格样本的语义维度（样本数 {counts13}）")
    ax13.legend(fontsize=7)
    ax13.grid(axis="y", alpha=0.3)

    # 14) 维度适用率热力图（None 占比 → 1-适用率；空记忆样本会拉低适用率）
    ax14 = fig.add_subplot(4, 4, 14)
    applicable = np.array([
        [sum(1 for r in rows if dim_value(r, k) is not None) / len(rows) for k in OPTIONAL_DIMS]
        for rows in aligned
    ])
    im14 = ax14.imshow(applicable, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax14.set_xticks(range(len(OPTIONAL_DIMS)))
    ax14.set_xticklabels([name for k, name in DIMENSIONS if k in OPTIONAL_DIMS] + ["证据均值↑"],
                         rotation=25, ha="right", fontsize=8)
    ax14.set_yticks(range(len(labels)))
    ax14.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(OPTIONAL_DIMS)):
            ax14.text(j, i, f"{applicable[i, j]:.2f}", ha="center", va="center",
                      fontsize=8, color="white" if applicable[i, j] < 0.6 else "black")
    ax14.set_title("⑭ 维度适用率（越高=越多样本有判定依据）")
    fig.colorbar(im14, ax=ax14, fraction=0.046)

    # 15) 全量口径汇总表（各自全部样本，非对齐）
    ax15 = fig.add_subplot(4, 4, 15)
    ax15.axis("off")
    table_metrics = [
        ("n", lambda rs: f"{len(rs)}"),
        ("总分均值", lambda rs: f"{mean([r['total_score'] for r in rs]):.4f}"),
        ("总分中位数", lambda rs: f"{median([r['total_score'] for r in rs]):.4f}"),
        ("总分P10", lambda rs: f"{pct([r['total_score'] for r in rs], 10):.4f}"),
        ("JSON 全通过率", lambda rs: f"{json_share(rs, 1.0):.3f}"),
        ("JSON 无效率", lambda rs: f"{json_share(rs, 0.0):.3f}"),
        ("截断率", lambda rs: f"{trunc_rate(rs):.3f}"),
        ("记忆准确率", lambda rs: f"{mean_over_applicable([dim_value(r, 'memory_precision') for r in rs]):.3f}"),
        ("记忆召回", lambda rs: f"{mean_over_applicable([dim_value(r, 'memory_recall') for r in rs]):.3f}"),
        ("说话人归属", lambda rs: f"{mean_over_applicable([dim_value(r, 'speaker_attribution') for r in rs]):.3f}"),
        ("摘要质量", lambda rs: f"{mean_over_applicable([dim_value(r, 'summary_score') for r in rs]):.3f}"),
        ("意图匹配", lambda rs: f"{mean_over_applicable([dim_value(r, 'intent_score') for r in rs]):.3f}"),
        ("其它", lambda rs: f"{mean_over_applicable([dim_value(r, 'other') for r in rs]):.3f}"),
        ("幻觉惩罚↓", lambda rs: f"{mean_over_applicable([dim_value(r, 'hallucination_penalty') for r in rs]):.3f}"),
        ("证据均值↑", lambda rs: f"{mean_over_applicable([dim_value(r, 'grounding_mean') for r in rs]):.3f}"),
    ]
    cell_text = [[fn(s["rows"]) for s in series] for _, fn in table_metrics]
    table = ax15.table(cellText=cell_text,
                       rowLabels=[name for name, _ in table_metrics],
                       colLabels=labels,
                       cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.15)
    ax15.set_title("⑮ 全量口径汇总（各曲线自己的全部样本）", fontsize=10)

    # 16) 口径与读图说明
    ax16 = fig.add_subplot(4, 4, 16)
    ax16.axis("off")
    notes = [
        "口径：prompt 内嵌任务指令 + response_format json_schema(strict) + temperature=1.0",
        "评分：L1 规则(json_valid 四档) + L2 语义（记忆 P/R、归因、幻觉、摘要、意图、其它）加权",
        "权重：记忆P .18 / 记忆R .18 / 归因 .14 / 摘要 .15 / 意图 .10 / JSON .10 / 幻觉 .08 / 其它 .07",
        "None 语义：该维度无判定依据时剔除并重新归一化权重（不是按 0 计）",
        f"对齐：1~14 面板仅用公共前缀 n={n_common}；⑮ 用各自全量样本，两者不可混读",
        "方向：幻觉惩罚↓、grounding 证据均值↑；⑪ 已把幻觉取补，统一为“正=更好”",
        "截断：finish_reason=length 按调用失败降级（评分全 0），是本流程唯一残余失败模式",
        "读图重点：先看 ⑤/⑦ 结构层（能否稳定产出结构），再看 ⑬ 语义层（真变聪明了吗）",
        "",
        "可比性提醒：基础模型若未施加结构化输出约束，其 json_valid 与 L2 分数会被",
        "“解析失败→全 0”主导，与训练后模型不是同一口径；此时应以 ⑬（结构合格子集）为准。",
    ]
    ax16.text(0.0, 0.98, "\n".join(notes), va="top", ha="left", fontsize=8.5, family="monospace")

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(args.out, dpi=130)
    print(f"\n图已保存：{args.out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="渲染多个模型的评测对比图")
    # 旧接口（向后兼容）
    parser.add_argument("--scores-a", default=None)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--scores-b", default=None)
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--scores-c", default=None, help="可选的第三条曲线（如蒸馏后模型）")
    parser.add_argument("--label-c", default="C")
    # 新接口：任意多条曲线（与旧接口二选一，二者同时给出时以 --series 为准）
    parser.add_argument("--series", action="append", default=None, metavar="LABEL=PATH",
                        help="可重复：曲线标签=分数 JSONL 路径")
    parser.add_argument("--baseline", type=int, default=0, help="逐条对比/差值的基线曲线下标（默认 0）")
    parser.add_argument("--align", choices=("common", "none"), default="common",
                        help="common: 面板只取各曲线公共前缀（推荐）；none: 各用全部样本")
    parser.add_argument("--out", default="data/eval_compare.png")
    parser.add_argument("--title", default="Qwen3.5 2B vs 4B 评测对比")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.series:
        pairs: list[tuple[str, str]] = []
        for spec in args.series:
            if "=" not in spec:
                raise SystemExit(f"--series 需要 LABEL=PATH 形式，收到：{spec}")
            label, path = spec.split("=", 1)
            pairs.append((label.strip(), path.strip()))
    else:
        if not args.scores_a or not args.scores_b:
            raise SystemExit("需要 --series，或同时提供 --scores-a / --scores-b")
        pairs = [(args.label_a, args.scores_a), (args.label_b, args.scores_b)]
        if args.scores_c:
            pairs.append((args.label_c, args.scores_c))

    series = [{"label": label, "path": path, "rows": load_scores(path)} for label, path in pairs]
    labels = [s["label"] for s in series]

    if not 0 <= args.baseline < len(series):
        raise SystemExit(f"--baseline={args.baseline} 越界（共 {len(series)} 条曲线）")

    for s in series:
        summarize(s["rows"], f"{s['label']}｜{s['path']}")

    # 公共前缀对齐：短序列必须是长序列前缀才可比，这里按行号截断并显式声明
    n_common = min(len(s["rows"]) for s in series)
    lengths = {len(s["rows"]) for s in series}
    if args.align == "common":
        aligned = [s["rows"][:n_common] for s in series]
        if len(lengths) > 1:
            print(f"\n[对齐] 各曲线样本量不一致 {sorted(lengths)}，图内面板统一取前 {n_common} 条"
                  f"（需确认短序列是长序列的前缀，order 一致才可比）")
    else:
        aligned = [s["rows"] for s in series]
        n_common = min(lengths)
        print(f"\n[对齐] --align none：各曲线用各自全部样本 {sorted(lengths)}；"
              f"逐条面板仍按前 {n_common} 条对齐")

    # 逐条胜率矩阵的文本版（控制台）便于快速抓结论
    print("\n=== 逐条胜率矩阵（行 > 列 的样本占比，公共前缀对齐）===")
    matrix = win_rate_matrix(aligned)
    print("       " + "".join(f"{label[:10]:>12}" for label in labels))
    for i, label in enumerate(labels):
        cells = "".join(
            f"{'—':>12}" if i == j else f"{matrix[i, j]:>12.3f}" for j in range(len(labels))
        )
        print(f"{label[:6]:<6}{cells}")

    render(series, labels, aligned, args.baseline, args)


if __name__ == "__main__":
    main()
