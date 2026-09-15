# @author: ztwz
"""Level 1 规则评分：纯代码零成本，检查 JSON 合法性与结构约束。

json_valid 分级：0 = 无法解析出 JSON 对象；0.5 = 可解析但结构有缺陷；1.0 = 完整合法。
明细写入 checks，供报告定位失败原因。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.evaluation.output_parser import (
    MEMORY_CATEGORIES,
    MEMORY_CATEGORY_LIMITS,
    REQUIRED_FIELDS,
    AnalysisResult,
    memory_items,
    normalize_text,
)

# rolling_summary 硬上限（与后端 SUMMARY_MAX_LENGTH 对齐）
ROLLING_SUMMARY_MAX_CHARS = 2000


@dataclass
class RuleResult:
    """L1 评分结果：json_valid 分数 + 各检查项明细"""

    json_valid: float
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"json_valid": self.json_valid, "checks": self.checks}


def score_rules(student: AnalysisResult) -> RuleResult:
    """L1 规则评分。无法解析直接 0 分；可解析则逐项检查字段/分类/长度/重复。"""
    if not student.valid:
        return RuleResult(0.0, {"parseable": False})

    checks: dict[str, bool] = {"parseable": True}
    checks["fields_complete"] = all(name in student.raw for name in REQUIRED_FIELDS)
    checks["memory_categories_valid"] = _memory_categories_valid(student)
    checks["rolling_summary_length_ok"] = _rolling_summary_ok(student)
    checks["memory_no_duplicates"] = _memory_no_duplicates(student)

    # 全部通过 = 1.0；可解析但有缺陷 = 0.5
    json_valid = 1.0 if all(checks.values()) else 0.5
    return RuleResult(json_valid, checks)


def _memory_categories_valid(student: AnalysisResult) -> bool:
    """memory_snapshot 必须为 dict 且只含合法分类；分类缺失视为缺陷"""
    snapshot = student.raw.get("memory_snapshot")
    if not isinstance(snapshot, dict):
        return False
    if set(snapshot) - set(MEMORY_CATEGORIES):
        return False
    return all(category in snapshot for category in MEMORY_CATEGORIES)


def _rolling_summary_ok(student: AnalysisResult) -> bool:
    """rolling_summary 必须为非空字符串且不超硬上限"""
    summary = student.raw.get("rolling_summary")
    if not isinstance(summary, str) or not summary.strip():
        return False
    return len(summary) <= ROLLING_SUMMARY_MAX_CHARS


def _memory_no_duplicates(student: AnalysisResult) -> bool:
    """各分类条目归一化后不得重复，且条数不超分类上限"""
    for category, texts in memory_items(student).items():
        if len(texts) > MEMORY_CATEGORY_LIMITS[category]:
            return False
        normalized = [normalize_text(text) for text in texts]
        if len(set(normalized)) != len(normalized):
            return False
    return True
