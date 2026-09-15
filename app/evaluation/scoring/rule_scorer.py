# @author: ztwz
"""Level 1 规则评分：纯代码零成本，检查 JSON 合法性与结构约束。

json_valid 四档：
- 0.0  无法解析出 JSON 对象；
- 0.25 可解析但六字段不全（"勉强解析"不给半程结构分）；
- 0.5  六字段齐全但仍有其它缺陷（分类/长度/重复）；
- 1.0  全项通过。
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
# 注：schema 描述里的 "300–500 characters" 是给模型的软引导，实测线上 teacher 中位 423 字符、
# 从不触及本上限，因此这里保持后端一致的硬上限，不对 800~2000 区间额外扣分。
ROLLING_SUMMARY_MAX_CHARS = 2000

# 六字段期望类型（与线上 UserChatSchema 对齐）：键存在 + 类型正确才算"字段齐全"
FIELD_TYPES: dict[str, type] = {
    "analysis_summary": dict,
    "user_intent": dict,
    "paid_reaction": dict,
    "ai_performance": dict,
    "rolling_summary": str,
    "memory_snapshot": dict,
}
# 与 output_parser.REQUIRED_FIELDS 同源校验：两处字段表漂移会直接导致评分口径不一致
if set(FIELD_TYPES) != set(REQUIRED_FIELDS):
    raise RuntimeError("FIELD_TYPES 与 REQUIRED_FIELDS 不一致，请同步更新")


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
    checks["fields_complete"] = _fields_complete(student)
    checks["memory_categories_valid"] = _memory_categories_valid(student)
    checks["rolling_summary_length_ok"] = _rolling_summary_ok(student)
    checks["memory_no_duplicates"] = _memory_no_duplicates(student)

    return RuleResult(_json_valid_tier(checks), checks)


def _json_valid_tier(checks: dict[str, bool]) -> float:
    """四档打分：字段不全只给 0.25（避免"能解析"就白拿结构分）"""
    if not checks.get("parseable"):
        return 0.0
    if not checks["fields_complete"]:
        return 0.25
    return 1.0 if all(checks.values()) else 0.5


def _fields_complete(student: AnalysisResult) -> bool:
    """六字段齐全 = 键存在且类型符合 schema。

    只查"键存在"会让 `{"rolling_summary": null, ...}` 这种输出也算齐全（白拿结构分），
    故按 FIELD_TYPES 同时校验类型。
    """
    return all(
        isinstance(student.raw.get(name), expected) for name, expected in FIELD_TYPES.items()
    )


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
