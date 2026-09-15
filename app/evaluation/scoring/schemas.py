# @author: ztwz
"""评分数据结构与权重定义。"""
from __future__ import annotations

from dataclasses import dataclass, field

# 加权总分权重（用户定义，和恒为 1）
WEIGHTS = {
    "memory_precision": 0.25,
    "speaker_attribution": 0.20,
    "memory_recall": 0.15,
    "summary_score": 0.15,
    "intent_score": 0.10,
    "json_valid": 0.10,
    "other": 0.05,
}


@dataclass
class MemoryItemDetail:
    """单条 student memory 的评分明细（可回溯定位问题）"""

    category: str
    text: str
    match_type: str  # exact / semantic / none
    matched_teacher_text: str | None
    grounding: float  # 证据分级 1.0 / 0.5 / 0（含旧记忆/旧摘要/基本信息等先验证据源）
    evidence_line_index: int | None
    # 任意行归因规则：True=用户侧有证据（含先验段）；False=仅人设行强支持（归因错误）；None=无法判定
    evidence_is_user: bool | None


@dataclass
class SampleScore:
    """单样本完整评分（字段与用户定义的输出结构对齐）"""

    sample_id: str
    total_score: float
    json_valid: float
    intent_score: float
    summary_score: float
    memory_precision: float
    memory_recall: float
    speaker_attribution: float
    hallucination_penalty: float
    other: float
    # 明细：L1 检查项布尔、L2 逐条 memory 明细与 grounding 均值等
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化为 JSONL 行（details 原样保留）"""
        return {
            "sample_id": self.sample_id,
            "total_score": round(self.total_score, 4),
            "json_valid": round(self.json_valid, 4),
            "intent_score": round(self.intent_score, 4),
            "summary_score": round(self.summary_score, 4),
            "memory_precision": round(self.memory_precision, 4),
            "memory_recall": round(self.memory_recall, 4),
            "speaker_attribution": round(self.speaker_attribution, 4),
            "hallucination_penalty": round(self.hallucination_penalty, 4),
            "other": round(self.other, 4),
            "details": self.details,
        }


def weighted_total(scores: dict[str, float]) -> float:
    """按 WEIGHTS 加权求总分；各分项已由调用方归一到 0~1"""
    return round(sum(WEIGHTS[key] * value for key, value in scores.items()), 4)
