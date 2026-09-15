# @author: ztwz
"""评分数据结构与权重定义。"""
from __future__ import annotations

from dataclasses import dataclass, field

# 加权总分权重（和恒为 1）
#
# 相对上一版的调整（让"幻觉"与"漏记"在总分层面对称可辨）：
# - memory_precision 0.25 → 0.18：空预测不再白拿满分后，precision 的绝对影响已下降；
# - speaker_attribution 0.20 → 0.14：非核心维度，且归因只在可判定条目上统计（覆盖面变窄）；
# - memory_recall 0.15 → 0.18：漏记是主要失效模式，需与 precision 对称；
# - hallucination 新增 0.08：计入 1 - hallucination_penalty，幻觉不再只靠 precision 间接体现；
# - other 0.05 → 0.07：吸收余量。
WEIGHTS = {
    "memory_precision": 0.18,
    "memory_recall": 0.18,
    "speaker_attribution": 0.14,
    "hallucination": 0.08,
    "summary_score": 0.15,
    "intent_score": 0.10,
    "json_valid": 0.10,
    "other": 0.07,
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
    """单样本完整评分（字段与用户定义的输出结构对齐）。

    None 表示该维度不适用（无判定依据），序列化为 JSON null，统计时按"不适用"剔除。
    """

    sample_id: str
    total_score: float
    json_valid: float
    intent_score: float
    summary_score: float
    memory_precision: float | None
    memory_recall: float | None
    speaker_attribution: float | None
    hallucination_penalty: float | None
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
            "memory_precision": _round_opt(self.memory_precision),
            "memory_recall": _round_opt(self.memory_recall),
            "speaker_attribution": _round_opt(self.speaker_attribution),
            "hallucination_penalty": _round_opt(self.hallucination_penalty),
            "other": round(self.other, 4),
            "details": self.details,
        }


def _round_opt(value: float | None) -> float | None:
    """可选分项保留 None（JSON null），仅对数值四舍五入"""
    return None if value is None else round(value, 4)


def weighted_total(scores: dict[str, float | None]) -> float:
    """按 WEIGHTS 加权求总分；各分项已由调用方归一到 0~1。

    - 值为 None 的分项视为"不适用"：剔除后按剩余权重重新归一化，
      既不奖励（不按 0 计入分母而虚高）也不惩罚（不按 0 拉低总分）；
    - 传入未知分项直接抛错：权重表与分项必须严格对应，避免静默漏算；
    - 全部不适用 → 0.0（没有任何可评分维度）。
    """
    unknown = set(scores) - set(WEIGHTS)
    if unknown:
        raise KeyError(f"WEIGHTS 缺少分项权重：{sorted(unknown)}")
    applicable = {key: value for key, value in scores.items() if value is not None}
    if not applicable:
        return 0.0
    total_weight = sum(WEIGHTS[key] for key in applicable)
    return round(sum(WEIGHTS[key] * value for key, value in applicable.items()) / total_weight, 4)
