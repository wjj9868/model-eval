# @author: ztwz
"""Level 3 LLM Judge：困难样本的高阶评判，接口预留（第一版不实现）。

L1 Rule + L2 Embedding 覆盖绝大多数样本后，仅对边界案例（语义冲突、事实矛盾、
证据模糊）抽样送高级模型仲裁。实现时保持签名不变。
"""
from __future__ import annotations

from app.evaluation.output_parser import AnalysisResult
from app.evaluation.prompt_parser import PromptContext


def score_judge(
    teacher: AnalysisResult,
    student: AnalysisResult,
    ctx: PromptContext,
    judge_client,
) -> dict:
    """L3 预留：高级模型仲裁困难样本，返回与 L2 同构的修正分项。"""
    raise NotImplementedError("L3 LLM Judge 尚未实现，第一版仅 L1 Rule + L2 Embedding")
