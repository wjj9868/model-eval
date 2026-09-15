# @author: ztwz
"""评分聚合：解析 teacher/student 输出 → L1 + L2 评分 → 加权总分（SampleScore）。"""
from __future__ import annotations

from dataclasses import asdict

from app.evaluation.output_parser import parse_output
from app.evaluation.prompt_parser import parse_prompt
from app.evaluation.scoring.embedding_scorer import DEFAULT_SEMANTIC_THRESHOLD, score_embedding
from app.evaluation.scoring.rule_scorer import score_rules
from app.evaluation.scoring.schemas import SampleScore, weighted_total


def evaluate_sample(
    sample_id: str,
    prompt: str,
    teacher_text: str,
    student_text: str,
    embedder,
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
) -> SampleScore:
    """单样本完整评分：L1 规则 + L2 语义，按 WEIGHTS 加权出总分。"""
    teacher = parse_output(teacher_text)
    student = parse_output(student_text)
    ctx = parse_prompt(prompt)

    rule = score_rules(student)
    semantic = score_embedding(teacher, student, ctx, embedder, semantic_threshold)

    scores = {
        "memory_precision": semantic.memory_precision,
        "memory_recall": semantic.memory_recall,
        "speaker_attribution": semantic.speaker_attribution,
        # 幻觉是惩罚项，取补后与其它分项同向（越大越好）；无记忆条目时 penalty 为 None → 该维度不适用
        "hallucination": (
            None if semantic.hallucination_penalty is None else 1.0 - semantic.hallucination_penalty
        ),
        "summary_score": semantic.summary_score,
        "intent_score": semantic.intent_score,
        "json_valid": rule.json_valid,
        "other": semantic.other,
    }

    return SampleScore(
        sample_id=sample_id,
        total_score=weighted_total(scores),
        json_valid=rule.json_valid,
        intent_score=semantic.intent_score,
        summary_score=semantic.summary_score,
        memory_precision=semantic.memory_precision,
        memory_recall=semantic.memory_recall,
        speaker_attribution=semantic.speaker_attribution,
        hallucination_penalty=semantic.hallucination_penalty,
        other=semantic.other,
        details={
            "rule": rule.to_dict(),
            "memory_details": [asdict(detail) for detail in semantic.memory_details],
            "grounding_mean": semantic.grounding_mean,
        },
    )
