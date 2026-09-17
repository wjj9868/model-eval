# @author: ztwz
"""Level 2 语义评分：embedding 相似度驱动的多维评分。

包含 memory 精确/语义匹配（precision/recall）、原文证据 grounding、说话人归因、
summary/intent/other 语义相似度。embedder 为注入接口（测试可 mock，生产接
EmbeddingClient）。文本统一批量编码并按原文缓存向量，避免重复计算。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from app.evaluation.output_parser import AnalysisResult, memory_items, normalize_text
from app.evaluation.prompt_parser import PromptContext
from app.evaluation.scoring.schemas import MemoryItemDetail

# 语义等价判定阈值（cosine），可由调用方校准覆盖
DEFAULT_SEMANTIC_THRESHOLD = 0.85
# 证据分级阈值：(embedding 相似度, 词覆盖率) 任一超过即认定
GROUNDING_STRONG = (0.80, 0.60)  # → 1.0 明确支持
GROUNDING_WEAK = (0.60, 0.30)  # → 0.5 可推导
# 证据级别常量（_evidence_level 的三档取值，归因判定用）
GROUNDING_WEAK_LEVEL = 0.5
GROUNDING_STRONG_LEVEL = 1.0


@dataclass
class EmbeddingResult:
    """L2 评分结果（各分项 0~1；None = 该维度不适用，加权时剔除并重新归一化）"""

    memory_precision: float | None
    memory_recall: float | None
    speaker_attribution: float | None
    grounding_mean: float | None
    hallucination_penalty: float | None
    summary_score: float
    intent_score: float
    other: float
    memory_details: list[MemoryItemDetail] = field(default_factory=list)


def score_embedding(
    teacher: AnalysisResult,
    student: AnalysisResult,
    ctx: PromptContext,
    embedder,
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
) -> EmbeddingResult:
    """L2 语义评分主入口。

    - teacher/student：parse_output 的解析结果
    - ctx：parse_prompt 的上下文（grounding 证据源）
    - embedder：encode(texts) -> np.ndarray（已 L2 归一化）
    """
    vector_cache = _VectorCache(embedder)
    teacher_mem = memory_items(teacher)
    student_mem = memory_items(student)

    # student 输出无效（Base 跑飞）→ L2 全 0，不给空集语义的免费分（P=R=1 等）
    if not student.valid:
        return EmbeddingResult(
            memory_precision=0.0,
            memory_recall=0.0,
            speaker_attribution=0.0,
            grounding_mean=0.0,
            hallucination_penalty=1.0,
            summary_score=0.0,
            intent_score=0.0,
            other=0.0,
            memory_details=[],
        )

    # 全量文本一次性批量预编码（memory 条目 + 各文本字段 + 对话行），避免逐条编码
    _preload_all(teacher, student, teacher_mem, student_mem, ctx, vector_cache)

    details: list[MemoryItemDetail] = []
    matched_student_count = 0
    matched_teacher_count = 0
    teacher_total = sum(len(v) for v in teacher_mem.values())
    student_total = sum(len(v) for v in student_mem.values())

    # 先验证据源（旧记忆/旧摘要/基本信息）一次性提取
    prior_texts = _prior_texts(ctx)

    # 逐分类匹配 + grounding + speaker，明细逐条落 details
    for category in teacher_mem:
        t_texts = teacher_mem[category]
        s_texts = student_mem[category]
        matches = _match_category(t_texts, s_texts, vector_cache, semantic_threshold)
        matched_student_count += sum(1 for _, _, mtype in matches if mtype != "none")
        matched_teacher_count += sum(1 for _, _, mtype in matches if mtype != "none")
        for si, ti, mtype in matches:
            s_text = s_texts[si]
            grounding, evidence_idx, evidence_is_user = _evidence(s_text, ctx, prior_texts, vector_cache)
            details.append(
                MemoryItemDetail(
                    category=category,
                    text=s_text,
                    match_type=mtype,
                    matched_teacher_text=t_texts[ti] if mtype != "none" and ti is not None else None,
                    grounding=grounding,
                    evidence_line_index=evidence_idx,
                    evidence_is_user=evidence_is_user,
                )
            )

    precision = _precision(matched_student_count, student_total, teacher_total)
    recall = _recall(matched_teacher_count, teacher_total, student_total)

    # speaker 归因（任意行规则）：只统计"有证据且归因可判定"的条目——
    # 要求 grounding 达弱证据级且 evidence_is_user 非 None；无证据条目已由幻觉计入，不重复进分母。
    # 一条都不可判定 → None（不适用，加权时剔除）：既不免费满分，也不倒扣。
    judged = [
        d for d in details
        if d.grounding >= GROUNDING_WEAK_LEVEL and d.evidence_is_user is not None
    ]
    speaker = sum(float(d.evidence_is_user) for d in judged) / len(judged) if judged else None

    # grounding / 幻觉只有"真的写了记忆条目"时才可评：
    # 无条目 → None（不适用）。旧实现在这里给 grounding=1.0、幻觉=0，等于"什么都不写"白拿两项满分。
    if details:
        grounding_mean = round(sum(d.grounding for d in details) / len(details), 4)
        hallucination = round(sum(1 for d in details if d.grounding == 0) / len(details), 4)
    else:
        grounding_mean = None
        hallucination = None

    summary_score = _summary_score(teacher, student, vector_cache)
    intent_score = _intent_score(teacher, student, vector_cache)
    other = _other_score(teacher, student, vector_cache)

    return EmbeddingResult(
        memory_precision=precision,
        memory_recall=recall,
        speaker_attribution=speaker,
        grounding_mean=grounding_mean,
        hallucination_penalty=hallucination,
        summary_score=summary_score,
        intent_score=intent_score,
        other=other,
        memory_details=details,
    )


class _VectorCache:
    """文本向量缓存：同文本只编码一次；空文本不编码，cosine 直接记 0"""

    def __init__(self, embedder):
        self._embedder = embedder
        self._vectors: dict[str, np.ndarray] = {}

    def preload(self, texts: list[str]) -> None:
        """批量预编码（去重后一次调用 embedder，真实场景性能关键）"""
        pending = [t for t in dict.fromkeys(texts) if t and t not in self._vectors]
        if not pending:
            return
        vectors = self._embedder.encode(pending)
        for text, vector in zip(pending, vectors):
            self._vectors[text] = np.asarray(vector, dtype=np.float32)

    def cosine(self, a: str | None, b: str | None) -> float:
        """两文本 cosine 相似度（向量已 L2 归一化时等于点积）；空文本记 0"""
        if not a or not b:
            return 0.0
        if a not in self._vectors or b not in self._vectors:
            self.preload([a, b])
        va, vb = self._vectors[a], self._vectors[b]
        norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if norm == 0:
            return 0.0
        return float(np.dot(va, vb) / norm)


def _preload_all(
    teacher: AnalysisResult,
    student: AnalysisResult,
    teacher_mem: dict[str, list[str]],
    student_mem: dict[str, list[str]],
    ctx: PromptContext,
    cache: _VectorCache,
) -> None:
    """收集全部待编码文本一次性批量预编码（memory 条目 + 文本字段 + 对话行）"""
    texts: list[str] = []
    for items in (teacher_mem, student_mem):
        for texts_of_category in items.values():
            texts.extend(texts_of_category)
    for result in (teacher, student):
        if result.valid:
            for field in ("rolling_summary",):
                value = result.raw.get(field)
                if isinstance(value, str):
                    texts.append(value)
            for section in ("user_intent", "paid_reaction"):
                value = result.raw.get(section)
                if isinstance(value, dict):
                    for sub_key in ("reasoning", "analysis"):
                        sub = value.get(sub_key)
                        if isinstance(sub, str):
                            texts.append(sub)
    texts.extend(line.text for line in ctx.chat_records if line.text)
    texts.extend(_prior_texts(ctx))
    cache.preload(texts)


def _match_category(
    teacher_texts: list[str],
    student_texts: list[str],
    cache: _VectorCache,
    threshold: float,
) -> list[tuple[int, int | None, str]]:
    """单分类条目匹配：exact（归一化相等）优先，未命中的做语义一对一贪心配对。

    返回 [(student_idx, teacher_idx | None, match_type)]，match_type ∈ exact/semantic/none。
    """
    results: dict[int, tuple[int | None, str]] = {}
    used_teacher: set[int] = set()

    # 第一轮：exact
    for si, s_text in enumerate(student_texts):
        s_norm = normalize_text(s_text)
        for ti, t_text in enumerate(teacher_texts):
            if ti in used_teacher:
                continue
            if normalize_text(t_text) == s_norm:
                results[si] = (ti, "exact")
                used_teacher.add(ti)
                break

    # 第二轮：语义（未 exact 的 student 与剩余 teacher 两两，按相似度降序贪心一对一）
    pairs: list[tuple[float, int, int]] = []
    pending_students = [si for si in range(len(student_texts)) if si not in results]
    free_teacher = [ti for ti in range(len(teacher_texts)) if ti not in used_teacher]
    for si in pending_students:
        for ti in free_teacher:
            sim = cache.cosine(student_texts[si], teacher_texts[ti])
            if sim > threshold:
                pairs.append((sim, si, ti))
    for sim, si, ti in sorted(pairs, reverse=True):
        if si in results or ti in used_teacher:
            continue
        results[si] = (ti, "semantic")
        used_teacher.add(ti)

    return [(si, *results.get(si, (None, "none"))) for si in range(len(student_texts))]


def _prior_texts(ctx: PromptContext) -> list[str]:
    """先验证据源：旧记忆快照行 + 旧滚动摘要 + 用户基本信息行。

    这些段落是 prompt 给定的用户既有上下文，条目命中即视为用户侧支持
    （记忆延续/背景事实），不计幻觉、不判归因错误。
    """
    sources: list[str] = []
    if ctx.old_memory_snapshot:
        sources.extend(ln.strip("- ").strip() for ln in ctx.old_memory_snapshot.splitlines() if ln.strip())
    if ctx.old_rolling_summary:
        sources.append(ctx.old_rolling_summary)
    if ctx.user_info:
        sources.extend(ln.strip("- ").strip() for ln in ctx.user_info.splitlines() if ln.strip())
    return sources


def _evidence(
    text: str,
    ctx: PromptContext,
    prior_texts: list[str],
    cache: _VectorCache,
) -> tuple[float, int | None, bool | None]:
    """单条 memory 的证据分级、最佳对话行定位与说话人归因。

    归因采用任意行规则（修复原"最佳匹配行属于谁"的系统性偏置）：
    - 用户侧支持 = 任一用户行 OR 任一先验段命中（弱证据即可）
    - 归因错误 = 人设行达强证据 且 用户侧零证据（弱人设匹配不构成错误）
    - 其余（仅未知说话人行/仅弱人设匹配/无证据）无法判定，返回 None

    返回 (grounding, evidence_line_index, evidence_is_user)。
    无任何证据源时返回 (0.5, None, None) 中性。
    """
    if not ctx.chat_records and not prior_texts:
        return 0.5, None, None

    words = _word_set(text)
    user_lvl = persona_lvl = other_lvl = 0.0
    best_chat_lvl = 0.0
    best_sim = -1.0
    best_idx: int | None = None
    for idx, line in enumerate(ctx.chat_records):
        sim = cache.cosine(text, line.text)
        overlap = len(words & _word_set(line.text)) / len(words) if words else 0.0
        level = _evidence_level(sim, overlap)
        if line.is_user is True:
            user_lvl = max(user_lvl, level)
        elif line.is_user is False:
            persona_lvl = max(persona_lvl, level)
        else:
            other_lvl = max(other_lvl, level)
        if level > best_chat_lvl or (level == best_chat_lvl and sim > best_sim):
            best_chat_lvl, best_sim, best_idx = level, sim, idx
    for cand in prior_texts:
        sim = cache.cosine(text, cand)
        overlap = len(words & _word_set(cand)) / len(words) if words else 0.0
        user_lvl = max(user_lvl, _evidence_level(sim, overlap))

    grounding = max(user_lvl, persona_lvl, other_lvl)
    # 证据行仅在达到弱证据级以上才有意义（原实现 level=0 也返回最高相似行，属 bug）
    evidence_idx = best_idx if best_chat_lvl > 0 else None

    if user_lvl >= GROUNDING_WEAK_LEVEL:
        is_user: bool | None = True
    elif persona_lvl >= GROUNDING_STRONG_LEVEL:
        is_user = False
    else:
        is_user = None
    return grounding, evidence_idx, is_user


def _evidence_level(sim: float, overlap: float) -> float:
    """证据分级：明确支持 1.0 / 可推导 0.5 / 无证据 0"""
    if sim > GROUNDING_STRONG[0] or overlap > GROUNDING_STRONG[1]:
        return 1.0
    if sim > GROUNDING_WEAK[0] or overlap > GROUNDING_WEAK[1]:
        return 0.5
    return 0.0


def _word_set(text: str) -> set[str]:
    """词覆盖率特征集合：英文/数字连续段整体为词，中文连续段按字符二元组切分。

    Python 正则的 \\w 会把整段连续汉字当成一个"词"，中文句子间几乎永不整段相等，
    词覆盖率通道因此系统性失效（证据分级退化为纯 cosine 单通道）；
    改用 bigram 后中文 overlap 才真正参与 grounding/归因判定。
    """
    words = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(run) == 1:
            words.add(run)
        else:
            words.update(run[i : i + 2] for i in range(len(run) - 1))
    return words


def _summary_score(teacher: AnalysisResult, student: AnalysisResult, cache: _VectorCache) -> float:
    """rolling_summary 语义相似度；student 缺失或非字符串记 0"""
    s_summary = student.raw.get("rolling_summary") if student.valid else None
    t_summary = teacher.raw.get("rolling_summary") if teacher.valid else None
    if not isinstance(s_summary, str) or not s_summary.strip():
        return 0.0
    return max(0.0, cache.cosine(s_summary, t_summary if isinstance(t_summary, str) else None))


def _intent_score(teacher: AnalysisResult, student: AnalysisResult, cache: _VectorCache) -> float:
    """user_intent 评分：primary 枚举 0.8 + secondary 集合 Jaccard 0.1 + reasoning 语义 0.1。

    primary_intent 是意图识别的核心信号（业务上主判断依据），权重占绝对主导：
    只要 primary 一致即得 0.8 基础分，secondary / reasoning 仅作小幅加分（区分细粒度质量）。
    历史口径为 0.5/0.2/0.3（primary 一致但 reasoning 缺失时只得分 0.5，被认为过度惩罚）。

    secondary_intents 非 schema 必填字段，teacher 常为空（实测 81%）。Jaccard 规则与 memory P/R 对齐：
    双方都空 → 1.0（一致地"无次要意图"）；仅一方为空 → 0.0（不一致）。
    """
    t_intent = teacher.raw.get("user_intent") if teacher.valid else {}
    s_intent = student.raw.get("user_intent") if student.valid else {}
    t_intent = t_intent if isinstance(t_intent, dict) else {}
    s_intent = s_intent if isinstance(s_intent, dict) else {}

    enum_match = 1.0 if _norm_enum(t_intent.get("primary_intent")) == _norm_enum(s_intent.get("primary_intent")) and s_intent.get("primary_intent") else 0.0

    t_sec = _enum_set(t_intent.get("secondary_intents"))
    s_sec = _enum_set(s_intent.get("secondary_intents"))
    jaccard = len(t_sec & s_sec) / len(t_sec | s_sec) if t_sec | s_sec else 1.0

    t_reason = t_intent.get("reasoning")
    s_reason = s_intent.get("reasoning")
    reasoning_sim = cache.cosine(s_reason, t_reason) if isinstance(s_reason, str) and s_reason.strip() else 0.0

    return round(0.8 * enum_match + 0.1 * jaccard + 0.1 * max(0.0, reasoning_sim), 4)


def _other_score(teacher: AnalysisResult, student: AnalysisResult, cache: _VectorCache) -> float:
    """other = ai_performance 与 paid_reaction 组合，各占 50%。

    ai_performance：identity_awareness 枚举 0.5 + naturalness_score 接近度 0.5；
    paid_reaction：triggered 布尔 0.4 + reaction_type 枚举 0.3 + analysis 语义 0.3。
    """
    t_perf = _as_dict(teacher, "ai_performance")
    s_perf = _as_dict(student, "ai_performance")
    identity_match = 1.0 if _norm_enum(s_perf.get("identity_awareness")) and _norm_enum(t_perf.get("identity_awareness")) == _norm_enum(s_perf.get("identity_awareness")) else 0.0
    score_closeness = _score_closeness(t_perf.get("naturalness_score"), s_perf.get("naturalness_score"))
    ai_perf_sub = 0.5 * identity_match + 0.5 * score_closeness

    t_paid = _as_dict(teacher, "paid_reaction")
    s_paid = _as_dict(student, "paid_reaction")
    triggered_match = 1.0 if t_paid.get("triggered") == s_paid.get("triggered") and s_paid.get("triggered") is not None else 0.0
    reaction_match = 1.0 if _norm_enum(s_paid.get("reaction_type")) and _norm_enum(t_paid.get("reaction_type")) == _norm_enum(s_paid.get("reaction_type")) else 0.0
    t_analysis = t_paid.get("analysis")
    s_analysis = s_paid.get("analysis")
    analysis_sim = cache.cosine(s_analysis, t_analysis) if isinstance(s_analysis, str) and s_analysis.strip() else 0.0
    paid_sub = 0.4 * triggered_match + 0.3 * reaction_match + 0.3 * max(0.0, analysis_sim)

    return round(0.5 * ai_perf_sub + 0.5 * paid_sub, 4)


def _as_dict(result: AnalysisResult, key: str) -> dict:
    value = result.raw.get(key) if result.valid else None
    return value if isinstance(value, dict) else {}


def _norm_enum(value) -> str:
    """枚举值归一化（大写字符串），非法类型归空串"""
    return value.strip().upper() if isinstance(value, str) else ""


def _enum_set(value) -> set[str]:
    """枚举数组归一化为集合"""
    if not isinstance(value, list):
        return set()
    return {_norm_enum(v) for v in value if _norm_enum(v)}


def _score_closeness(teacher_score, student_score) -> float:
    """数字评分接近度：1 - |a-b|/9（1~10 分制），clamp 到 0~1；非法值 0"""
    if not isinstance(teacher_score, (int, float)) or not isinstance(student_score, (int, float)):
        return 0.0
    return max(0.0, 1.0 - abs(teacher_score - student_score) / 9.0)


def _precision(matched_student: int, student_total: int, teacher_total: int) -> float | None:
    """记忆精确率；teacher 无条目时无 ground truth → None（不适用，不计入加权）。

    - teacher 有条目：student 空 → 0.0（全漏，不因"没写错的"白拿满分）；否则 命中/条目数；
    - 双方都空 → 1.0（一致地"无可记"）；
    - teacher 空 + student 有条目 → None：没有对照说这些条目错，不奖励也不倒扣；
      若属胡编，由 grounding/幻觉维度（只依赖 prompt 证据）兜底惩罚。
    """
    if teacher_total:
        return matched_student / student_total if student_total else 0.0
    return 1.0 if student_total == 0 else None


def _recall(matched_teacher: int, teacher_total: int, student_total: int) -> float | None:
    """记忆召回率；teacher 无条目时无可召回项 → None（不适用，不计入加权）。

    - teacher 有条目：命中/条目数（student 空即 0.0，全漏）；
    - 双方都空 → 1.0；
    - teacher 空 + student 有条目 → None（无对照，谈不上漏记）。
    """
    if teacher_total:
        return matched_teacher / teacher_total
    return 1.0 if student_total == 0 else None
