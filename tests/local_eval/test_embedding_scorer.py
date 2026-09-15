# @author: ztwz
"""L2 embedding_scorer + aggregator 测试：mock embedder 纯逻辑全路径。"""
import pytest

from app.evaluation.output_parser import parse_output
from app.evaluation.prompt_parser import parse_prompt
from app.evaluation.scoring.aggregator import evaluate_sample
from app.evaluation.scoring.embedding_scorer import score_embedding
from app.evaluation.scoring.schemas import WEIGHTS
from tests.local_eval.fakes import MockEmbedder, make_prompt

# 测试用记忆条目
FACT_27 = "User is 27 years old."
FACT_27_VARIANT = "user is 27 years old"  # 归一化后与 FACT_27 相同
FACT_FISHING = "User likes fishing on weekends."
FACT_SEMANTIC_T = "User enjoys playful banter."
FACT_SEMANTIC_S = "User likes humorous teasing."
BANANAS = "User is a banana astronaut from Mars."


class TestMemoryMatching:
    def _score(self, teacher_text, student_text, prompt, alias_map=None):
        embedder = MockEmbedder(alias_map)
        result = score_embedding(parse_output(teacher_text), parse_output(student_text), parse_prompt(prompt), embedder)
        return result

    def test_exact_match_full_precision_recall(self, teacher_factory, prompt_factory):
        result = self._score(
            teacher_factory(facts=[FACT_27]),
            teacher_factory(facts=[FACT_27]),
            prompt_factory(),
        )
        assert result.memory_precision == 1.0
        assert result.memory_recall == 1.0
        assert result.memory_details[0].match_type == "exact"

    def test_normalized_exact_match(self, teacher_factory, prompt_factory):
        """大小写/标点/空格差异归一化后视为相同"""
        result = self._score(
            teacher_factory(facts=[FACT_27]),
            teacher_factory(facts=[FACT_27_VARIANT]),
            prompt_factory(),
        )
        assert result.memory_precision == 1.0
        assert result.memory_details[0].match_type == "exact"

    def test_semantic_match_via_alias(self, teacher_factory, prompt_factory):
        """语义等价（embedding 相似度 > 阈值）"""
        result = self._score(
            teacher_factory(facts=[FACT_SEMANTIC_T]),
            teacher_factory(facts=[FACT_SEMANTIC_S]),
            prompt_factory(),
            alias_map={(FACT_SEMANTIC_T, FACT_SEMANTIC_S): 0.92},
        )
        assert result.memory_precision == 1.0
        assert result.memory_details[0].match_type == "semantic"

    def test_semantic_below_threshold_is_miss(self, teacher_factory, prompt_factory):
        """相似度低于阈值 → 不匹配"""
        result = self._score(
            teacher_factory(facts=[FACT_SEMANTIC_T]),
            teacher_factory(facts=[FACT_SEMANTIC_S]),
            prompt_factory(),
            alias_map={(FACT_SEMANTIC_T, FACT_SEMANTIC_S): 0.70},
        )
        assert result.memory_precision == 0.0
        assert result.memory_details[0].match_type == "none"

    def test_partial_match(self, teacher_factory, prompt_factory):
        """student 多一条凭空记忆：P=0.5, R=1.0，多余条目 match_type=none"""
        result = self._score(
            teacher_factory(facts=[FACT_27]),
            teacher_factory(facts=[FACT_27, FACT_FISHING]),
            prompt_factory(),
        )
        assert result.memory_precision == 0.5
        assert result.memory_recall == 1.0
        types = {d.text: d.match_type for d in result.memory_details}
        assert types[FACT_FISHING] == "none"

    def test_one_to_one_greedy_matching(self, teacher_factory, prompt_factory):
        """两条 student 都语义接近同一条 teacher → 只允许一对一配对"""
        other = "User adores funny joking."
        result = self._score(
            teacher_factory(facts=[FACT_SEMANTIC_T]),
            teacher_factory(facts=[FACT_SEMANTIC_S, other]),
            prompt_factory(),
            alias_map={
                (FACT_SEMANTIC_T, FACT_SEMANTIC_S): 0.95,
                (FACT_SEMANTIC_T, other): 0.90,
            },
        )
        assert result.memory_precision == 0.5
        assert result.memory_recall == 1.0

    def test_both_empty_memory(self, teacher_factory, prompt_factory):
        """双方都无 memory → 空集语义 P=R=1"""
        result = self._score(teacher_factory(), teacher_factory(), prompt_factory())
        assert result.memory_precision == 1.0
        assert result.memory_recall == 1.0
        assert result.memory_details == []

    def test_student_empty_teacher_has(self, teacher_factory, prompt_factory):
        """student 漏记全部 → P=1（没写错的）R=0"""
        result = self._score(teacher_factory(facts=[FACT_27]), teacher_factory(), prompt_factory())
        assert result.memory_precision == 1.0
        assert result.memory_recall == 0.0

    def test_student_has_teacher_empty(self, teacher_factory, prompt_factory):
        """teacher 无 memory、student 凭空生成 → P=0 R=1（凭空记忆被罚）"""
        result = self._score(teacher_factory(), teacher_factory(facts=[FACT_FISHING]), prompt_factory())
        assert result.memory_precision == 0.0
        assert result.memory_recall == 1.0

    def test_cross_category_no_match(self, teacher_factory, prompt_factory):
        """同内容放错分类（facts→preferences）→ 分类内无法命中，P 与 R 双降"""
        result = self._score(
            teacher_factory(facts=[FACT_27]),
            teacher_factory(preferences=[FACT_27]),
            prompt_factory(),
        )
        assert result.memory_precision == 0.0
        assert result.memory_recall == 0.0


class TestGroundingAndSpeaker:
    """证据与归因：任意行规则 + 先验证据源（旧记忆/旧摘要/基本信息）。

    归因判定规则（审计修复后）：
    - 用户侧支持 = 任一用户行或先验段命中（弱证据即可）→ 记 1
    - 归因错误 = 人设行强证据 且 用户侧零证据 → 记 0
    - 其余无法判定 → 不进归因分母（speaker 记满分，不掺中性 0.5）
    """

    def _score(self, student_text, chat_lines, alias_map=None, prompt=None):
        embedder = MockEmbedder(alias_map)
        teacher = parse_output("{}")
        student = parse_output(student_text)
        prompt = prompt or _prompt_with_chat(chat_lines)
        return score_embedding(teacher, student, parse_prompt(prompt), embedder)

    def test_strong_evidence_from_user(self, teacher_factory):
        """用户原话 → 词全覆盖 → grounding 1.0，speaker 1.0"""
        result = self._score(
            teacher_factory(facts=["I'm an engineer from Texas"]),
            [("2026-09-15 02:41:00", "Jay", "I'm an engineer from Texas, I love hiking.")],
        )
        detail = result.memory_details[0]
        assert detail.grounding == 1.0
        assert detail.evidence_is_user is True
        assert result.speaker_attribution == 1.0
        assert result.hallucination_penalty == 0.0

    def test_persona_speech_attributed_to_user_penalized(self, teacher_factory):
        """人设发言被记成用户记忆且用户侧零证据 → speaker 0.0（重罚）"""
        result = self._score(
            teacher_factory(facts=["I enjoy yoga every morning"]),
            [
                ("2026-09-15 02:41:00", "Jay", "hello"),
                ("2026-09-15 02:42:16", "Raven", "I enjoy yoga every morning."),
            ],
        )
        detail = result.memory_details[0]
        assert detail.evidence_is_user is False
        assert result.speaker_attribution == 0.0

    def test_echo_persona_strong_user_weak_not_error(self, teacher_factory):
        """回声场景：人设行完整复述（强证据）+ 用户行简短确认（弱证据）→ 归因正确。

        原实现只看最佳匹配行（会落人设行）导致误判，任意行规则修复该偏置。
        """
        result = self._score(
            teacher_factory(facts=["User enjoys Stroking"]),
            [
                ("2026-09-15 02:41:00", "Jay", "I enjoy Stroking"),
                ("2026-09-15 02:42:16", "Raven", "User enjoys Stroking"),
            ],
        )
        detail = result.memory_details[0]
        assert detail.evidence_is_user is True
        assert detail.grounding == 1.0
        assert result.speaker_attribution == 1.0

    def test_weak_persona_only_not_attribution_error(self, teacher_factory):
        """仅人设行弱证据（词覆盖 0.3~0.6）且用户侧零证据 → 无法判定，不判归因错误"""
        result = self._score(
            teacher_factory(facts=["yoga morning session practice"]),
            [("2026-09-15 02:42:16", "Raven", "I enjoy yoga every morning.")],
        )
        detail = result.memory_details[0]
        assert detail.evidence_is_user is None
        assert result.speaker_attribution == 1.0  # 无可归因条目 → 满分

    def test_weak_evidence_half(self, teacher_factory):
        """词覆盖率 0.3~0.6 → grounding 0.5（可推导）"""
        result = self._score(
            teacher_factory(facts=["engineer from somewhere new"]),
            [("2026-09-15 02:41:00", "Jay", "I'm an engineer from Texas.")],
        )
        assert result.memory_details[0].grounding == 0.5

    def test_no_evidence_hallucination(self, teacher_factory):
        """完全无证据 → grounding 0，计入幻觉惩罚；归因无错可判不掺中性分"""
        result = self._score(
            teacher_factory(facts=[BANANAS]),
            [("2026-09-15 02:41:00", "Jay", "I'm an engineer from Texas.")],
        )
        assert result.memory_details[0].grounding == 0.0
        assert result.memory_details[0].evidence_is_user is None
        assert result.hallucination_penalty == 1.0
        assert result.speaker_attribution == 1.0

    def test_unknown_speaker_excluded_from_attribution(self, teacher_factory):
        """证据行说话人无法判定（第三方）→ 条目不进归因分母"""
        result = self._score(
            teacher_factory(facts=["I'm an engineer from Texas"]),
            [("2026-09-15 02:41:00", "Stranger", "I'm an engineer from Texas.")],
        )
        detail = result.memory_details[0]
        assert detail.evidence_is_user is None
        assert result.speaker_attribution == 1.0

    def test_no_evidence_source_neutral(self, teacher_factory):
        """对话与先验段全缺失 → grounding 0.5 中性、不计幻觉"""
        result = self._score(
            teacher_factory(facts=[FACT_27]),
            [],
            prompt="## INCREMENTAL CHAT RECORDS\n会话：Jay - Raven\n",
        )
        detail = result.memory_details[0]
        assert detail.grounding == 0.5
        assert result.hallucination_penalty == 0.0
        assert result.speaker_attribution == 1.0

    def test_no_chat_prior_unsupported_is_hallucination(self, teacher_factory):
        """无对话、先验段（基本信息）存在但不支持该事实 → 记幻觉、不判归因"""
        result = self._score(teacher_factory(facts=[FACT_27]), [])
        assert result.memory_details[0].grounding == 0.0
        assert result.hallucination_penalty == 1.0
        assert result.speaker_attribution == 1.0

    def test_fact_supported_by_user_info(self, teacher_factory):
        """基本信息段支持的事实（背景事实）→ 有 grounding、不计幻觉、归因正确"""
        result = self._score(teacher_factory(facts=["Jay is a man"]), [])
        detail = result.memory_details[0]
        assert detail.grounding == 0.5
        assert detail.evidence_is_user is True
        assert result.hallucination_penalty == 0.0
        assert result.speaker_attribution == 1.0

    def test_carried_from_old_memory_not_hallucination(self, teacher_factory):
        """沿用旧记忆快照的事实（记忆延续）→ 有 grounding、不计幻觉、归因正确"""
        prompt = make_prompt(chat_lines=[], old_memory="- User works in finance")
        result = self._score(teacher_factory(facts=["User works in finance"]), [], prompt=prompt)
        detail = result.memory_details[0]
        assert detail.grounding == 1.0
        assert detail.evidence_is_user is True
        assert result.hallucination_penalty == 0.0
        assert result.speaker_attribution == 1.0

    def test_carried_from_old_summary_not_hallucination(self, teacher_factory):
        """沿用旧滚动摘要的事实（记忆延续）→ 不计幻觉"""
        prompt = make_prompt(chat_lines=[], old_summary="User works in finance and enjoys it.")
        result = self._score(teacher_factory(facts=["User works in finance"]), [], prompt=prompt)
        assert result.memory_details[0].grounding == 1.0
        assert result.hallucination_penalty == 0.0


class TestSummaryIntentOther:
    def test_identical_summary_full_score(self, teacher_factory, prompt_factory, mock_embedder):
        result = score_embedding(
            parse_output(teacher_factory(rolling_summary="A B C")),
            parse_output(teacher_factory(rolling_summary="A B C")),
            parse_prompt(prompt_factory()),
            mock_embedder,
        )
        assert result.summary_score == pytest.approx(1.0, abs=1e-6)

    def test_different_summary_near_zero(self, teacher_factory, prompt_factory, mock_embedder):
        result = score_embedding(
            parse_output(teacher_factory(rolling_summary="User talked about work.")),
            parse_output(teacher_factory(rolling_summary=" Completely different topic entirely.")),
            parse_prompt(prompt_factory()),
            mock_embedder,
        )
        assert result.summary_score < 0.1

    def test_missing_summary_zero(self, teacher_factory, prompt_factory, mock_embedder):
        import json

        student = json.loads(teacher_factory(rolling_summary="x"))
        student["rolling_summary"] = None
        result = score_embedding(
            parse_output(teacher_factory(rolling_summary="x")),
            parse_output(json.dumps(student)),
            parse_prompt(prompt_factory()),
            mock_embedder,
        )
        assert result.summary_score == 0.0

    def test_intent_full_score(self, teacher_factory, prompt_factory, mock_embedder):
        """primary 枚举 + secondary 集合 + reasoning 全同 → 1.0"""
        result = score_embedding(
            parse_output(teacher_factory(primary_intent="SEXUAL_INTENT", secondary_intents=["COMPANIONSHIP"], reasoning="evidence quoted")),
            parse_output(teacher_factory(primary_intent="SEXUAL_INTENT", secondary_intents=["COMPANIONSHIP"], reasoning="evidence quoted")),
            parse_prompt(prompt_factory()),
            mock_embedder,
        )
        assert result.intent_score == pytest.approx(1.0, abs=1e-3)

    def test_intent_enum_case_insensitive(self, teacher_factory, prompt_factory, mock_embedder):
        """枚举大小写归一化后匹配"""
        result = score_embedding(
            parse_output(teacher_factory(primary_intent="SEXUAL_INTENT")),
            parse_output(teacher_factory(primary_intent="sexual_intent")),
            parse_prompt(prompt_factory()),
            mock_embedder,
        )
        # primary 全分 + secondary 全同（均空）+ reasoning 全同 → 1.0
        assert result.intent_score == pytest.approx(1.0, abs=1e-3)

    def test_intent_partial(self, teacher_factory, prompt_factory, mock_embedder):
        """primary 不同、reasoning 不同、secondary 部分重合 → 仅 secondary 得分"""
        result = score_embedding(
            parse_output(teacher_factory(primary_intent="SEXUAL_INTENT", secondary_intents=["A", "B"], reasoning="first reasoning text")),
            parse_output(teacher_factory(primary_intent="COMPANIONSHIP", secondary_intents=["A"], reasoning="second different reasoning")),
            parse_prompt(prompt_factory()),
            mock_embedder,
        )
        # 0.5*0(枚举) + 0.2*(1/2 jaccard) + 0.3*0(reasoning 正交)
        assert result.intent_score == pytest.approx(0.2 * 0.5, abs=1e-3)

    def test_other_full_score(self, teacher_factory, prompt_factory, mock_embedder):
        result = score_embedding(
            parse_output(teacher_factory(naturalness_score=8, identity_awareness="NOT_DETECTED", triggered=False, reaction_type="NOT_TRIGGERED", analysis="no paid content")),
            parse_output(teacher_factory(naturalness_score=8, identity_awareness="NOT_DETECTED", triggered=False, reaction_type="NOT_TRIGGERED", analysis="no paid content")),
            parse_prompt(prompt_factory()),
            mock_embedder,
        )
        assert result.other == pytest.approx(1.0, abs=1e-3)

    def test_naturalness_closeness(self, teacher_factory, prompt_factory, mock_embedder):
        """评分接近度：8 vs 5 → ai_perf 子分 = 0.5*1 + 0.5*(1-3/9)；paid 部分满分"""
        result = score_embedding(
            parse_output(teacher_factory(naturalness_score=8)),
            parse_output(teacher_factory(naturalness_score=5)),
            parse_prompt(prompt_factory()),
            mock_embedder,
        )
        ai_perf_sub = 0.5 * 1 + 0.5 * (1 - 3 / 9)
        expected = 0.5 * ai_perf_sub + 0.5 * 1.0
        assert result.other == pytest.approx(expected, abs=1e-3)

    def test_invalid_student_all_zero(self, teacher_factory, prompt_factory, mock_embedder):
        """Base 乱输出 → L2 各分项 0，不抛异常"""
        result = score_embedding(
            parse_output(teacher_factory()),
            parse_output("total garbage output"),
            parse_prompt(prompt_factory()),
            mock_embedder,
        )
        assert result.summary_score == 0.0
        assert result.intent_score == 0.0
        assert result.other == 0.0
        assert result.memory_precision == 0.0


class TestAggregator:
    def test_full_pipeline_perfect_student(self, teacher_factory, prompt_factory, mock_embedder):
        """student 与 teacher 完全一致 → 各分项满分，总分为 1（满分 100 制换算前）"""
        prompt = prompt_factory()
        teacher_text = teacher_factory()
        score = evaluate_sample("s1", prompt, teacher_text, teacher_text, mock_embedder)
        assert score.total_score == pytest.approx(1.0, abs=1e-3)
        assert score.json_valid == 1.0
        assert score.memory_precision == 1.0
        assert score.details["rule"]["json_valid"] == 1.0

    def test_garbage_student_zero(self, teacher_factory, prompt_factory, mock_embedder):
        score = evaluate_sample("s2", prompt_factory(), teacher_factory(), "garbage", mock_embedder)
        assert score.total_score == 0.0
        assert score.json_valid == 0.0

    def test_weighted_total_formula(self, teacher_factory, prompt_factory, mock_embedder):
        """总分严格按权重线性组合"""
        prompt = prompt_factory()
        teacher_text = teacher_factory(facts=[FACT_27])
        # student 只多一条凭空记忆：precision 0.5 其余满分
        student_text = teacher_factory(facts=[FACT_27, BANANAS])
        score = evaluate_sample("s3", prompt, teacher_text, student_text, mock_embedder)
        expected = (
            WEIGHTS["memory_precision"] * 0.5
            + WEIGHTS["speaker_attribution"] * score.speaker_attribution
            + WEIGHTS["memory_recall"] * 1.0
            + WEIGHTS["summary_score"] * score.summary_score
            + WEIGHTS["intent_score"] * score.intent_score
            + WEIGHTS["json_valid"] * 1.0
            + WEIGHTS["other"] * score.other
        )
        assert score.total_score == pytest.approx(expected, abs=1e-4)

    def test_weights_sum_to_one(self):
        assert sum(WEIGHTS.values()) == pytest.approx(1.0)

    def test_to_dict_fields(self, teacher_factory, prompt_factory, mock_embedder):
        score = evaluate_sample("s4", prompt_factory(), teacher_factory(), teacher_factory(), mock_embedder)
        payload = score.to_dict()
        for key in (
            "sample_id", "total_score", "json_valid", "intent_score", "summary_score",
            "memory_precision", "memory_recall", "speaker_attribution",
            "hallucination_penalty", "other", "details",
        ):
            assert key in payload


class TestVectorCache:
    """_VectorCache 白盒契约：空 preload、懒加载补编码、零向量防御"""

    def test_empty_preload_is_noop(self):
        from app.evaluation.scoring.embedding_scorer import _VectorCache

        cache = _VectorCache(MockEmbedder())
        cache.preload([])
        assert cache.cosine("a", "b") == pytest.approx(0.0, abs=1e-9)

    def test_lazy_load_unseen_text(self):
        """cosine 遇到未预载文本时自动补编码，结果与预载一致"""
        from app.evaluation.scoring.embedding_scorer import _VectorCache

        cache = _VectorCache(MockEmbedder())
        cache.preload(["x"])
        lazy = cache.cosine("x", "y")
        eager_cache = _VectorCache(MockEmbedder())
        eager_cache.preload(["x", "y"])
        assert lazy == pytest.approx(eager_cache.cosine("x", "y"), abs=1e-6)

    def test_zero_vector_returns_zero(self):
        """embedder 返回零向量（真实模型对异常文本可能如此）→ cosine 记 0 不崩"""
        import numpy as np

        from app.evaluation.scoring.embedding_scorer import _VectorCache

        class ZeroEmbedder:
            def encode(self, texts):
                return np.zeros((len(texts), 8), dtype=np.float32)

        cache = _VectorCache(ZeroEmbedder())
        assert cache.cosine("a", "b") == 0.0


def _prompt_with_chat(chat_lines: list[tuple[str, str, str]]) -> str:
    """构造含指定对话行的 prompt（Jay=用户，Raven=人设，其他昵称=未知）"""
    block = "\n".join(f"[{ts}] {speaker}: {text}" for ts, speaker, text in chat_lines)
    return (
        "## USER BASIC INFORMATION\n- Name: Jay, Gender: Man, Age: 30\n\n"
        "## INCREMENTAL CHAT RECORDS\n会话：Jay - Raven\n" + block + "\n"
    )
