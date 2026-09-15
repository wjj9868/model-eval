# @author: ztwz
"""L1 rule_scorer 测试：json_valid 分级与各结构检查。"""
import json

from app.evaluation.output_parser import parse_output
from app.evaluation.scoring.rule_scorer import score_rules


def _student(fields: dict) -> str:
    """构造 student 输出：默认完整合法，按 fields 覆写指定字段"""
    base = {
        "analysis_summary": {"overall_impression": "ok", "user_engagement_level": "MEDIUM"},
        "user_intent": {"primary_intent": "UNKNOWN", "secondary_intents": [], "reasoning": "r"},
        "paid_reaction": {"triggered": False, "reaction_type": "NOT_TRIGGERED", "analysis": "a"},
        "ai_performance": {"naturalness_score": 5, "identity_awareness": "NOT_DETECTED", "naturalness_reasoning": "n"},
        "rolling_summary": "s" * 300,
        "memory_snapshot": {"facts": ["f1"], "preferences": [], "avoid_topics": [], "active_events": [], "open_loops": []},
    }
    base.update(fields)
    return json.dumps(base, ensure_ascii=False)


class TestJsonValidLevels:
    def test_complete_valid_output(self):
        result = score_rules(parse_output(_student({})))
        assert result.json_valid == 1.0
        assert all(result.checks.values())

    def test_unparseable_is_zero(self):
        result = score_rules(parse_output("complete garbage"))
        assert result.json_valid == 0.0
        assert result.checks == {"parseable": False}

    def test_missing_field_is_quarter(self):
        """可解析但六字段不全 → 只给 0.25（不再给半程结构分）"""
        output = json.loads(_student({}))
        del output["rolling_summary"]
        result = score_rules(parse_output(json.dumps(output)))
        assert result.json_valid == 0.25
        assert result.checks["fields_complete"] is False

    def test_wrong_field_type_counts_as_incomplete(self):
        """字段存在但类型不符（如 rolling_summary=null）→ 不算齐全，只给 0.25"""
        result = score_rules(parse_output(_student({"rolling_summary": None})))
        assert result.checks["fields_complete"] is False
        assert result.json_valid == 0.25

    def test_parseable_with_defects_is_half(self):
        """可解析但 memory 分类缺失 → 0.5"""
        result = score_rules(parse_output(_student({"memory_snapshot": {"facts": ["a"]}})))
        assert result.json_valid == 0.5
        assert result.checks["memory_categories_valid"] is False


class TestMemoryChecks:
    def test_unknown_memory_category(self):
        result = score_rules(parse_output(_student({
            "memory_snapshot": {"facts": [], "preferences": [], "avoid_topics": [], "active_events": [], "open_loops": [], "hobbies": ["x"]}
        })))
        assert result.checks["memory_categories_valid"] is False

    def test_memory_snapshot_not_dict(self):
        result = score_rules(parse_output(_student({"memory_snapshot": ["list"]})))
        assert result.checks["memory_categories_valid"] is False

    def test_category_over_limit(self):
        """条数超分类上限（facts > 10）→ 缺陷"""
        result = score_rules(parse_output(_student({
            "memory_snapshot": {"facts": [f"f{i}" for i in range(11)], "preferences": [], "avoid_topics": [], "active_events": [], "open_loops": []}
        })))
        assert result.checks["memory_no_duplicates"] is False
        assert result.json_valid == 0.5

    def test_duplicate_entries(self):
        """同分类归一化后重复条目 → 缺陷"""
        result = score_rules(parse_output(_student({
            "memory_snapshot": {"facts": ["User is 27.", "user is 27"], "preferences": [], "avoid_topics": [], "active_events": [], "open_loops": []}
        })))
        assert result.checks["memory_no_duplicates"] is False


class TestRollingSummaryChecks:
    def test_empty_summary(self):
        result = score_rules(parse_output(_student({"rolling_summary": "  "})))
        assert result.checks["rolling_summary_length_ok"] is False

    def test_summary_not_string(self):
        result = score_rules(parse_output(_student({"rolling_summary": None})))
        assert result.checks["rolling_summary_length_ok"] is False

    def test_over_limit_summary(self):
        """超过硬上限 2000 字符 → 缺陷（与后端 SUMMARY_MAX_LENGTH 对齐）"""
        result = score_rules(parse_output(_student({"rolling_summary": "s" * 2001})))
        assert result.checks["rolling_summary_length_ok"] is False

    def test_boundary_exactly_2000_chars(self):
        """边界：恰好 2000 字符 → 合法"""
        result = score_rules(parse_output(_student({"rolling_summary": "s" * 2000})))
        assert result.checks["rolling_summary_length_ok"] is True
