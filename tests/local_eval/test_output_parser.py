# @author: ztwz
"""output_parser 测试：正常/围栏/杂文本/截断/乱码全路径。"""
import json

from app.evaluation.output_parser import (
    memory_items,
    normalize_text,
    parse_output,
)


def _valid_output() -> str:
    return json.dumps(
        {
            "analysis_summary": {"overall_impression": "ok", "user_engagement_level": "MEDIUM"},
            "user_intent": {"primary_intent": "CASUAL_ENTERTAINMENT", "secondary_intents": [], "reasoning": "r"},
            "paid_reaction": {"triggered": False, "reaction_type": "NOT_TRIGGERED", "analysis": "a"},
            "ai_performance": {"naturalness_score": 8, "identity_awareness": "NOT_DETECTED", "naturalness_reasoning": "n"},
            "rolling_summary": "summary text",
            "memory_snapshot": {
                "facts": ["User is 27 years old."],
                "preferences": ["User enjoys playful banter."],
                "avoid_topics": [],
                "active_events": [{"content": "Interview on Friday", "expires_at": "2026-09-20"}],
                "open_loops": ["Ask about the trip"],
            },
        },
        ensure_ascii=False,
    )


class TestParseOutput:
    def test_plain_json(self):
        result = parse_output(_valid_output())
        assert result.valid is True
        assert result.parse_error is None
        assert result.raw["rolling_summary"] == "summary text"

    def test_fenced_json(self):
        result = parse_output(f"```json\n{_valid_output()}\n```")
        assert result.valid is True
        assert result.raw["memory_snapshot"]["facts"] == ["User is 27 years old."]

    def test_fenced_without_language_tag(self):
        result = parse_output(f"```\n{_valid_output()}\n```")
        assert result.valid is True

    def test_json_with_leading_trailing_noise(self):
        """Base 模型常见：前后杂文本 + JSON 主体"""
        result = parse_output(f"Here is my analysis:\n{_valid_output()}\nHope this helps!")
        assert result.valid is True
        assert result.raw["ai_performance"]["naturalness_score"] == 8

    def test_json_with_nested_braces_in_strings(self):
        """JSON 字符串值内含花括号，括号配对需字符串感知"""
        payload = _valid_output().replace("summary text", "user said {weird} stuff }")
        result = parse_output("noise noise\n" + payload + "\ntail")
        assert result.valid is True
        assert "{weird}" in result.raw["rolling_summary"]

    def test_truncated_json_extracts_partial(self):
        """截断的不完整 JSON：在平衡点提取出合法子对象（容错），L1 会因字段缺失降级"""
        truncated = _valid_output()[: len(_valid_output()) // 2]
        result = parse_output(truncated)
        assert result.valid is False or "rolling_summary" not in result.raw

    def test_garbage_text(self):
        """纯乱码（Base 模型跑飞输出）"""
        result = parse_output("asdkjhaskjdhkajshd 12378123 @@@@ ???")
        assert result.valid is False
        assert result.parse_error == "no valid JSON object found"

    def test_empty_and_none(self):
        assert parse_output("").valid is False
        assert parse_output("   \n  ").valid is False
        assert parse_output(None).valid is False

    def test_non_object_json(self):
        """合法 JSON 但不是对象（数组/数字）→ 无有效结构"""
        assert parse_output("[1, 2, 3]").valid is False
        assert parse_output("42").valid is False

    def test_picks_longest_valid_object(self):
        """多个平衡块时取最长可解析对象"""
        short = '{"a": 1}'
        result = parse_output(f"{short} then {_valid_output()} end")
        assert result.valid is True
        assert "rolling_summary" in result.raw

    def test_escaped_quotes_inside_strings(self):
        """JSON 字符串含转义引号，不破坏配对扫描"""
        payload = json.dumps({"rolling_summary": 'he said "hi {ok}" and left', "memory_snapshot": {}}, ensure_ascii=False)
        result = parse_output(f"prefix\n{payload}\nsuffix")
        assert result.valid is True


class TestMemoryItems:
    def test_extracts_all_categories(self):
        result = parse_output(_valid_output())
        items = memory_items(result)
        assert items["facts"] == ["User is 27 years old."]
        assert items["preferences"] == ["User enjoys playful banter."]
        assert items["active_events"] == ["Interview on Friday"]
        assert items["open_loops"] == ["Ask about the trip"]
        assert items["avoid_topics"] == []

    def test_invalid_result_returns_empty(self):
        items = memory_items(parse_output("garbage"))
        assert all(v == [] for v in items.values())

    def test_snapshot_not_dict_returns_empty(self):
        result = parse_output(json.dumps({"memory_snapshot": ["not", "a", "dict"]}))
        items = memory_items(result)
        assert all(v == [] for v in items.values())

    def test_unknown_category_ignored(self):
        result = parse_output(json.dumps({"memory_snapshot": {"facts": ["a"], "hobbies": ["b"]}}))
        items = memory_items(result)
        assert items["facts"] == ["a"]
        assert "hobbies" not in items

    def test_active_events_accepts_str_and_content(self):
        """对象条目取 content，裸字符串条目直接收（统一容错），非法 content 丢弃"""
        result = parse_output(
            json.dumps({"memory_snapshot": {"active_events": [{"content": "ok", "expires_at": "x"}, {"content": 123}, "plain"]}})
        )
        assert memory_items(result)["active_events"] == ["ok", "plain"]


class TestNormalizeText:
    def test_case_space_punctuation(self):
        assert normalize_text("User is 27 years old.") == normalize_text(" user IS 27 years old ")
        assert normalize_text("A.B, C!") == normalize_text("abc")
        assert normalize_text("  ") == ""

    def test_chinese_and_unicode(self):
        assert normalize_text("用户喜欢爬山。") == normalize_text("用户喜欢爬山")
