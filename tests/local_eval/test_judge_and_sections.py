# @author: ztwz
"""L3 judge 预留接口测试 + _clean_section_body 全分支覆盖。"""
import pytest

from app.evaluation.output_parser import parse_output
from app.evaluation.prompt_parser import parse_prompt
from app.evaluation.scoring.judge import score_judge


class TestJudgeReserved:
    def test_not_implemented(self, mock_embedder):
        with pytest.raises(NotImplementedError):
            score_judge(parse_output("{}"), parse_output("{}"), parse_prompt("x"), mock_embedder)


class TestSectionBodyCleaning:
    def test_section_present_with_content(self):
        ctx = parse_prompt("## 旧滚动摘要\n\n有内容。\n\n## 增量聊天记录\n会话：A - B\n")
        assert ctx.old_rolling_summary == "有内容。"

    def test_section_present_blank(self):
        ctx = parse_prompt("## 旧滚动摘要\n   \n\n## 增量聊天记录\n会话：A - B\n")
        assert ctx.old_rolling_summary is None
