# @author: ztwz
"""prompt_parser 测试：三套模板变体、说话人标注、旧摘要段提取。"""
from app.evaluation.prompt_parser import parse_prompt

# v3 英文模板（线上主流，74488 条）
V3_EN_PROMPT = """## ROLE
You are a conversation analysis and memory management expert.

## OUTPUT LANGUAGE
Use the same language as the chat records.

## USER BASIC INFORMATION
- Name: Jay, Gender: Man, Age: 38
- Personal introduction: nothing special

## CURRENT DATE
2026-09-15

## OLD ROLLING SUMMARY

## OLD MEMORY SNAPSHOT

## INCREMENTAL CHAT RECORDS
会话：Jay - Raven
[2026-09-15 07:39:56] Raven: Ignoring me, Jay? My pole is cold tonight
[2026-09-15 07:40:01] Raven: Come over, I'll show you around
[2026-09-15 07:41:30] Jay: hey sorry was working

## ANALYSIS REQUIREMENTS (analysis_summary / user_intent / paid_reaction / ai_performance)

## ROLLING SUMMARY REQUIREMENTS (rolling_summary)

## MEMORY SNAPSHOT REQUIREMENTS (memory_snapshot)
"""

# v3 中文模板（过渡期 14 条）
V3_ZH_PROMPT = """## ROLE
你是一名对话分析与记忆管理专家，为虚拟人聊天应用工作。

## 输入说明
- 系统只保留最近的原始聊天

## 用户基本信息
- Name: Mrbig, Gender: Man, Age: 44

## 当前日期
2026-07-28

## 旧滚动摘要
用户此前询问了 Nicole 的居住地。

## 旧记忆快照
{"facts": ["User works in finance"]}

## 增量聊天记录
会话：Mrbig - Nicole Adams
[2026-07-28 23:24:15] Mrbig: where from
[2026-07-28 23:24:26] Nicole Adams: I'm based in the city!

## 分析要求（analysis_summary / user_intent / paid_reaction / ai_performance）
"""

# v1 混合期模板（无旧摘要/记忆段）
V1_MIXED_PROMPT = """## ROLE
你是一名经验丰富的AI对话分析专家，专门为虚拟人聊天应用提供深入的对话评估。

## TASK
分析下面的对话。

## 用户基本信息
- Name: A.Juice, Gender: Man, Age: 39

## 用户聊天记录
会话：A.Juice - Naomi Hayes
[2026-07-29 03:24:31] Naomi Hayes: Still ghosting me?
[2026-07-29 06:10:58] A.Juice: whats up? not ghosting just busy.
"""


class TestTemplateVersion:
    def test_v3_en(self):
        assert parse_prompt(V3_EN_PROMPT).template_version == "v3-en"

    def test_v3_zh(self):
        assert parse_prompt(V3_ZH_PROMPT).template_version == "v3-zh"

    def test_v1_mixed(self):
        assert parse_prompt(V1_MIXED_PROMPT).template_version == "v1-mixed"

    def test_unknown(self):
        assert parse_prompt("random text without role").template_version == "unknown"


class TestChatRecords:
    def test_v3_en_lines_and_speakers(self):
        ctx = parse_prompt(V3_EN_PROMPT)
        assert len(ctx.chat_records) == 3
        assert ctx.user_name == "Jay"
        assert ctx.persona_name == "Raven"
        # 说话人归因：Raven=人设(False)，Jay=用户(True)
        assert ctx.chat_records[0].is_user is False
        assert ctx.chat_records[1].is_user is False
        assert ctx.chat_records[2].is_user is True
        assert ctx.chat_records[2].text == "hey sorry was working"

    def test_v3_zh_template(self):
        ctx = parse_prompt(V3_ZH_PROMPT)
        assert len(ctx.chat_records) == 2
        assert ctx.chat_records[0].is_user is True  # Mrbig = 用户
        assert ctx.chat_records[1].is_user is False  # Nicole Adams = 人设
        assert ctx.user_name == "Mrbig"

    def test_v1_mixed_template(self):
        ctx = parse_prompt(V1_MIXED_PROMPT)
        assert len(ctx.chat_records) == 2
        assert ctx.chat_records[0].is_user is False
        assert ctx.chat_records[1].is_user is True

    def test_third_speaker_unknown(self):
        """聊天行说话人既非用户也非人设 → is_user=None（评分降级不重罚）"""
        # 追加行须插入 chat 段内（其他段的方括号行不属于聊天记录）
        prompt = V3_EN_PROMPT.replace(
            "hey sorry was working",
            "hey sorry was working\n[2026-09-15 08:00:00] Stranger: hello there",
        )
        ctx = parse_prompt(prompt)
        assert len(ctx.chat_records) == 4
        assert ctx.chat_records[-1].is_user is None

    def test_bracket_inside_message_not_misparsed(self):
        """消息内容里的方括号（如 [shared a photo]）不产生误匹配行"""
        prompt = V3_EN_PROMPT.replace(
            "hey sorry was working", "sent [shared a photo] to you"
        )
        ctx = parse_prompt(prompt)
        assert len(ctx.chat_records) == 3
        assert ctx.chat_records[2].text == "sent [shared a photo] to you"

    def test_empty_chat_section(self):
        """聊天段无消息行 → 空列表不抛异常"""
        prompt = V3_EN_PROMPT.split("[2026-09-15 07:39:56]")[0]
        ctx = parse_prompt(prompt)
        assert ctx.chat_records == []


class TestOldContext:
    def test_v3_zh_old_summary_and_memory(self):
        ctx = parse_prompt(V3_ZH_PROMPT)
        assert ctx.old_rolling_summary == "用户此前询问了 Nicole 的居住地。"
        assert ctx.old_memory_snapshot == '{"facts": ["User works in finance"]}'

    def test_v3_en_empty_old_sections(self):
        """英文模板旧段为空 → None"""
        ctx = parse_prompt(V3_EN_PROMPT)
        assert ctx.old_rolling_summary is None
        assert ctx.old_memory_snapshot is None
        assert ctx.current_date == "2026-09-15"

    def test_v1_mixed_no_old_sections(self):
        ctx = parse_prompt(V1_MIXED_PROMPT)
        assert ctx.old_rolling_summary is None
        assert ctx.old_memory_snapshot is None


class TestNames:
    def test_name_with_spaces_and_dots(self):
        prompt = V1_MIXED_PROMPT
        ctx = parse_prompt(prompt)
        assert ctx.user_name == "A.Juice"
        assert ctx.persona_name == "Naomi Hayes"

    def test_session_line_fallback_without_user_info(self):
        """用户信息段缺失时，从会话行按 ' - ' 分割兜底"""
        prompt = V3_EN_PROMPT.replace("- Name: Jay, Gender: Man, Age: 38\n", "")
        ctx = parse_prompt(prompt)
        assert ctx.user_name == "Jay"
        assert ctx.persona_name == "Raven"
        # 说话人仍能正确归因
        assert ctx.chat_records[0].is_user is False

    def test_persona_name_with_hyphen(self):
        """人设昵称本身含空格时以已知用户名锚定提取"""
        prompt = V3_EN_PROMPT.replace("Raven", "Mary Jane")
        ctx = parse_prompt(prompt)
        assert ctx.user_name == "Jay"
        assert ctx.persona_name == "Mary Jane"
