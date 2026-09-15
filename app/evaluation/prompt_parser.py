# @author: ztwz
"""prompt 解析：从渲染后的分析任务模板中提取结构化上下文。

供 grounding（原文证据）与 speaker attribution（说话人归因）评分使用。
覆盖三套线上模板变体：v1 混合期 / v3 中文 / v3 英文（段落标题中英文映射）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# 段落语义键 → 各模板变体的标题别名（标题不含 "## " 前缀）
SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "chat": ("用户聊天记录", "增量聊天记录", "INCREMENTAL CHAT RECORDS"),
    "user_info": ("用户基本信息", "USER BASIC INFORMATION"),
    "old_summary": ("旧滚动摘要", "OLD ROLLING SUMMARY"),
    "old_memory": ("旧记忆快照", "OLD MEMORY SNAPSHOT"),
    "current_date": ("当前日期", "CURRENT DATE"),
}

# 段落标题行（## 开头，标题可含中英文/括号/斜杠等）
_SECTION_LINE_RE = re.compile(r"^##\s+(.+?)\s*$", re.M)

# 聊天行：[yyyy-MM-dd HH:mm:ss] 说话人: 内容（锚定行首时间戳，聊天内容中的方括号不会误匹配）
_CHAT_LINE_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (.+?): (.*)$", re.M)

# 用户信息行：- Name: X, Gender: ..., Age: ...
_USER_NAME_RE = re.compile(r"^- Name: (.+?), Gender:", re.M)

# 会话行：会话：用户 - 人设（两套语言模板中该行均为中文硬编码）
_SESSION_RE = re.compile(r"^会话：(.+?) - (.+)$", re.M)

# 模板版本指纹（取 prompt 前 60 字符判断）
_TEMPLATE_FINGERPRINTS = (
    ("v3-en", "You are a conversation analysis and memory managemen"),
    ("v1-mixed", "你是一名经验丰富的AI对话分析专家"),
    ("v3-zh", "你是一名对话分析与记忆管理专家"),
)


@dataclass
class ChatLine:
    """一条聊天消息。is_user：True=用户，False=人设，None=无法判定。"""

    ts: str
    speaker: str
    text: str
    is_user: bool | None


@dataclass
class PromptContext:
    """prompt 解析出的结构化上下文"""

    template_version: str
    chat_records: list[ChatLine] = field(default_factory=list)
    user_name: str | None = None
    persona_name: str | None = None
    old_rolling_summary: str | None = None
    old_memory_snapshot: str | None = None
    user_info: str | None = None
    current_date: str | None = None


def parse_prompt(prompt: str) -> PromptContext:
    """解析 prompt 为 PromptContext。解析失败的字段保持默认值，不抛异常。"""
    version = _detect_version(prompt)
    sections = _split_sections(prompt)
    user_name, persona_name = _extract_names(prompt, sections.get("user_info", ""))
    chat_records = _extract_chat_lines(sections.get("chat", ""), user_name, persona_name)

    return PromptContext(
        template_version=version,
        chat_records=chat_records,
        user_name=user_name,
        persona_name=persona_name,
        old_rolling_summary=_clean_section_body(sections.get("old_summary")) if "old_summary" in sections else None,
        old_memory_snapshot=_clean_section_body(sections.get("old_memory")) if "old_memory" in sections else None,
        user_info=_clean_section_body(sections.get("user_info")) if "user_info" in sections else None,
        current_date=_clean_section_body(sections.get("current_date")) if "current_date" in sections else None,
    )


def _detect_version(prompt: str) -> str:
    """按 prompt 头部指纹判断模板版本，未知返回 'unknown'"""
    head = prompt[:60]
    for version, fingerprint in _TEMPLATE_FINGERPRINTS:
        if fingerprint in head:
            return version
    return "unknown"


def _split_sections(prompt: str) -> dict[str, str]:
    """按 ## 标题切分段落，语义键映射后返回 {语义键: 段落内容}"""
    matches = list(_SECTION_LINE_RE.finditer(prompt))
    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        title = match.group(1)
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        semantic_key = _map_title(title)
        if semantic_key is not None and semantic_key not in sections:
            sections[semantic_key] = prompt[body_start:body_end].strip("\n")
    return sections


def _map_title(title: str) -> str | None:
    """段落标题 → 语义键；标题可能带后缀（如 'INCREMENTAL CHAT RECORDS' 完整匹配，
    '用户聊天记录' 为 v1 短标题），用包含匹配兼容变体。"""
    for semantic_key, aliases in SECTION_ALIASES.items():
        if any(alias in title or title in alias for alias in aliases):
            return semantic_key
    return None


def _extract_names(prompt: str, user_info_body: str) -> tuple[str | None, str | None]:
    """提取用户名与人设昵称。

    用户名优先从用户信息段 '- Name: X, Gender:' 提取；
    人设名从会话行 '会话：用户 - 人设' 提取（以已知用户名锚定更稳）；
    用户名缺失时会话行按 ' - ' 分割兜底。
    """
    name_match = _USER_NAME_RE.search(user_info_body) or _USER_NAME_RE.search(prompt)
    session_match = _SESSION_RE.search(prompt)
    user_name = name_match.group(1).strip() if name_match else None

    persona_name = None
    if session_match:
        if user_name and session_match.group(1).strip() == user_name:
            persona_name = session_match.group(2).strip()
        elif user_name is None:
            user_name = session_match.group(1).strip()
            persona_name = session_match.group(2).strip()
    return user_name, persona_name


def _extract_chat_lines(chat_body: str, user_name: str | None, persona_name: str | None) -> list[ChatLine]:
    """解析聊天段：每行 [时间] 说话人: 内容，按已知昵称标注 is_user"""
    records = []
    for match in _CHAT_LINE_RE.finditer(chat_body):
        ts, speaker, text = match.group(1), match.group(2), match.group(3)
        if user_name is not None and speaker == user_name:
            is_user = True
        elif persona_name is not None and speaker == persona_name:
            is_user = False
        else:
            is_user = None
        records.append(ChatLine(ts=ts, speaker=speaker, text=text, is_user=is_user))
    return records


def _clean_section_body(body: str) -> str | None:
    """段落体清洗：去首尾空白，空内容归一为 None（调用点已保证段存在）"""
    cleaned = body.strip()
    return cleaned if cleaned else None
