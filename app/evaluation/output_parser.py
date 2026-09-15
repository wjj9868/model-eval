# @author: ztwz
"""模型输出解析：任意文本（含 Base 模型乱输出）容错提取结构化 JSON。

与后端 UserChatSchema 六字段对齐：analysis_summary / user_intent / paid_reaction /
ai_performance / rolling_summary / memory_snapshot。永不抛异常，解析失败返回 valid=False。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# 结构化输出的六个顶层字段（与 UserChatSchema.schema() 对齐）
REQUIRED_FIELDS = ("analysis_summary", "user_intent", "paid_reaction",
                   "ai_performance", "rolling_summary", "memory_snapshot")

# memory_snapshot 合法分类 → 条数上限（与后端 MEMORY_LIMITS 对齐）
MEMORY_CATEGORIES = ("facts", "preferences", "avoid_topics", "active_events", "open_loops")
MEMORY_CATEGORY_LIMITS = {"facts": 10, "preferences": 10, "avoid_topics": 5, "active_events": 5, "open_loops": 5}

# 代码围栏剥壳：```json ... ``` 或 ``` ... ```
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\s*```", re.S)


@dataclass
class AnalysisResult:
    """一次模型输出的解析结果。valid 表示成功提取出 JSON 对象，raw 为解析后的 dict。"""

    valid: bool
    parse_error: str | None = None
    raw: dict = field(default_factory=dict)


def parse_output(text: str) -> AnalysisResult:
    """容错解析模型输出文本，提取六字段结构化 JSON。

    解析顺序：剥代码围栏 → 直接 json.loads → 括号配对扫描提取最长合法 JSON 对象。
    """
    if not isinstance(text, str) or not text.strip():
        return AnalysisResult(False, "empty output")

    # 剥 ```json 围栏（保留围栏内内容；无围栏时用原文）
    candidates: list[str] = []
    stripped = text.strip()
    fenced = _FENCE_RE.search(stripped)
    candidates.append(fenced.group(1) if fenced else stripped)

    for candidate in candidates:
        obj = _try_loads_object(candidate)
        if obj is not None:
            return AnalysisResult(True, None, obj)

    # 括号配对扫描：从每个 '{' 起提取平衡块，取最长可解析对象
    best = _extract_balanced_json(stripped)
    if best is not None:
        return AnalysisResult(True, None, best)

    return AnalysisResult(False, "no valid JSON object found")


def _try_loads_object(text: str) -> dict | None:
    """尝试 json.loads 并要求结果为 dict，失败返回 None"""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _extract_balanced_json(text: str) -> dict | None:
    """括号配对扫描：从每个 '{' 起找字符串感知的平衡块，返回最长可解析的 JSON 对象。

    字符串感知：跳过双引号内的内容（含转义），避免 JSON 值里的花括号干扰配对。
    """
    best: dict | None = None
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        end = _find_balanced_end(text, start)
        if end is None:
            continue
        obj = _try_loads_object(text[start : end + 1])
        if obj is not None and (best is None or len(obj) > len(best)):
            best = obj
    return best


def _find_balanced_end(text: str, start: int) -> int | None:
    """从 text[start]='{' 起做字符串感知的括号配对，返回匹配 '}' 的下标，不平衡返回 None"""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def memory_items(result: AnalysisResult) -> dict[str, list[str]]:
    """提取 memory_snapshot 各分类的条目文本列表（active_events 对象取 content）。

    分类缺失返回空列表；非五分类的键忽略。供评分器做条目级匹配。
    """
    items: dict[str, list[str]] = {category: [] for category in MEMORY_CATEGORIES}
    if not result.valid:
        return items
    snapshot = result.raw.get("memory_snapshot")
    if not isinstance(snapshot, dict):
        return items
    for category in MEMORY_CATEGORIES:
        entries = snapshot.get(category)
        if not isinstance(entries, list):
            continue
        texts = []
        for entry in entries:
            if isinstance(entry, str):
                texts.append(entry)
            elif isinstance(entry, dict) and isinstance(entry.get("content"), str):
                texts.append(entry["content"])
        items[category] = texts
    return items


def normalize_text(text: str) -> str:
    """exact match 前的归一化：小写、去全部空白、去常见标点"""
    lowered = text.lower()
    stripped = re.sub(r"[\s\W_]+", "", lowered, flags=re.UNICODE)
    return stripped
