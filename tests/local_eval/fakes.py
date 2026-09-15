# @author: ztwz
"""确定性 mock embedder 与输出/prompt 工厂：纯类实现，供 conftest 与测试复用。"""
import json

import numpy as np


class MockEmbedder:
    """确定性 mock：按文本哈希生成正交基向量，相同文本必得相同向量。

    - 完全相同文本 → cosine = 1.0
    - 不同文本 → cosine ≈ 0（正交基几乎不重合）
    - 语义等价对通过 alias_map (a, b) -> sim 控制
    """

    def __init__(self, alias_map: dict[tuple[str, str], float] | None = None):
        self._alias_map = alias_map or {}
        self._vectors: dict[str, np.ndarray] = {}
        self.dim = 128

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts])

    def _vector(self, text: str) -> np.ndarray:
        if text not in self._vectors:
            self._vectors[text] = self._make_vector(text)
        return self._vectors[text]

    def _make_vector(self, text: str) -> np.ndarray:
        """生成严格正交的 one-hot 单位向量；别名对按目标相似度混合"""
        for (a, b), sim in self._alias_map.items():
            if text == a and b in self._vectors:
                return self._blend(self._vectors[b], sim)
            if text == b and a in self._vectors:
                return self._blend(self._vectors[a], sim)
        vec = np.zeros(self.dim, dtype=np.float64)
        vec[hash(text) % self.dim] = 1.0
        return vec

    def _blend(self, base: np.ndarray, sim: float) -> np.ndarray:
        """生成与 base 余弦相似度恰为 sim 的单位向量（正交分量取空闲维度）"""
        base_idx = int(np.argmax(base))
        # 正交方向取 base 位置的相邻维度，避开 base 本身
        other_idx = (base_idx + 1 + (hash(sim) % (self.dim - 1))) % self.dim
        if other_idx == base_idx:
            other_idx = (base_idx + 1) % self.dim
        vec = np.zeros(self.dim, dtype=np.float64)
        vec[base_idx] = sim
        vec[other_idx] = np.sqrt(max(0.0, 1.0 - sim**2))
        return vec


def make_teacher_output(
    facts: list[str] | None = None,
    preferences: list[str] | None = None,
    avoid_topics: list[str] | None = None,
    active_events: list[dict] | None = None,
    open_loops: list[str] | None = None,
    rolling_summary: str = "User discussed work stress and weekend plans with the persona.",
    primary_intent: str = "CASUAL_ENTERTAINMENT",
    secondary_intents: list[str] | None = None,
    reasoning: str = "User asked casual questions to pass time.",
    triggered: bool = False,
    reaction_type: str = "NOT_TRIGGERED",
    analysis: str = "No paid content mentioned.",
    naturalness_score: int = 8,
    identity_awareness: str = "NOT_DETECTED",
) -> str:
    """构造合法 teacher 输出 JSON 字符串"""
    return json.dumps(
        {
            "analysis_summary": {
                "overall_impression": "Casual friendly chat.",
                "user_engagement_level": "MEDIUM",
            },
            "user_intent": {
                "primary_intent": primary_intent,
                "secondary_intents": secondary_intents or [],
                "reasoning": reasoning,
            },
            "paid_reaction": {
                "triggered": triggered,
                "reaction_type": reaction_type,
                "analysis": analysis,
            },
            "ai_performance": {
                "naturalness_score": naturalness_score,
                "identity_awareness": identity_awareness,
                "naturalness_reasoning": "Fluent replies.",
            },
            "rolling_summary": rolling_summary,
            "memory_snapshot": {
                "facts": facts or [],
                "preferences": preferences or [],
                "avoid_topics": avoid_topics or [],
                "active_events": active_events or [],
                "open_loops": open_loops or [],
            },
        },
        ensure_ascii=False,
    )


def make_prompt(
    user_name: str = "Jay",
    persona_name: str = "Raven",
    chat_lines: list[tuple[str, str, str]] | None = None,
    old_summary: str = "",
    old_memory: str = "",
) -> str:
    """构造 v3 英文模板 prompt（含对话记录，供 grounding/speaker 测试）"""
    if chat_lines is None:
        chat_lines = [
            ("2026-09-15 02:41:00", "Jay", "I'm an engineer from Texas, I love hiking."),
            ("2026-09-15 02:42:16", "Raven", "Nice! I enjoy yoga every morning."),
        ]
    chat_block = "\n".join(f"[{ts}] {speaker}: {text}" for ts, speaker, text in chat_lines)
    return (
        "## ROLE\nYou are a conversation analysis and memory management expert.\n\n"
        "## USER BASIC INFORMATION\n"
        f"- Name: {user_name}, Gender: Man, Age: 30\n\n"
        "## CURRENT DATE\n2026-09-15\n\n"
        "## OLD ROLLING SUMMARY\n"
        f"{old_summary}\n\n"
        "## OLD MEMORY SNAPSHOT\n"
        f"{old_memory}\n\n"
        "## INCREMENTAL CHAT RECORDS\n"
        f"会话：{user_name} - {persona_name}\n"
        f"{chat_block}\n\n"
        "## ANALYSIS REQUIREMENTS (analysis_summary / user_intent / paid_reaction / ai_performance)\n\n"
        "## ROLLING SUMMARY REQUIREMENTS (rolling_summary)\n\n"
        "## MEMORY SNAPSHOT REQUIREMENTS (memory_snapshot)\n"
    )
