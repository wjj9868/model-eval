# @author: ztwz
"""归因误判审计：拆解 clean_train_data 剔除样本的真实原因（只输出计数，不输出内容）。

针对原 speaker_attribution 判定的三个质疑逐一量化：
1. 回声偏置：persona 常复述/提问用户内容，最佳匹配行落 persona ≠ 事实来自 persona。
   逐条改判：persona 行达强证据 且 用户行也存在弱+证据 → ECHO（非真错误）。
2. 记忆延续：prompt 含旧记忆快照/旧滚动摘要，teacher 合法沿用旧事实，
   增量对话无证据 → 原方法记 grounding=0 → speaker=0.5 被误剔除。
3. 说话人识别失败：user_name/persona_name 提取失败 → is_user=None 中性分 0.5 被误剔除。
"""
from __future__ import annotations

import collections
import json

from app.evaluation.embeddings import EmbeddingClient
from app.evaluation.output_parser import memory_items
from app.evaluation.prompt_parser import parse_prompt
from app.evaluation.scoring.embedding_scorer import (
    GROUNDING_STRONG,
    GROUNDING_WEAK,
    _VectorCache,
    _evidence_level,
    _word_set,
)


def _best_level(text: str, candidates: list[str], cache: _VectorCache) -> float:
    """text 对候选行集合的最高证据分级"""
    words = _word_set(text)
    best = 0.0
    for cand in candidates:
        sim = cache.cosine(text, cand)
        overlap = len(words & _word_set(cand)) / len(words) if words else 0.0
        best = max(best, _evidence_level(sim, overlap))
    return best


def _old_sources(ctx, prompt: str) -> list[str]:
    """旧记忆快照行 + 旧滚动摘要 + 用户基本信息段（合法延续/背景证据源）"""
    sources: list[str] = []
    if ctx.old_memory_snapshot:
        sources.extend(ln.strip("- ").strip() for ln in ctx.old_memory_snapshot.splitlines() if ln.strip())
    if ctx.old_rolling_summary:
        sources.append(ctx.old_rolling_summary)
    # 用户基本信息段：parser 未保留原文，这里直接切段落取行
    from app.evaluation.prompt_parser import _split_sections
    user_info = _split_sections(prompt).get("user_info", "")
    sources.extend(ln.strip("- ").strip() for ln in user_info.splitlines() if ln.strip())
    return sources


def main() -> None:
    embedder = EmbeddingClient()
    cache = _VectorCache(embedder)

    item_cls: collections.Counter = collections.Counter()   # 逐条事实分类
    sample_cls: collections.Counter = collections.Counter()  # 逐样本分类
    name_fail = 0          # 用户名或人设名提取失败样本数
    unknown_lines = 0      # is_user=None 的聊天行总数
    total_lines = 0

    with open("data/sample.jsonl", "r", encoding="utf-8") as fin:
        for line in fin:
            record = json.loads(line)
            ctx = parse_prompt(record["prompt"])
            output = json.loads(record["output"])
            # 构造仅用于 memory_items 的最小 AnalysisResult
            from app.evaluation.output_parser import parse_output
            result = parse_output(record["output"])
            facts = [t for texts in memory_items(result).values() for t in texts]

            if ctx.user_name is None or ctx.persona_name is None:
                name_fail += 1
            for cl in ctx.chat_records:
                total_lines += 1
                if cl.is_user is None:
                    unknown_lines += 1

            user_texts = [cl.text for cl in ctx.chat_records if cl.is_user is True]
            persona_texts = [cl.text for cl in ctx.chat_records if cl.is_user is False]
            old_texts = _old_sources(ctx, record["prompt"])

            # 预编码
            cache.preload(facts + user_texts + persona_texts + old_texts)

            sample_true_err = False
            for fact in facts:
                lvl_persona = _best_level(fact, persona_texts, cache)
                lvl_user = _best_level(fact, user_texts, cache)
                lvl_old = _best_level(fact, old_texts, cache)

                if lvl_persona >= 1.0 and lvl_user < 0.5:
                    cls = "TRUE_ERR"       # 仅 persona 行支持：真归因错误
                    sample_true_err = True
                elif lvl_persona >= 1.0 and lvl_user >= 0.5:
                    cls = "ECHO"           # 双方都有证据：回声偏置误报
                elif lvl_user >= 0.5:
                    cls = "USER_OK"        # 用户行支持：正常
                elif lvl_old >= 0.5:
                    cls = "CARRIED"        # 旧记忆延续：合法，原方法误罚
                else:
                    cls = "NO_EVIDENCE"    # 增量对话与旧记忆都无证据
                item_cls[cls] += 1

            sample_cls["HAS_TRUE_ERR" if sample_true_err else "NO_TRUE_ERR"] += 1

    print("=== 逐条事实分类（共 %d 条）===" % sum(item_cls.values()))
    for k, v in item_cls.most_common():
        print(f"{k}: {v} ({v / sum(item_cls.values()):.1%})")
    print("\n=== 逐样本 ===")
    for k, v in sample_cls.most_common():
        print(f"{k}: {v}")
    print(f"\n名字提取失败样本: {name_fail}/200")
    print(f"聊天行 is_user=None 占比: {unknown_lines}/{total_lines}")


if __name__ == "__main__":
    main()
