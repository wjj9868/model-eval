# @author: ztwz
"""user_chat_analysis 训练数据导出：线上 vibee 库 manage_model_log → JSONL 数据集（只读）。

数据源口径：
- 来源表 manage_model_log（模型调用日志），task='user_chat_analysis'，status=1；
- prompt 列即模型完整输入（模板已渲染：用户信息 + 旧摘要/旧记忆 + 增量消息）；
- response 列为 OpenRouter/网关原始响应，模型输出在 choices[0].message.content；
- manage_chat_analysis 是程序校验后的业务落库（无输入侧），不能还原训练对，不作为导出来源；
- status=1 但 prompt 为空的记录是 force 压缩清空路径（未调模型），已被过滤。

两阶段导出（EXPLAIN ANALYZE 实测：主键顺序扫每凑 1000 匹配行需回表 3.2 万行、约 4 秒，
冷数据下回表 IO 是瓶颈，故拆为两阶段把回表从 ~245 万次降到 ~8.7 万次）：
- 阶段1：覆盖索引 uniq_task_uuid(task,uuid) 扫全部目标 id（InnoDB 二级索引叶子自带主键 id，
  零回表），一次查询、应用侧排序分片；
- 阶段2：按 id 分批 IN 主键点查，有效性过滤下推 SQL，只读真正需要的行。

样本有效性校验（与后端 OpenRouterTaskCaller 解析规则对齐）：
- 输出提取三级优先：message.content > message.reasoning > choice.text；
  兼容网关路径 Anthropic 原生格式（content blocks）；
- 提取出的输出必须可解析为 JSON 对象且含 rolling_summary 字段（结构化漂移样本丢弃）；
- prompt+output 完全相同的样本按 sha1 去重。

其他性能设计：
- 阶段1 索引只扫一遍、阶段2 全部主键点查：无 OFFSET、无 filesort、无无用行整读；
- autocommit=True，批间释放快照，不长持 undo 防止 purge 滞后影响线上；
- 输出 UTF-8 JSONL（ensure_ascii=False + 紧凑分隔符），逐批 flush 落盘，内存恒定；
- 断点续传：每批更新 <out>.state.json，--resume 从 last_id 续导（追加模式）。

用法：
  # 1) 先小量验证样本质量
  python scripts/export_user_chat_analysis.py --limit 200 --out data/sample.jsonl
  # 2) 确认无误后全量导出（预计约 7.5 万条，输出数百 MB）
  python scripts/export_user_chat_analysis.py
  # 3) 中断后续传
  python scripts/export_user_chat_analysis.py --resume
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import pymysql

# 线上只读账号（命令行参数可覆盖）
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 3307
DEFAULT_USER = "analyst"
DEFAULT_PASSWORD = "bVZFLUujzPTA"
DEFAULT_DB = "vibee"

# 每批行数：阶段2 主键点查的 IN 列表大小
DEFAULT_BATCH_SIZE = 1000
# 进度日志输出间隔（批）
LOG_EVERY_BATCHES = 5

# 阶段1：覆盖索引取全部目标 id（二级索引叶子自带主键 id，零回表；id 排序在应用侧做）
ID_SCAN_SQL = (
    "SELECT id FROM manage_model_log FORCE INDEX (uniq_task_uuid) "
    "WHERE task = 'user_chat_analysis' AND id > %s"
)

# 阶段2：主键点查取行，有效性过滤全部下推 SQL
# 注意：含 '{}' 字面量，不能用 str.format（会把 '{}' 当占位符），拼占位符用 replace
ROW_FETCH_SQL = (
    "SELECT id, prompt, response FROM manage_model_log "
    "WHERE id IN ({placeholders}) "
    "AND status = 1 AND prompt != '' AND response NOT IN ('', '{}')"
)


def extract_model_output(response_json: str) -> str | None:
    """从响应原文提取模型输出文本。

    支持三种格式（与后端各时期调用路径对齐）：
    - OpenRouter/OpenAI 兼容（含网关回填体）：choices[0].message.content > reasoning > choice.text；
    - Gemini 原生（早期直连路径）：candidates[0].content.parts[].text；
    - Anthropic 原生（网关路径）：content blocks。
    解析失败返回 None。
    """
    try:
        obj = json.loads(response_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None

    # OpenRouter / OpenAI 兼容格式
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message")
        if isinstance(message, dict):
            for field in ("content", "reasoning"):
                text = message.get(field)
                if isinstance(text, str) and text.strip():
                    return text
        text = choice.get("text")
        if isinstance(text, str) and text.strip():
            return text
        return None

    # Gemini 原生格式（早期直连路径）
    candidates = obj.get("candidates")
    if isinstance(candidates, list) and candidates:
        first = candidates[0] if isinstance(candidates[0], dict) else {}
        content = first.get("content")
        if isinstance(content, dict):
            parts = content.get("parts")
            if isinstance(parts, list):
                joined = "".join(
                    part.get("text", "") for part in parts if isinstance(part, dict)
                )
                if joined.strip():
                    return joined
        return None

    # Anthropic 原生格式（网关路径）
    blocks = obj.get("content")
    if isinstance(blocks, list):
        joined = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if joined.strip():
            return joined
    return None


def is_valid_output(output: str) -> bool:
    """结构化输出校验：可解析为 JSON 对象且含核心字段 rolling_summary"""
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and "rolling_summary" in parsed


def build_record(fmt: str, prompt: str, output: str) -> dict:
    """构造一行数据集记录：raw 为 prompt/output 字段，messages 为 OpenAI 对话格式"""
    if fmt == "messages":
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output},
            ],
        }
    return {"prompt": prompt, "output": output}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 user_chat_analysis 训练数据集（线上只读）")
    parser.add_argument("--out", default="data/user_chat_analysis.jsonl", help="输出 JSONL 路径")
    parser.add_argument(
        "--format", choices=("raw", "messages"), default="raw",
        help="raw: prompt/output 字段；messages: OpenAI messages 对话格式",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="阶段2 每批点查行数")
    parser.add_argument("--limit", type=int, default=0, help="最多导出样本数，0 为不限")
    parser.add_argument("--since-id", type=int, default=0, help="从指定 log id 之后导出")
    parser.add_argument("--resume", action="store_true", help="从 <out>.state.json 断点续导（追加写入）")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--db", default=DEFAULT_DB)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state_file = out.parent / (out.name + ".state.json")

    since_id = args.since_id
    if args.resume:
        if not state_file.exists():
            sys.exit("--resume 需要 state 文件（先全量导出，或改用 --since-id）")
        since_id = max(since_id, json.loads(state_file.read_text(encoding="utf-8")).get("last_id", 0))

    # autocommit=True：每批独立读视图，批间释放快照，避免长事务拖慢线上 purge
    # ssl_disabled=True：线上 3307 为明文端口，新版 pymysql 默认 SSL 升级会握手失败
    conn = pymysql.connect(
        host=args.host, port=args.port, user=args.user, password=args.password,
        database=args.db, charset="utf8mb4", connect_timeout=10, read_timeout=60,
        autocommit=True, ssl_disabled=True,
    )

    # 阶段1：覆盖索引一次取全部目标 id（纯索引扫描，无回表）
    with conn.cursor() as cur:
        cur.execute(ID_SCAN_SQL, (since_id,))
        ids = sorted(row[0] for row in cur.fetchall())

    # 小样导出时批量随 limit 联动，避免 limit=200 却按 1000 行/批拉大字段
    effective_batch = min(args.batch_size, args.limit) if args.limit else args.batch_size
    total_batches = (len(ids) + effective_batch - 1) // effective_batch
    print(f"id scan done: {len(ids)} rows, {total_batches} batches", flush=True)

    seen_hashes: set[str] = set()
    exported = skipped_extract = skipped_invalid = duplicated = 0
    last_id = since_id
    started = time.monotonic()
    # 续传为追加写入，全新导出覆盖
    mode = "a" if args.resume else "w"

    try:
        # .gz 后缀自动 gzip 压缩输出（文本压缩比约 5:1，磁盘与回传都省）
        if out.suffix == ".gz":
            opener, mode_arg = gzip.open, mode + "t"
        else:
            opener, mode_arg = open, mode
        with opener(out, mode_arg, encoding="utf-8", newline="\n") as f, conn.cursor() as cur:
            for batch_no in range(total_batches):
                chunk = ids[batch_no * effective_batch : (batch_no + 1) * effective_batch]
                placeholders = ",".join(["%s"] * len(chunk))
                cur.execute(ROW_FETCH_SQL.replace("{placeholders}", placeholders), chunk)
                rows_by_id = {row[0]: row for row in cur.fetchall()}
                for log_id in chunk:
                    row = rows_by_id.get(log_id)
                    if row is None:
                        continue
                    _, prompt, response = row
                    output = extract_model_output(response or "")
                    if output is None:
                        skipped_extract += 1
                        continue
                    if not is_valid_output(output):
                        skipped_invalid += 1
                        continue
                    digest = hashlib.sha1(f"{prompt}\x00{output}".encode("utf-8")).hexdigest()
                    if digest in seen_hashes:
                        duplicated += 1
                        continue
                    seen_hashes.add(digest)
                    record = build_record(args.format, prompt, output)
                    f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    exported += 1
                    if args.limit and exported >= args.limit:
                        break
                f.flush()
                # 游标推进到本批末尾（含被 SQL 过滤的 id），断点状态每批落盘
                last_id = chunk[-1]
                state_file.write_text(
                    json.dumps({"last_id": last_id, "exported": exported, "format": args.format}),
                    encoding="utf-8",
                )
                if (batch_no + 1) % LOG_EVERY_BATCHES == 0 or (batch_no + 1) == total_batches:
                    elapsed = time.monotonic() - started
                    print(
                        f"batch {batch_no + 1}/{total_batches}, exported={exported}, "
                        f"skipped={skipped_extract + skipped_invalid}, dup={duplicated}, "
                        f"elapsed={elapsed:.0f}s",
                        flush=True,
                    )
                if args.limit and exported >= args.limit:
                    break
    finally:
        conn.close()

    elapsed = time.monotonic() - started
    print(
        f"done: exported={exported}, skipped_extract={skipped_extract}, "
        f"skipped_invalid={skipped_invalid}, duplicated={duplicated}, "
        f"last_id={last_id}, elapsed={elapsed:.0f}s\noutput: {out}"
    )


if __name__ == "__main__":
    main()
