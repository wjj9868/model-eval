# @author: ztwz
"""测试用例路由：增删改查、批量导入（粘贴/文件 txt/csv/jsonl）、批量删除、删除保护。"""
import csv
import hashlib
import json
import logging
from io import StringIO
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.exceptions import BusinessError
from app.models import EvalTask, PromptCase
from app.schemas import ImportResult, PageOut, PromptBatchDelete, PromptCreate, PromptOut, PromptUpdate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/prompts", tags=["测试用例"])

MAX_CONTENT_LENGTH = 2000


def _content_hash(content: str) -> str:
    """内容 SHA-256（去重键）"""

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _get_or_404(db: Session, prompt_id: int) -> PromptCase:
    prompt = db.get(PromptCase, prompt_id)
    if prompt is None:
        raise BusinessError("用例不存在", status_code=404)
    return prompt


def _check_duplicate(db: Session, content: str, exclude_id: int | None = None) -> None:
    """内容级查重（排除自身）"""

    stmt = select(PromptCase.id).where(PromptCase.content_hash == _content_hash(content))
    if exclude_id is not None:
        stmt = stmt.where(PromptCase.id != exclude_id)
    if db.scalar(stmt) is not None:
        raise BusinessError("已存在相同内容的用例")


@router.get("", response_model=PageOut[PromptOut])
def list_prompts(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    keyword: str | None = Query(None, max_length=100),
    category: str | None = Query(None, max_length=50),
    db: Session = Depends(get_db),
):
    """用例分页列表（关键字 + 分类筛选）"""

    stmt = select(PromptCase)
    if keyword:
        stmt = stmt.where(PromptCase.content.like(f"%{keyword}%"))
    if category:
        stmt = stmt.where(PromptCase.category == category)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    items = db.scalars(stmt.order_by(PromptCase.id.desc()).offset((page - 1) * page_size).limit(page_size)).all()
    return PageOut[PromptOut](items=list(items), total=total, page=page, page_size=page_size)


@router.get("/categories", response_model=list[str])
def list_categories(db: Session = Depends(get_db)):
    """全部分类（用于筛选下拉）"""

    return list(db.scalars(select(PromptCase.category).distinct().order_by(PromptCase.category)).all())


@router.post("", response_model=PromptOut)
def create_prompt(payload: PromptCreate, db: Session = Depends(get_db)):
    """新建用例"""

    _check_duplicate(db, payload.content)
    prompt = PromptCase(
        content_hash=_content_hash(payload.content),
        content=payload.content,
        category=payload.category,
        expected_answer=payload.expected_answer,
        enabled=payload.enabled,
    )
    db.add(prompt)
    db.commit()
    db.refresh(prompt)
    return prompt


@router.put("/{prompt_id}", response_model=PromptOut)
def update_prompt(prompt_id: int, payload: PromptUpdate, db: Session = Depends(get_db)):
    """修改用例（内容变化时重建去重键）"""

    prompt = _get_or_404(db, prompt_id)
    _check_duplicate(db, payload.content, exclude_id=prompt_id)
    prompt.content = payload.content
    prompt.content_hash = _content_hash(payload.content)
    prompt.category = payload.category
    prompt.expected_answer = payload.expected_answer
    prompt.enabled = payload.enabled
    db.commit()
    db.refresh(prompt)
    return prompt


@router.delete("/{prompt_id}")
def delete_prompt(prompt_id: int, db: Session = Depends(get_db)):
    """删除用例（已被评测任务引用时禁止）"""

    prompt = _get_or_404(db, prompt_id)
    referenced = db.scalar(select(func.count()).select_from(EvalTask).where(EvalTask.prompt_id == prompt_id))
    if referenced:
        raise BusinessError(f"该用例已被 {referenced} 条评测任务引用，不能删除")
    db.delete(prompt)
    db.commit()
    return {"deleted": True}


@router.post("/batch-delete")
def batch_delete(payload: PromptBatchDelete, db: Session = Depends(get_db)):
    """批量删除（原子：任一被引用则整批拒绝）"""

    ids = list(dict.fromkeys(payload.ids))
    referenced = db.scalar(select(func.count()).select_from(EvalTask).where(EvalTask.prompt_id.in_(ids)))
    if referenced:
        raise BusinessError(f"所选用例中有 {referenced} 条被评测任务引用，请移除后重试")
    deleted = db.query(PromptCase).filter(PromptCase.id.in_(ids)).delete(synchronize_session=False)
    db.commit()
    return {"deleted": deleted}


@router.post("/import", response_model=ImportResult)
async def import_prompts(
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    default_category: str = Form("默认"),
    db: Session = Depends(get_db),
):
    """批量导入：粘贴文本（每行一条）或上传 txt / csv / jsonl"""

    if file is not None:
        raw = (await file.read()).decode("utf-8", errors="replace")
        parsed_lines = _parse_source(raw, Path(file.filename or "").suffix.lower(), default_category)
    elif text and text.strip():
        parsed_lines = _parse_source(text, "", default_category)
    else:
        raise BusinessError("请提供粘贴文本或上传文件")

    try:
        return _import_lines(db, parsed_lines)
    except IntegrityError:
        db.rollback()
        raise BusinessError("导入存在并发冲突，请重试") from None


def _parse_source(raw: str, suffix: str, default_category: str) -> list[tuple[str, str]]:
    """按来源格式解析 (content, category) 列表（不包含空行）"""

    if suffix in (".json", ".jsonl"):
        parsed = []
        for line_no, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BusinessError(f"JSON 第 {line_no} 行解析失败：{exc.msg}") from None
            content = str(row.get("content", "")).strip()
            parsed.append((content, str(row.get("category") or default_category)))
        return parsed
    if suffix == ".csv":
        parsed = []
        for row in csv.reader(StringIO(raw)):
            if not row or not row[0].strip():
                continue
            content = row[0].strip()
            if content.lower() == "content" and len(row) > 1 and row[1].strip().lower() == "category":
                continue  # 表头
            parsed.append((content, row[1].strip() if len(row) > 1 and row[1].strip() else default_category))
        return parsed
    return [(line.strip(), default_category) for line in raw.splitlines() if line.strip()]


def _import_lines(db: Session, parsed_lines: list[tuple[str, str]]) -> ImportResult:
    """去重落库：内容已存在则跳过，空/超长计为无效"""

    existing = set(db.scalars(select(PromptCase.content_hash)).all())
    new_items = []
    skipped = invalid = 0
    for content, category in parsed_lines:
        if not content or len(content) > MAX_CONTENT_LENGTH:
            invalid += 1
            continue
        content_hash = _content_hash(content)
        if content_hash in existing:
            skipped += 1
            continue
        existing.add(content_hash)
        new_items.append(
            PromptCase(content_hash=content_hash, content=content, category=(category or "默认")[:50])
        )
    if new_items:
        db.add_all(new_items)
        db.commit()
    return ImportResult(imported_count=len(new_items), skipped_count=skipped, invalid_count=invalid)