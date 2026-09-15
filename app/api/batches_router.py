# @author: ztwz
"""评测批次路由：新建（落库+入队）、列表、详情（按用例分组对比）、重试失败项。"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.enums import BatchStatus, Provider, TaskStatus
from app.exceptions import BusinessError
from app.models import EvalBatch, EvalTask, ModelConfig, PromptCase
from app.schemas import BatchCreate, BatchDetailVO, BatchListVO, ModelBriefVO, PageOut, PromptGroupVO, TaskVO
from app.services.batch_service import BatchService, derive_batch_status
from app.tasks.dispatcher import enqueue_task_ids

router = APIRouter(prefix="/api/batches", tags=["评测批次"])


def _to_list_vo(batch: EvalBatch, status: BatchStatus) -> BatchListVO:
    """ORM 批次转列表 VO（status 用派生态）"""

    fields = {field: getattr(batch, field) for field in BatchListVO.model_fields if field != "status"}
    return BatchListVO(**fields, status=status)


@router.post("", response_model=BatchListVO)
def create_batch(payload: BatchCreate, db: Session = Depends(get_db)):
    """新建评测批次：单事务落库后异步入队（入队失败不影响批次已落库，可重试）"""

    service = BatchService(db)
    batch = service.create_batch(payload.name, payload.model_ids, payload.prompt_ids, payload.remark)
    task_ids = list(
        db.scalars(select(EvalTask.id).where(EvalTask.batch_id == batch.id, EvalTask.status == TaskStatus.PENDING)).all()
    )
    # eager 模式（测试）下入队即同步执行；先结束当前读事务，再重新读取以获得最新计数
    enqueue_task_ids(task_ids)
    db.commit()
    batch = db.get(EvalBatch, batch.id, populate_existing=True)
    return _to_list_vo(batch, derive_batch_status(batch))


@router.get("", response_model=PageOut[BatchListVO])
def list_batches(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    status: BatchStatus | None = Query(None),
    db: Session = Depends(get_db),
):
    """批次列表（按派生态筛选，倒序）"""

    rows, total = BatchService(db).list_batches(page, page_size, status)
    items = [_to_list_vo(batch, derived) for batch, derived in rows]
    return PageOut[BatchListVO](items=items, total=total, page=page, page_size=page_size)


@router.get("/{batch_id}", response_model=BatchDetailVO)
def batch_detail(
    batch_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    status: TaskStatus | None = Query(None),
    db: Session = Depends(get_db),
):
    """批次详情：按用例分页，同输入各模型输出并列（任务状态可筛选）"""

    batch = db.get(EvalBatch, batch_id, populate_existing=True)
    if batch is None:
        raise BusinessError("批次不存在", status_code=404)

    base = (
        select(EvalTask, PromptCase, ModelConfig)
        .join(PromptCase, PromptCase.id == EvalTask.prompt_id)
        .join(ModelConfig, ModelConfig.id == EvalTask.model_id)
        .where(EvalTask.batch_id == batch_id)
    )

    # 按用例 ID 分页，保证一个用例的模型输出不被切断
    prompt_stmt = select(EvalTask.prompt_id).where(EvalTask.batch_id == batch_id)
    if status is not None:
        prompt_stmt = prompt_stmt.where(EvalTask.status == status)
        base = base.where(EvalTask.status == status)
    matched_total = db.scalar(select(func.count()).select_from(prompt_stmt.distinct().subquery())) or 0
    page_prompt_ids = list(
        db.scalars(
            prompt_stmt.distinct().order_by(EvalTask.prompt_id).offset((page - 1) * page_size).limit(page_size)
        ).all()
    )

    models_vo = list(
        db.scalars(
            select(ModelConfig)
            .join(EvalTask, EvalTask.model_id == ModelConfig.id)
            .where(EvalTask.batch_id == batch_id)
            .distinct()
            .order_by(ModelConfig.id)
        ).all()
    )

    groups: list[PromptGroupVO] = []
    if page_prompt_ids:
        rows = db.execute(
            base.where(EvalTask.prompt_id.in_(page_prompt_ids)).order_by(EvalTask.prompt_id, EvalTask.model_id)
        ).all()
        for task, prompt, model in rows:
            results = TaskVO(
                task_id=task.id,
                model_id=task.model_id,
                model_name=model.name,
                provider=Provider(model.provider),
                status=TaskStatus(task.status),
                output=task.output,
                error=task.error,
                response_time_ms=task.response_time_ms,
                attempt_count=task.attempt_count,
                started_at=task.started_at,
                finished_at=task.finished_at,
            )
            if not groups or groups[-1].prompt_id != prompt.id:
                groups.append(PromptGroupVO(prompt_id=prompt.id, content=prompt.content, category=prompt.category, results=[results]))
            else:
                groups[-1].results.append(results)

    fields = {field: getattr(batch, field) for field in BatchListVO.model_fields if field != "status"}
    return BatchDetailVO(
        **fields,
        status=derive_batch_status(batch),
        models=[ModelBriefVO(id=m.id, name=m.name, provider=Provider(m.provider), model_name=m.model_name) for m in models_vo],
        prompts=groups,
        page=page,
        page_size=page_size,
        matched_total=matched_total,
    )


@router.post("/{batch_id}/retry-failed")
def retry_failed(batch_id: int, db: Session = Depends(get_db)):
    """重试失败任务：FAILED 重置为 PENDING 并重新入队"""

    service = BatchService(db)
    task_ids = service.retry_failed(batch_id)
    enqueue_task_ids(task_ids)
    return {"retried_count": len(task_ids)}