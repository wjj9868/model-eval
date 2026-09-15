# @author: ztwz
"""工作台统计路由。"""
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.enums import BatchStatus, TaskStatus
from app.models import EvalBatch, EvalTask, ModelConfig, PromptCase
from app.schemas import BatchListVO, StatsOut
from app.services.batch_service import BatchService

router = APIRouter(prefix="/api/stats", tags=["工作台"])


@router.get("", response_model=StatsOut)
def get_stats(db: Session = Depends(get_db)):
    """工作台统计：资源数量、任务成功率、最近批次"""

    model_count = db.scalar(select(func.count()).select_from(ModelConfig).where(ModelConfig.enabled.is_(True))) or 0
    prompt_count = db.scalar(select(func.count()).select_from(PromptCase).where(PromptCase.enabled.is_(True))) or 0
    batch_count = db.scalar(select(func.count()).select_from(EvalBatch)) or 0

    task_stats = dict(
        db.execute(
            select(EvalTask.status, func.count()).group_by(EvalTask.status)
        ).all()
    )
    succeed = task_stats.get(TaskStatus.SUCCESS, 0)
    failed = task_stats.get(TaskStatus.FAILED, 0)
    total = succeed + failed

    rows, _ = BatchService(db).list_batches(1, 10)
    recent = [
        BatchListVO(**{field: getattr(batch, field) for field in BatchListVO.model_fields if field != "status"}, status=status)
        for batch, status in rows
    ]
    return StatsOut(
        model_count=model_count,
        prompt_count=prompt_count,
        batch_count=batch_count,
        total_tasks=total,
        succeed_tasks=succeed,
        failed_tasks=failed,
        success_rate=round(succeed / total, 4) if total else None,
        recent_batches=recent,
    )