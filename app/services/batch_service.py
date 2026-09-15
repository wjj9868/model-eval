# @author: ztwz
"""评测批次业务：建批落库、重试失败项、终态派生。

调用方职责划分：
- 本服务只负责数据库事务（建批一次事务、重试一次事务）；
- 任务入队由 api 层调用 app.tasks.dispatcher 完成，与落库解耦，中断可重入。
"""
from sqlalchemy import case, func, select, update
from sqlalchemy.orm import Session

from app.enums import BatchStatus, TaskStatus
from app.exceptions import BusinessError
from app.models import EvalBatch, EvalTask, ModelConfig, PromptCase


def _derived_status_expr():
    """批次派生态的 SQL 表达式（维度：任务计数），用于列表筛选与返回"""

    return case(
        (
            EvalBatch.completed_tasks >= EvalBatch.total_tasks,
            case(
                (EvalBatch.failed_tasks == 0, BatchStatus.COMPLETED),
                (EvalBatch.succeed_tasks == 0, BatchStatus.FAILED),
                else_=BatchStatus.PARTIAL,
            ),
        ),
        (EvalBatch.completed_tasks > 0, BatchStatus.RUNNING),
        else_=BatchStatus.PENDING,
    )


def derive_batch_status(batch: EvalBatch) -> BatchStatus:
    """批次终态派生：所有任务完成后按成功/失败拆分，否则按进度区分"""

    if batch.total_tasks == 0:
        return BatchStatus.COMPLETED
    if batch.completed_tasks >= batch.total_tasks:
        if batch.failed_tasks == 0:
            return BatchStatus.COMPLETED
        if batch.succeed_tasks == 0:
            return BatchStatus.FAILED
        return BatchStatus.PARTIAL
    return BatchStatus.RUNNING if batch.completed_tasks > 0 else BatchStatus.PENDING


class BatchService:
    """评测批次核心业务"""

    def __init__(self, db: Session):
        self.db = db

    def create_batch(self, name: str, model_ids: list[int], prompt_ids: list[int], remark: str | None = None) -> EvalBatch:
        """单事务建批：校验 -> 落批次 -> 落 N×M 任务；返回批次（任务已落库未入队）"""

        if not name.strip():
            raise BusinessError("批次名称不能为空")
        model_ids = list(dict.fromkeys(model_ids))
        prompt_ids = list(dict.fromkeys(prompt_ids))
        if not model_ids or not prompt_ids:
            raise BusinessError("请至少选择一个模型和一个用例")

        models = self.db.scalars(
            select(ModelConfig).where(ModelConfig.id.in_(model_ids), ModelConfig.enabled.is_(True))
        ).all()
        if len(models) != len(model_ids):
            raise BusinessError("部分模型不存在或已禁用")
        prompts = self.db.scalars(select(PromptCase).where(PromptCase.id.in_(prompt_ids))).all()
        if len(prompts) != len(prompt_ids):
            raise BusinessError("部分用例不存在")

        batch = EvalBatch(
            name=name.strip(),
            remark=remark,
            model_count=len(models),
            prompt_count=len(prompts),
            total_tasks=len(models) * len(prompts),
        )
        self.db.add(batch)
        self.db.flush()  # 取得批次 ID
        self.db.add_all(
            EvalTask(batch_id=batch.id, model_id=model.id, prompt_id=prompt.id,
                     status=TaskStatus.PENDING, input=prompt.content)
            for prompt in prompts
            for model in models
        )
        self.db.commit()
        return batch

    def retry_failed(self, batch_id: int) -> list[int]:
        """将批次内 FAILED 任务重置为 PENDING（并回退计数），返回可重新投递的任务 ID"""

        batch = self.db.get(EvalBatch, batch_id)
        if batch is None:
            raise BusinessError("批次不存在", status_code=404)

        updated = self.db.execute(
            update(EvalTask)
            .where(EvalTask.batch_id == batch_id, EvalTask.status == TaskStatus.FAILED)
            .values(status=TaskStatus.PENDING, error=None, output=None, finished_at=None)
        )
        reset_count = updated.rowcount
        if reset_count == 0:
            self.db.commit()
            return []
        # 回退批次计数与完成时间，保证终态派生一致
        self.db.execute(
            update(EvalBatch)
            .where(EvalBatch.id == batch_id)
            .values(
                completed_tasks=EvalBatch.completed_tasks - reset_count,
                failed_tasks=EvalBatch.failed_tasks - reset_count,
                finished_at=None,
            )
            .execution_options(synchronize_session=False)
        )
        self.db.commit()
        return list(
            self.db.scalars(
                select(EvalTask.id).where(EvalTask.batch_id == batch_id, EvalTask.status == TaskStatus.PENDING)
            ).all()
        )

    def list_batches(self, page: int, page_size: int, status: BatchStatus | None = None) -> tuple[list[tuple[EvalBatch, BatchStatus]], int]:
        """批次列表（倒序），状态按派生态筛选，返回 (行, 总数)"""

        derived = _derived_status_expr()
        where = derived == status if status else None
        count = self.db.scalar(select(func.count()).select_from(EvalBatch).where(where)) if where is not None \
            else self.db.scalar(select(func.count()).select_from(EvalBatch))
        stmt = select(EvalBatch, derived.label("derived_status")).order_by(EvalBatch.id.desc()).offset(
            (page - 1) * page_size
        ).limit(page_size)
        if where is not None:
            stmt = stmt.where(where)
        rows = [(batch, BatchStatus(derived_status)) for batch, derived_status in self.db.execute(stmt).all()]
        return rows, count or 0