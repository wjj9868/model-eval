# @author: ztwz
"""评测任务执行：CAS 抢占 -> 调用模型 -> 终态回写（含批次计数）。

幂等与恢复设计：
- 行级 CAS：仅 PENDING 可抢占为 RUNNING，重复投递直接跳过；
- 网络类错误：归还抢占（回 PENDING）后交给 Celery 按阶梯重试，重试耗尽置 FAILED；
- worker 启动时将历史 RUNNING 归位 PENDING，崩溃遗留的任务可重新消费。
"""
import logging

from celery.signals import worker_process_init
from sqlalchemy import case, func, update

from app.database import SessionLocal
from app.enums import TaskStatus
from app.models import EvalBatch, EvalTask, ModelConfig
from app.services.model_runner import ModelCallError, call_model
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.run_eval.run_eval_task", max_retries=2)
def run_eval_task(self, task_id: int) -> str:
    """执行单个评测任务（幂等），返回执行结果描述"""

    db = SessionLocal()
    try:
        task = db.get(EvalTask, task_id)
        if task is None:
            return "task not found"

        # 1) CAS 抢占：仅 PENDING 可执行；重复投递（已被处理）直接跳过
        claimed = db.execute(
            update(EvalTask)
            .where(EvalTask.id == task_id, EvalTask.status == TaskStatus.PENDING)
            .values(status=TaskStatus.RUNNING, started_at=func.now(), attempt_count=EvalTask.attempt_count + 1)
        ).rowcount
        db.commit()
        if claimed == 0:
            return "already processed"

        config = db.get(ModelConfig, task.model_id)
        if config is None or not config.enabled:
            _fail(db, task_id, task.batch_id, "模型配置不存在或已禁用，无法评测")
            return "model missing"

        # 2) 调用模型；网络类错误阶梯重试，耗尽后计失败
        try:
            result = call_model(config, task.input)
        except ModelCallError as exc:
            if self.request.retries < self.max_retries:
                _release(db, task_id)
                raise self.retry(countdown=2 ** self.request.retries, exc=exc) from exc
            _fail(db, task_id, task.batch_id, str(exc))
            return "failed"
        except Exception as exc:  # noqa: BLE001 兜底异常，避免任务丢失
            logger.exception("评测任务异常，task_id=%s", task_id)
            if self.request.retries < self.max_retries:
                _release(db, task_id)
                raise self.retry(countdown=2 ** self.request.retries, exc=exc) from exc
            _fail(db, task_id, task.batch_id, f"内部错误：{type(exc).__name__}")
            return "failed"

        # 3) 终态回写（CAS 防重复累计计数）
        _succeed(db, task_id, task.batch_id, result.text, result.response_time_ms)
        return "ok"
    finally:
        db.close()


def _release(db, task_id: int) -> None:
    """归还抢占：RUNNING 归位 PENDING，供 Celery 重试时再次抢占"""

    db.execute(
        update(EvalTask)
        .where(EvalTask.id == task_id, EvalTask.status == TaskStatus.RUNNING)
        .values(status=TaskStatus.PENDING, started_at=None)
    )
    db.commit()


def _succeed(db, task_id: int, batch_id: int, output: str, response_time_ms: int) -> None:
    """成功回写：任务置 SUCCESS，批次成功计数原子自增"""

    updated = db.execute(
        update(EvalTask)
        .where(EvalTask.id == task_id, EvalTask.status == TaskStatus.RUNNING)
        .values(status=TaskStatus.SUCCESS, output=output, response_time_ms=response_time_ms, finished_at=func.now())
    ).rowcount
    if updated:
        _bump_batch(db, batch_id, succeed=True)
    db.commit()


def _fail(db, task_id: int, batch_id: int, error: str) -> None:
    """失败回写：任务置 FAILED，批次失败计数原子自增"""

    updated = db.execute(
        update(EvalTask)
        .where(EvalTask.id == task_id, EvalTask.status == TaskStatus.RUNNING)
        .values(status=TaskStatus.FAILED, error=error[:1000], finished_at=func.now())
    ).rowcount
    if updated:
        _bump_batch(db, batch_id, succeed=False)
    db.commit()


def _bump_batch(db, batch_id: int, succeed: bool) -> None:
    """批次计数自增；全部完成时写入批次完成时间"""

    if succeed:
        counter = {"succeed_tasks": EvalBatch.succeed_tasks + 1}
    else:
        counter = {"failed_tasks": EvalBatch.failed_tasks + 1}
    db.execute(
        update(EvalBatch)
        .where(EvalBatch.id == batch_id)
        .values(
            completed_tasks=EvalBatch.completed_tasks + 1,
            finished_at=case(
                (EvalBatch.completed_tasks + 1 >= EvalBatch.total_tasks, func.now()), else_=EvalBatch.finished_at
            ),
            **counter,
        )
        .execution_options(synchronize_session=False)
    )


@worker_process_init.connect
def _reset_stale_running_tasks(*args, **kwargs) -> None:
    """worker 启动时将遗留 RUNNING 任务归位 PENDING，保证崩溃后可恢复消费"""

    db = SessionLocal()
    try:
        db.execute(update(EvalTask).where(EvalTask.status == TaskStatus.RUNNING).values(status=TaskStatus.PENDING))
        db.commit()
    finally:
        db.close()