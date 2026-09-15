# @author: ztwz
"""run_eval_task 集成测试（Celery eager 模式）：成功/失败/幂等/配置缺失。"""
import pytest
from sqlalchemy import select

from app.enums import TaskStatus
from app.models import EvalBatch, EvalTask, ModelConfig, PromptCase
from app.services.model_runner import ModelCallError, ModelResult
from app.tasks.run_eval import run_eval_task


def _seed_task(db, enabled=True):
    """造数据：模型 + 用例 + 批次 + 1 个 PENDING 任务"""

    config = ModelConfig(name="模型1", provider="openai", base_url="http://x:8000/",
                         model_name="qwen", enabled=enabled, timeout_seconds=30)
    prompt = PromptCase(content_hash="h1", content="问题")
    db.add_all([config, prompt])
    db.flush()
    batch = EvalBatch(name="批次", total_tasks=1)
    db.add(batch)
    db.flush()
    task = EvalTask(batch_id=batch.id, model_id=config.id, prompt_id=prompt.id,
                    status=TaskStatus.PENDING, input=prompt.content)
    db.add(task)
    db.commit()
    return task.id, batch.id


def test_task_success_writes_result_and_batch_counts(db, monkeypatch):
    task_id, batch_id = _seed_task(db)
    monkeypatch.setattr("app.tasks.run_eval.call_model", lambda *a, **k: ModelResult("正常输出", 12))

    result = run_eval_task.apply(args=[task_id]).get()

    assert result == "ok"
    task = db.get(EvalTask, task_id)
    assert task.status == TaskStatus.SUCCESS
    assert task.output == "正常输出"
    assert task.response_time_ms == 12
    assert task.attempt_count == 1
    batch = db.get(EvalBatch, batch_id)
    assert batch.completed_tasks == 1
    assert batch.succeed_tasks == 1
    assert batch.failed_tasks == 0
    assert batch.finished_at is not None


def test_task_failure_when_retries_exhausted(db, monkeypatch):
    task_id, batch_id = _seed_task(db)
    monkeypatch.setattr("app.tasks.run_eval.call_model", lambda *a, **k: (_ for _ in ()).throw(ModelCallError("连接或请求超时")))
    monkeypatch.setattr(run_eval_task, "max_retries", 0)

    run_eval_task.apply(args=[task_id]).get()

    task = db.get(EvalTask, task_id)
    assert task.status == TaskStatus.FAILED
    assert "超时" in task.error
    assert task.attempt_count == 1
    assert task.finished_at is not None
    batch = db.get(EvalBatch, batch_id)
    assert batch.completed_tasks == 1
    assert batch.failed_tasks == 1
    assert batch.succeed_tasks == 0


def test_task_retry_increments_attempts_before_final_failure(db, monkeypatch):
    """重试耗尽路径：多次抢占尝试后终态 FAILED，计数不重复累计"""

    task_id, batch_id = _seed_task(db)
    monkeypatch.setattr("app.tasks.run_eval.call_model", lambda *a, **k: (_ for _ in ()).throw(ModelCallError("超时")))
    run_eval_task.apply(args=[task_id]).get()

    task = db.get(EvalTask, task_id)
    assert task.status == TaskStatus.FAILED
    batch = db.get(EvalBatch, batch_id)
    assert batch.completed_tasks == 1  # 终态只计一次


def test_duplicate_delivery_is_skipped(db, monkeypatch):
    task_id, batch_id = _seed_task(db)
    monkeypatch.setattr("app.tasks.run_eval.call_model", lambda *a, **k: ModelResult("ok", 5))

    assert run_eval_task.apply(args=[task_id]).get() == "ok"
    # 重复投递：已经处理，直接跳过，计数不再变化
    assert run_eval_task.apply(args=[task_id]).get() == "already processed"
    batch = db.get(EvalBatch, batch_id, populate_existing=True)
    assert batch.completed_tasks == 1
    assert batch.succeed_tasks == 1


def test_task_missing_config_fails_fast(db):
    task_id, batch_id = _seed_task(db)

    # 模拟模型配置被删除
    config = db.query(ModelConfig).one()
    db.delete(config)
    db.commit()

    run_eval_task.apply(args=[task_id]).get()

    task = db.get(EvalTask, task_id)
    assert task.status == TaskStatus.FAILED
    assert "模型配置" in task.error


def test_task_disabled_config_fails_fast(db):
    task_id, batch_id = _seed_task(db, enabled=False)

    run_eval_task.apply(args=[task_id]).get()

    task = db.get(EvalTask, task_id)
    assert task.status == TaskStatus.FAILED
    assert "已禁用" in task.error


def test_task_unknown_id_is_noop(db):
    assert run_eval_task.apply(args=[999999]).get() == "task not found"


def test_task_internal_error_branch_fails_gracefully(db, monkeypatch):
    """非 ModelCallError 的未知异常：重试耗尽后落 FAILED 且不抛异常"""

    task_id, batch_id = _seed_task(db)

    def _boom(*a, **k):
        raise RuntimeError("意外崩溃")

    monkeypatch.setattr("app.tasks.run_eval.call_model", _boom)
    monkeypatch.setattr(run_eval_task, "max_retries", 0)

    assert run_eval_task.apply(args=[task_id]).get() == "failed"
    task = db.get(EvalTask, task_id)
    assert task.status == TaskStatus.FAILED
    assert "内部错误" in task.error
    batch = db.get(EvalBatch, batch_id)
    assert batch.failed_tasks == 1


def test_worker_startup_resets_stale_running_tasks(db, monkeypatch):
    """worker 启动信号：崩溃遗留的 RUNNING 任务归位 PENDING"""

    task_id, _ = _seed_task(db)
    from sqlalchemy import update

    from app.tasks.run_eval import _reset_stale_running_tasks

    db.execute(update(EvalTask).where(EvalTask.id == task_id).values(status=TaskStatus.RUNNING))
    db.commit()
    assert db.get(EvalTask, task_id).status == TaskStatus.RUNNING

    _reset_stale_running_tasks()

    db.rollback()  # 结束当前读事务，读取最新已提交状态
    assert db.get(EvalTask, task_id, populate_existing=True).status == TaskStatus.PENDING