# @author: ztwz
"""batch_service 单元/集成测试：建批校验、重试失败项、终态派生、列表筛选。"""
import pytest
from sqlalchemy import func, select

from app.enums import BatchStatus, TaskStatus
from app.exceptions import BusinessError
from app.models import EvalBatch, EvalTask, ModelConfig, PromptCase
from app.services.batch_service import BatchService, derive_batch_status


def _seed(db, model_count=2, prompt_count=2):
    """造数：启用模型 + 用例"""

    models = [
        ModelConfig(name=f"模型{i}", provider="ollama", base_url="http://x:11434/", model_name=f"m{i}")
        for i in range(model_count)
    ]
    prompts = [PromptCase(content_hash=f"h{i}", content=f"问题{i}") for i in range(prompt_count)]
    db.add_all(models + prompts)
    db.commit()
    return models, prompts


def test_create_batch_happy(db):
    models, prompts = _seed(db)
    batch = BatchService(db).create_batch("批次A", [models[0].id, models[1].id], [prompts[0].id, prompts[1].id])

    db.refresh(batch)
    assert batch.total_tasks == 4
    assert batch.model_count == 2
    assert batch.prompt_count == 2
    tasks = db.scalars(select(EvalTask).where(EvalTask.batch_id == batch.id)).all()
    assert len(tasks) == 4
    assert all(t.status == TaskStatus.PENDING for t in tasks)
    # 输入为当前用例内容快照
    assert {t.input for t in tasks} == {"问题0", "问题1"}


def test_create_batch_deduplicates_ids(db):
    models, prompts = _seed(db)
    batch = BatchService(db).create_batch("批次A", [models[0].id, models[0].id], [prompts[0].id])
    assert batch.total_tasks == 1


def test_create_batch_requires_name(db):
    models, prompts = _seed(db)
    with pytest.raises(BusinessError, match="名称"):
        BatchService(db).create_batch("   ", [models[0].id], [prompts[0].id])


def test_create_batch_requires_selection(db):
    models, prompts = _seed(db)
    with pytest.raises(BusinessError, match="选择一个模型"):
        BatchService(db).create_batch("批次", [], [prompts[0].id])
    with pytest.raises(BusinessError, match="选择一个模型"):
        BatchService(db).create_batch("批次", [models[0].id, models[0].id], [])


def test_create_batch_rejects_disabled_model(db):
    models, prompts = _seed(db)
    models[1].enabled = False
    db.commit()
    with pytest.raises(BusinessError, match="不存在或已禁用"):
        BatchService(db).create_batch("批次", [models[0].id, models[1].id], [prompts[0].id])


def test_create_batch_rejects_unknown_ids(db):
    models, prompts = _seed(db)
    with pytest.raises(BusinessError, match="不存在或已禁用"):
        BatchService(db).create_batch("批次", [models[0].id, 99999], [prompts[0].id])
    with pytest.raises(BusinessError, match="用例不存在"):
        BatchService(db).create_batch("批次", [models[0].id], [prompts[0].id, 88888])


def test_retry_failed_resets_and_rollbacks_counts(db):
    models, prompts = _seed(db)
    batch = EvalBatch(name="批次", total_tasks=2, completed_tasks=1, failed_tasks=1)
    db.add(batch)
    db.flush()
    db.add(
        EvalTask(batch_id=batch.id, model_id=models[0].id, prompt_id=prompts[0].id,
                 status=TaskStatus.FAILED, error="boom", input="q")
    )
    db.commit()

    pending_ids = BatchService(db).retry_failed(batch.id)

    assert len(pending_ids) == 1
    db.refresh(batch)
    assert batch.completed_tasks == 0
    assert batch.failed_tasks == 0
    task = db.get(EvalTask, pending_ids[0])
    assert task.status == TaskStatus.PENDING
    assert task.error is None


def test_retry_failed_none(db):
    models, prompts = _seed(db)
    batch = EvalBatch(name="批次", total_tasks=1)
    db.add(batch)
    db.commit()
    assert BatchService(db).retry_failed(batch.id) == []


def test_retry_failed_unknown_batch(db):
    with pytest.raises(BusinessError) as exc:
        BatchService(db).retry_failed(99999)
    assert exc.value.status_code == 404


def test_derive_batch_status_branches():
    cases = [
        (dict(total_tasks=0, completed_tasks=0, failed_tasks=0, succeed_tasks=0), BatchStatus.COMPLETED),
        (dict(total_tasks=4, completed_tasks=0, failed_tasks=0, succeed_tasks=0), BatchStatus.PENDING),
        (dict(total_tasks=4, completed_tasks=1, failed_tasks=0, succeed_tasks=1), BatchStatus.RUNNING),
        (dict(total_tasks=4, completed_tasks=4, failed_tasks=0, succeed_tasks=4), BatchStatus.COMPLETED),
        (dict(total_tasks=4, completed_tasks=4, failed_tasks=2, succeed_tasks=2), BatchStatus.PARTIAL),
        (dict(total_tasks=4, completed_tasks=4, failed_tasks=4, succeed_tasks=0), BatchStatus.FAILED),
    ]
    for kwargs, expected in cases:
        assert derive_batch_status(EvalBatch(**kwargs)) == expected


def test_list_batches_order_and_filter(db):
    models, prompts = _seed(db)
    svc = BatchService(db)
    svc.create_batch("批次1", [models[0].id], [prompts[0].id])
    svc.create_batch("批次2", [models[0].id], [prompts[0].id])

    rows, total = svc.list_batches(1, 10)
    assert total == 2
    assert [b.name for b, _ in rows] == ["批次2", "批次1"]
    assert all(status == BatchStatus.PENDING for _, status in rows)

    rows, total = svc.list_batches(1, 10, BatchStatus.PENDING)
    assert total == 2
    rows, total = svc.list_batches(1, 10, BatchStatus.COMPLETED)
    assert total == 0


def test_list_batches_pagination(db):
    models, prompts = _seed(db)
    svc = BatchService(db)
    for i in range(5):
        svc.create_batch(f"批次{i}", [models[0].id], [prompts[0].id])

    rows, total = svc.list_batches(1, 2)
    assert total == 5
    assert len(rows) == 2
    rows, _ = svc.list_batches(3, 2)
    assert len(rows) == 1