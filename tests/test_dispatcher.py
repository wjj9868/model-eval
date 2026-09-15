# @author: ztwz
"""任务分发器测试：空列表短路、eager 分支、生产 producer 投递分支。"""
from app.tasks import dispatcher


class _FakeProducerPool:
    """模拟 Celery producer pool"""

    def __init__(self):
        self.producer = object()

    def acquire(self):
        return self

    def __enter__(self):
        return self.producer

    def __exit__(self, *args):
        return False


def test_enqueue_empty_is_noop(monkeypatch):
    called = []

    def _fake(*a, **k):
        called.append(a)

    monkeypatch.setattr(dispatcher.run_eval_task, "apply_async", _fake)
    dispatcher.enqueue_task_ids([])
    assert called == []


def test_enqueue_eager_path_runs_synchronously(monkeypatch):
    called = []

    def _fake(args, **kwargs):
        called.append(args[0])
        return None

    monkeypatch.setattr(dispatcher.settings, "task_always_eager", True)
    monkeypatch.setattr(dispatcher.run_eval_task, "apply_async", _fake)
    dispatcher.enqueue_task_ids([1, 2, 3])
    assert called == [1, 2, 3]


def test_enqueue_producer_path_reuses_single_producer(monkeypatch):
    """生产模式：复用同一 producer 逐条投递，避免大批次重复建连"""

    calls = []
    pool = _FakeProducerPool()

    class _FakeApp:
        @property
        def producer_pool(self):
            return pool

    def _fake(args, producer, **kwargs):
        calls.append((args[0], producer))
        return None

    monkeypatch.setattr(dispatcher, "celery_app", _FakeApp())
    monkeypatch.setattr(dispatcher.settings, "task_always_eager", False)
    monkeypatch.setattr(dispatcher.run_eval_task, "apply_async", _fake)

    dispatcher.enqueue_task_ids([7, 8])

    assert [c[0] for c in calls] == [7, 8]
    assert {id(c[1]) for c in calls} == {id(pool.producer)}  # 全部复用同一 producer