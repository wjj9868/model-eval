# @author: ztwz
"""任务队列分发：将评测任务 ID 投递到 Celery 队列。

大批次复用单个 producer 连接循环投递，避免逐条建连。
"""
from app.config import get_settings
from app.tasks.celery_app import celery_app
from app.tasks.run_eval import run_eval_task

settings = get_settings()


def enqueue_task_ids(task_ids: list[int]) -> None:
    """按序投递任务（eager 模式直接同步执行，用于测试）"""

    if not task_ids:
        return
    if settings.task_always_eager:
        for task_id in task_ids:
            run_eval_task.apply_async(args=[task_id])
        return
    with celery_app.producer_pool.acquire() as producer:
        for task_id in task_ids:
            run_eval_task.apply_async(args=[task_id], producer=producer)