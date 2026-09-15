# @author: ztwz
"""Celery 应用：broker/backend 均为 Redis。"""
from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "model_eval",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.run_eval"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,  # 任务完成后再确认消息，worker 崩溃可重投
    task_reject_on_worker_lost=True,  # worker 丢失时拒绝并重试
    task_track_started=False,
    task_time_limit=settings.default_timeout_seconds * 3 + 60,  # 硬限制兜底
    task_soft_time_limit=settings.default_timeout_seconds * 3 + 30,
    result_expires=3600,
    task_always_eager=settings.task_always_eager,  # 测试环境同步执行
)