# @author: ztwz
"""评测任务 Worker 启动入口（Windows 下 Celery 需用 solo 池）。

启动前请先启动 Redis（本机 6379）。worker 启动时会自动恢复遗留 RUNNING 任务。
"""
from app.config import get_settings
from app.tasks.celery_app import celery_app


def main() -> None:
    settings = get_settings()
    print(f"  Worker 启动中（broker: {settings.celery_broker_url}）...")
    worker = celery_app.Worker(pool="solo", concurrency=1, loglevel="info")
    worker.start()


if __name__ == "__main__":
    main()