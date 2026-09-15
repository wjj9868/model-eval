# @author: ztwz
"""测试基础设施：测试库 model_eval_test、Celery eager 模式、表清空、公共夹具。

环境变量必须在导入 app 模块之前设置，保证 Settings 读取到测试配置。
"""
import os

os.environ["MODEL_EVAL_MYSQL_DB"] = "model_eval_test"
os.environ["MODEL_EVAL_TASK_ALWAYS_EAGER"] = "true"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.database import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402

settings = get_settings()


@pytest.fixture(scope="session", autouse=True)
def _prepare_database():
    """会话级：确保测试库存在"""

    admin_engine = create_engine(settings.mysql_url_for("mysql"), poolclass=NullPool, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        conn.execute(
            text(
                "CREATE DATABASE IF NOT EXISTS model_eval_test "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        )
    admin_engine.dispose()
    yield


@pytest.fixture(autouse=True)
def _clean_tables(_prepare_database):
    """用例级：建表并清空全部数据，保证用例隔离"""

    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
    yield


@pytest.fixture()
def db():
    """数据库会话"""

    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture()
def client():
    """FastAPI 测试客户端（运行 lifespan 自动建表）"""

    with TestClient(app) as c:
        yield c