# @author: ztwz
"""数据库引擎与会话管理（SQLAlchemy 2.0 + PyMySQL）。"""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.mysql_url_for(),
    pool_pre_ping=True,
    pool_recycle=3600,
    # 表默认 utf8mb4，避免中文乱码
    connect_args={"init_command": "SET NAMES utf8mb4"},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """ORM 声明式基类"""


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：请求级数据库会话，请求结束关闭"""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()