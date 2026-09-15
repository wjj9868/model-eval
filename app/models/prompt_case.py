# @author: ztwz
"""测试用例表：单个模型评测输入。"""
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class PromptCase(Base):
    """评测输入用例，content_hash 唯一索引保证内容级去重"""

    __tablename__ = "prompt_case"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment="主键")
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, comment="内容 SHA-256，用于去重")
    content: Mapped[str] = mapped_column(Text, comment="用例内容")
    category: Mapped[str] = mapped_column(String(50), default="默认", comment="分类")
    expected_answer: Mapped[str | None] = mapped_column(Text, nullable=True, comment="期望答案（预留给后续自动评分）")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, comment="是否启用")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), comment="创建时间")
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now(), comment="更新时间")