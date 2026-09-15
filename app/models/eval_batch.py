# @author: ztwz
"""评测批次表：一次 「多模型 × 多用例」 对比评测的容器。"""
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class EvalBatch(Base):
    """评测批次，任务计数在任务终态回写时原子自增，终态由计数派生"""

    __tablename__ = "eval_batch"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment="主键")
    name: Mapped[str] = mapped_column(String(100), comment="批次名称")
    remark: Mapped[str | None] = mapped_column(String(500), nullable=True, comment="备注")
    model_count: Mapped[int] = mapped_column(Integer, default=0, comment="模型数量")
    prompt_count: Mapped[int] = mapped_column(Integer, default=0, comment="用例数量")
    total_tasks: Mapped[int] = mapped_column(Integer, default=0, comment="任务总数（模型×用例）")
    completed_tasks: Mapped[int] = mapped_column(Integer, default=0, comment="已完成任务数（成功+失败）")
    succeed_tasks: Mapped[int] = mapped_column(Integer, default=0, comment="成功任务数")
    failed_tasks: Mapped[int] = mapped_column(Integer, default=0, comment="失败任务数")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), comment="创建时间")
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, comment="完成时间")