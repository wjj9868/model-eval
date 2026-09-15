# @author: ztwz
"""评测任务表：批次内 「一个模型 × 一个用例」 的最小执行单元。"""
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class EvalTask(Base):
    """评测任务，行级幂等：唯一键防重 + CAS 状态流转防重复执行"""

    __tablename__ = "eval_task"
    __table_args__ = (UniqueConstraint("batch_id", "model_id", "prompt_id", name="uk_batch_model_prompt"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment="主键")
    batch_id: Mapped[int] = mapped_column(BigInteger, index=True, comment="所属批次")
    model_id: Mapped[int] = mapped_column(BigInteger, index=True, comment="模型配置 ID")
    prompt_id: Mapped[int] = mapped_column(BigInteger, index=True, comment="用例 ID")
    status: Mapped[str] = mapped_column(String(20), default="PENDING", comment="任务状态，见 app.enums.TaskStatus")
    input: Mapped[str] = mapped_column(Text, comment="执行时的输入快照（独立于用例后续编辑）")
    output: Mapped[str | None] = mapped_column(Text, nullable=True, comment="模型输出")
    error: Mapped[str | None] = mapped_column(String(1000), nullable=True, comment="失败原因")
    response_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="响应耗时（毫秒）")
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, comment="已执行尝试次数")
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, comment="开始执行时间")
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, comment="结束时间")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), comment="创建时间")