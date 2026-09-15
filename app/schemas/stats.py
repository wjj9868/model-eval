# @author: ztwz
"""工作台统计响应模型。"""
from pydantic import BaseModel, Field

from app.schemas.batch import BatchListVO


class StatsOut(BaseModel):
    """工作台统计"""

    model_count: int = Field(description="启用的模型配置数")
    prompt_count: int = Field(description="启用的用例数")
    batch_count: int = Field(description="评测批次总数")
    total_tasks: int = Field(description="累计任务数")
    succeed_tasks: int = Field(description="累计成功任务数")
    failed_tasks: int = Field(description="累计失败任务数")
    success_rate: float | None = Field(default=None, description="累计成功率（无任务时为 null）")
    recent_batches: list[BatchListVO] = Field(description="最近批次（按创建时间倒序）")