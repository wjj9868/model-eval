# @author: ztwz
"""评测批次相关请求/响应模型。"""
from datetime import datetime

from pydantic import BaseModel, Field

from app.enums import BatchStatus, Provider, TaskStatus


class BatchCreate(BaseModel):
    """新建评测批次"""

    name: str = Field(min_length=1, max_length=100, description="批次名称")
    model_ids: list[int] = Field(min_length=1, description="参与评测的模型配置 ID")
    prompt_ids: list[int] = Field(min_length=1, description="参与评测的用例 ID")
    remark: str | None = Field(default=None, max_length=500, description="备注")


class BatchListVO(BaseModel):
    """批次列表项（终态由计数派生）"""

    id: int
    name: str
    remark: str | None
    status: BatchStatus
    model_count: int
    prompt_count: int
    total_tasks: int
    completed_tasks: int
    succeed_tasks: int
    failed_tasks: int
    created_at: datetime
    finished_at: datetime | None

    model_config = {"from_attributes": True}


class ModelBriefVO(BaseModel):
    """模型简要信息"""

    id: int
    name: str
    provider: Provider
    model_name: str


class TaskVO(BaseModel):
    """评测任务结果（单个模型 × 单个用例）"""

    task_id: int
    model_id: int
    model_name: str
    provider: Provider
    status: TaskStatus
    output: str | None
    error: str | None
    response_time_ms: int | None
    attempt_count: int
    started_at: datetime | None
    finished_at: datetime | None


class PromptGroupVO(BaseModel):
    """按用例分组的任务结果（同输入多模型输出对比）"""

    prompt_id: int
    content: str
    category: str
    results: list[TaskVO]


class BatchDetailVO(BatchListVO):
    """批次详情：批次信息 + 参与模型 + 分页的分组结果"""

    models: list[ModelBriefVO]
    prompts: list[PromptGroupVO]
    page: int
    page_size: int
    matched_total: int = Field(description="当前筛选命中的任务组数（按用例去重）")