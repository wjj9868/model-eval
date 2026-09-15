# @author: ztwz
"""Pydantic 请求/响应模型（类型安全传参，禁止业务层用 dict）。"""

from app.schemas.batch import (
    BatchCreate,
    BatchDetailVO,
    BatchListVO,
    ModelBriefVO,
    PromptGroupVO,
    TaskVO,
)
from app.schemas.common import PageOut
from app.schemas.model_config import (
    ModelConfigCreate,
    ModelConfigEnabledUpdate,
    ModelConfigOut,
    ModelConfigUpdate,
    TestConnectionResult,
)
from app.schemas.prompt import ImportLine, ImportResult, PromptBatchDelete, PromptCreate, PromptOut, PromptUpdate
from app.schemas.stats import StatsOut

__all__ = [
    "ModelConfigCreate",
    "ModelConfigUpdate",
    "ModelConfigOut",
    "ModelConfigEnabledUpdate",
    "TestConnectionResult",
    "PromptCreate",
    "PromptUpdate",
    "PromptOut",
    "ImportLine",
    "ImportResult",
    "PromptBatchDelete",
    "BatchCreate",
    "BatchListVO",
    "BatchDetailVO",
    "ModelBriefVO",
    "PromptGroupVO",
    "TaskVO",
    "StatsOut",
    "PageOut",
]