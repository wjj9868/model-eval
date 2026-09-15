# @author: ztwz
"""测试用例相关请求/响应模型。"""
from datetime import datetime

from pydantic import BaseModel, Field


class _PromptBase(BaseModel):
    """用例公共字段"""

    content: str = Field(min_length=1, max_length=2000, description="用例内容")
    category: str = Field(default="默认", min_length=1, max_length=50, description="分类")
    expected_answer: str | None = Field(default=None, description="期望答案（预留）")
    enabled: bool = Field(default=True, description="是否启用")


class PromptCreate(_PromptBase):
    """新建用例"""


class PromptUpdate(_PromptBase):
    """修改用例"""


class PromptOut(BaseModel):
    """用例响应"""

    id: int
    content: str
    category: str
    expected_answer: str | None
    enabled: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ImportLine(BaseModel):
    """批量导入的单行数据（用于校验后落库）"""

    content: str
    category: str = "默认"


class PromptBatchDelete(BaseModel):
    """批量删除用例请求"""

    ids: list[int] = Field(min_length=1, description="用例 ID 列表")


class ImportResult(BaseModel):
    """批量导入结果"""

    imported_count: int = Field(description="新增条数")
    skipped_count: int = Field(description="因内容重复跳过的条数")
    invalid_count: int = Field(description="空内容或超长被忽略的条数")