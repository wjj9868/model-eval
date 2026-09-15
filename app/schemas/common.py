# @author: ztwz
"""通用分页响应模型。"""
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class PageOut(BaseModel, Generic[T]):
    """统一分页响应结构"""

    items: list[T]
    total: int
    page: int
    page_size: int