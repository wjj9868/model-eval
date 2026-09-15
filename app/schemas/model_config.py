# @author: ztwz
"""模型配置相关请求/响应模型。"""
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.enums import Provider


class _ModelConfigBase(BaseModel):
    """模型配置公共字段"""

    name: str = Field(min_length=1, max_length=100, description="配置名称（唯一）")
    provider: Provider = Field(description="接入协议")
    base_url: str = Field(max_length=255, description="服务地址，如 http://127.0.0.1:11434")
    api_key: str | None = Field(default=None, max_length=255, description="OpenAI 兼容接口的 API Key")
    model_name: str = Field(min_length=1, max_length=100, description="模型名")
    system_prompt: str | None = Field(default=None, description="系统提示词")
    max_tokens: int = Field(default=1024, ge=1, le=8192, description="最大生成 token 数")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="采样温度")
    timeout_seconds: int = Field(default=300, ge=1, le=900, description="单次调用超时（秒）")
    remark: str | None = Field(default=None, max_length=500, description="备注")

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        """服务地址必须是 http(s) 服务根地址（不以 / 结尾的自动补全）"""

        if not (value.startswith("http://") or value.startswith("https://")):
            raise ValueError("base_url 必须以 http:// 或 https:// 开头")
        return value.rstrip("/") + "/"


class ModelConfigCreate(_ModelConfigBase):
    """新建模型配置"""


class ModelConfigUpdate(_ModelConfigBase):
    """修改模型配置"""

    enabled: bool = Field(default=True, description="是否启用")


class ModelConfigOut(_ModelConfigBase):
    """模型配置响应"""

    id: int
    enabled: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ModelConfigEnabledUpdate(BaseModel):
    """启停状态更新"""

    enabled: bool = Field(description="是否启用")


class TestConnectionResult(BaseModel):
    """连通性测试结果"""

    success: bool = Field(description="是否连通")
    message: str = Field(description="展示信息，失败时为原因")
    response_time_ms: int | None = Field(default=None, description="响应耗时（毫秒）")
    sample: str | None = Field(default=None, description="返回文本预览")