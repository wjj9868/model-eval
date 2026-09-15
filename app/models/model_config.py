# @author: ztwz
"""模型配置表：声明如何调用某个已部署的开源模型。"""
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ModelConfig(Base):
    """模型接入配置（provider 存储字符串，语义见 app.enums.Provider）"""

    __tablename__ = "model_config"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment="主键")
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True, comment="配置名称（唯一）")
    provider: Mapped[str] = mapped_column(String(20), comment="接入协议：ollama / openai")
    base_url: Mapped[str] = mapped_column(String(255), comment="服务地址，如 http://127.0.0.1:11434")
    api_key: Mapped[str | None] = mapped_column(String(255), nullable=True, comment="OpenAI 兼容接口的 API Key（可选）")
    model_name: Mapped[str] = mapped_column(String(100), comment="模型名，如 qwen2.5:7b 或 /models/Qwen2.5-7B-Instruct")
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True, comment="可选系统提示词")
    max_tokens: Mapped[int] = mapped_column(Integer, default=1024, comment="最大生成 token 数")
    temperature: Mapped[float] = mapped_column(Numeric(3, 2), default=0.7, comment="采样温度")
    timeout_seconds: Mapped[int] = mapped_column(Integer, default=300, comment="单次调用超时（秒）")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, comment="是否启用")
    remark: Mapped[str | None] = mapped_column(String(500), nullable=True, comment="备注")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), comment="创建时间")
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now(), comment="更新时间")