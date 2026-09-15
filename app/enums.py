# @author: ztwz
"""业务枚举：模型接入协议、任务状态、批次状态（禁止硬编码魔法值）。"""
from enum import StrEnum


class Provider(StrEnum):
    """模型接入协议"""

    OLLAMA = "ollama"  # Ollama 原生接口（/api/chat）
    OPENAI = "openai"  # OpenAI 兼容接口（vLLM / LM Studio / Xinference 等）


class TaskStatus(StrEnum):
    """评测任务状态"""

    PENDING = "PENDING"  # 已落库、待执行
    RUNNING = "RUNNING"  # 执行中
    SUCCESS = "SUCCESS"  # 成功
    FAILED = "FAILED"  # 失败（可重试）


class BatchStatus(StrEnum):
    """评测批次状态（终态由任务计数派生，不在落库时维护）"""

    PENDING = "PENDING"  # 待执行
    RUNNING = "RUNNING"  # 执行中
    PARTIAL = "PARTIAL"  # 全部完成但部分失败
    COMPLETED = "COMPLETED"  # 全部成功
    FAILED = "FAILED"  # 全部失败