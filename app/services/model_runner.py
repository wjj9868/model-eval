# @author: ztwz
"""模型调用：按协议构造请求、调用外部服务并标准化返回结果。

支持两种协议：
- Provider.OLLAMA：POST {base_url}/api/chat（stream=False）
- Provider.OPENAI：POST {base_url}/v1/chat/completions（OpenAI 兼容，vLLM / LM Studio 等）
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from app.enums import Provider


class ModelCallError(Exception):
    """模型调用异常（message 为可直接展示的中文原因）"""


@dataclass
class ModelResult:
    """一次模型调用的标准化输出"""

    text: str
    response_time_ms: int


def call_model(
    config, user_input: str, connect_timeout: float = 10.0, timeout_override: int | None = None
) -> ModelResult:
    """调用模型并返回标准化结果；任何失败抛 ModelCallError。

    参数说明：
    - config      : ModelConfig ORM 对象（含协议/根地址/超时等）
    - user_input  : 用户输入
    - connect_timeout : 建连超时（秒）
    - timeout_override : 覆盖配置中的读取超时（测试连通性时用小超时）

    base_url 语义为服务根地址：Ollama 自动拼 /api/chat，OpenAI 兼容自动拼 /v1/chat/completions。
    """

    started = time.monotonic()
    read_timeout = timeout_override or config.timeout_seconds
    base_url = config.base_url.rstrip("/")
    messages = [{"role": "user", "content": user_input}]
    if config.system_prompt:
        messages.insert(0, {"role": "system", "content": config.system_prompt})

    try:
        if config.provider == Provider.OLLAMA:
            response = requests.post(
                f"{base_url}/api/chat",
                json={
                    "model": config.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": float(config.temperature), "num_predict": config.max_tokens},
                },
                timeout=(connect_timeout, read_timeout),
            )
        else:
            headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else None
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                headers=headers,
                json={
                    "model": config.model_name,
                    "messages": messages,
                    "temperature": float(config.temperature),
                    "max_tokens": config.max_tokens,
                },
                timeout=(connect_timeout, read_timeout),
            )
        response.raise_for_status()
        text = _extract_output(config.provider, response.json())
        return ModelResult(text=text, response_time_ms=int((time.monotonic() - started) * 1000))
    except (requests.Timeout, requests.ConnectionError) as exc:
        raise ModelCallError(f"连接或请求超时：{base_url}") from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "未知"
        raise ModelCallError(f"服务返回错误 HTTP {status}") from exc
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise ModelCallError(f"响应解析失败：{exc}") from exc


def _extract_output(provider: str, payload: dict) -> str:
    """按协议从响应 JSON 提取文本（键不存在时抛 KeyError）"""

    if provider == Provider.OLLAMA:
        return payload["message"]["content"]
    return payload["choices"][0]["message"]["content"]