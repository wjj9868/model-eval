# @author: ztwz
"""model_runner 单元测试：双协议 payload、异常映射、超时覆盖。"""
import pytest
import requests

import app.services.model_runner as model_runner
from app.enums import Provider
from app.models import ModelConfig
from app.services.model_runner import ModelCallError, call_model


def _config(provider=Provider.OLLAMA, base_url="http://127.0.0.1:11434/", **kwargs):
    """构造内存中的模型配置（不落库）"""

    return ModelConfig(
        id=1,
        name="test",
        provider=provider,
        base_url=base_url,
        model_name="qwen2.5:7b",
        system_prompt="你是助手",
        max_tokens=128,
        temperature=0.5,
        **kwargs,
    )


class _FakeResponse:
    """模拟 requests.Response"""

    def __init__(self, payload=None, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def _install_fake_post(monkeypatch, fake):
    calls = {}

    def _post(url, **kwargs):
        calls["url"] = url
        calls["kwargs"] = kwargs
        return fake

    monkeypatch.setattr(model_runner.requests, "post", _post)
    return calls


def test_ollama_success(monkeypatch):
    calls = _install_fake_post(monkeypatch, _FakeResponse({"message": {"content": "qwen 回复"}}))
    result = call_model(_config(), "你好")

    assert result.text == "qwen 回复"
    assert result.response_time_ms >= 0
    assert calls["url"] == "http://127.0.0.1:11434/api/chat"
    body = calls["kwargs"]["json"]
    assert body["model"] == "qwen2.5:7b"
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.5, "num_predict": 128}
    assert body["messages"][0] == {"role": "system", "content": "你是助手"}
    assert body["messages"][1] == {"role": "user", "content": "你好"}


def test_openai_success(monkeypatch):
    calls = _install_fake_post(
        monkeypatch, _FakeResponse({"choices": [{"message": {"content": "gpt 风格回复"}}]})
    )
    result = call_model(_config(provider=Provider.OPENAI, api_key="sk-abc"), "你好")

    assert result.text == "gpt 风格回复"
    assert calls["url"] == "http://127.0.0.1:11434/v1/chat/completions"
    body = calls["kwargs"]["json"]
    assert body["max_tokens"] == 128
    assert body["temperature"] == 0.5
    assert calls["kwargs"]["headers"] == {"Authorization": "Bearer sk-abc"}


def test_openai_without_api_key_omits_header(monkeypatch):
    calls = _install_fake_post(
        monkeypatch, _FakeResponse({"choices": [{"message": {"content": "ok"}}]})
    )
    call_model(_config(provider=Provider.OPENAI), "你好")
    assert calls["kwargs"]["headers"] is None


def test_timeout_override_applies(monkeypatch):
    calls = _install_fake_post(monkeypatch, _FakeResponse({"message": {"content": "ok"}}))
    call_model(_config(timeout_seconds=300), "你好", timeout_override=15)
    assert calls["kwargs"]["timeout"] == (10.0, 15)


def test_http_error_maps_to_model_call_error(monkeypatch):
    _install_fake_post(monkeypatch, _FakeResponse(status_code=503))
    with pytest.raises(ModelCallError, match="HTTP 503"):
        call_model(_config(), "你好")


def test_timeout_maps_to_model_call_error(monkeypatch):
    def _raise(*a, **k):
        raise requests.Timeout()

    monkeypatch.setattr(model_runner.requests, "post", _raise)
    with pytest.raises(ModelCallError, match="超时"):
        call_model(_config(), "你好")


def test_connection_error_maps_to_model_call_error(monkeypatch):
    def _raise(*a, **k):
        raise requests.ConnectionError()

    monkeypatch.setattr(model_runner.requests, "post", _raise)
    with pytest.raises(ModelCallError, match="超时"):
        call_model(_config(), "你好")


def test_invalid_json_maps_to_model_call_error(monkeypatch):
    _install_fake_post(monkeypatch, _FakeResponse(payload=ValueError("bad json")))
    with pytest.raises(ModelCallError, match="解析失败"):
        call_model(_config(), "你好")


def test_missing_field_maps_to_model_call_error(monkeypatch):
    _install_fake_post(monkeypatch, _FakeResponse({"choices": []}))
    with pytest.raises(ModelCallError, match="解析失败"):
        call_model(_config(provider=Provider.OPENAI), "你好")