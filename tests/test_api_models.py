# @author: ztwz
"""模型配置 API 测试：CRUD、去重、启停、连通性测试、删除保护。"""
from app.models import EvalBatch, EvalTask, ModelConfig, PromptCase
from app.services.model_runner import ModelCallError, ModelResult

BASE = "/api/models"


def _payload(**kwargs):
    data = dict(
        name="qwen-ollama",
        provider="ollama",
        base_url="http://127.0.0.1:11434/",
        model_name="qwen2.5:7b",
        max_tokens=512,
        temperature=0.7,
        timeout_seconds=60,
    )
    data.update(kwargs)
    return data


def test_model_crud_roundtrip(client):
    created = client.post(BASE, json=_payload()).json()
    assert created["id"] > 0
    assert created["provider"] == "ollama"
    assert created["enabled"] is True

    # 列表
    page = client.get(BASE, params={"page": 1, "page_size": 10}).json()
    assert page["total"] == 1
    assert page["items"][0]["name"] == "qwen-ollama"

    # 关键字检索
    assert client.get(BASE, params={"keyword": "qwen"}).json()["total"] == 1
    assert client.get(BASE, params={"keyword": "不存在"}).json()["total"] == 0

    # 修改
    updated = client.put(f"{BASE}/{created['id']}", json=_payload(name="qwen-new", temperature=0.3)).json()
    assert updated["name"] == "qwen-new"
    assert updated["temperature"] == 0.3

    # 启停
    assert client.patch(f"{BASE}/{created['id']}/enabled", json={"enabled": False}).json()["enabled"] is False

    # 删除
    resp = client.delete(f"{BASE}/{created['id']}")
    assert resp.status_code == 200
    assert client.get(BASE).json()["total"] == 0


def test_create_model_name_duplicate_rejected(client):
    client.post(BASE, json=_payload())
    resp = client.post(BASE, json=_payload(name="qwen-ollama", base_url="http://other:1/", model_name="x"))
    assert resp.status_code == 400
    assert "已存在" in resp.json()["detail"]


def test_create_model_invalid_base_url(client):
    resp = client.post(BASE, json=_payload(base_url="ftp://bad"))
    assert resp.status_code == 422


def test_update_unknown_model_404(client):
    resp = client.put(f"{BASE}/999", json=_payload())
    assert resp.status_code == 404
    assert client.patch(f"{BASE}/999/enabled", json={"enabled": True}).status_code == 404


def test_test_connection_success(client, monkeypatch):
    created = client.post(BASE, json=_payload()).json()
    monkeypatch.setattr(
        "app.api.models_router.call_model",
        lambda *a, **k: ModelResult("连通成功", 42),
    )
    result = client.post(f"{BASE}/{created['id']}/test").json()
    assert result["success"] is True
    assert result["response_time_ms"] == 42
    assert result["sample"] == "连通成功"


def test_test_connection_failure(client, monkeypatch):
    created = client.post(BASE, json=_payload()).json()

    def _raise(*a, **k):
        raise ModelCallError("连接或请求超时")

    monkeypatch.setattr("app.api.models_router.call_model", _raise)
    result = client.post(f"{BASE}/{created['id']}/test").json()
    assert result["success"] is False
    assert "超时" in result["message"]


def test_test_connection_unknown_error_is_structured(client, monkeypatch):
    created = client.post(BASE, json=_payload()).json()

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.api.models_router.call_model", _boom)
    result = client.post(f"{BASE}/{created['id']}/test").json()
    assert result["success"] is False
    assert "未知错误" in result["message"]


def test_delete_referenced_model_protected(client, db):
    config = ModelConfig(name="m", provider="ollama", base_url="http://x:1/", model_name="x")
    prompt = PromptCase(content_hash="h", content="q")
    db.add_all([config, prompt])
    db.flush()
    batch = EvalBatch(name="b", total_tasks=1)
    db.add(batch)
    db.flush()
    db.add(EvalTask(batch_id=batch.id, model_id=config.id, prompt_id=prompt.id, input="q"))
    db.commit()

    resp = client.delete(f"{BASE}/{config.id}")
    assert resp.status_code == 400
    assert "评测任务引用" in resp.json()["detail"]

    # 未引用模型可删除
    other = client.post(BASE, json=_payload(name="临时模型", base_url="http://y:1/", model_name="y")).json()
    assert client.delete(f"{BASE}/{other['id']}").status_code == 200