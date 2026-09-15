# @author: ztwz
"""工作台统计 API 测试。"""
from app.services.model_runner import ModelResult


def test_stats_empty(client):
    stats = client.get("/api/stats").json()
    assert stats["model_count"] == 0
    assert stats["prompt_count"] == 0
    assert stats["batch_count"] == 0
    assert stats["success_rate"] is None


def test_stats_with_data(client, monkeypatch):
    model = client.post("/api/models", json=dict(
        name="m1", provider="ollama", base_url="http://x:1/", model_name="q",
    )).json()
    prompt = client.post("/api/prompts", json=dict(content="q1")).json()
    monkeypatch.setattr("app.tasks.run_eval.call_model", lambda cfg, inp: ModelResult("ok", 5))
    client.post("/api/batches", json=dict(name="b1", model_ids=[model["id"]], prompt_ids=[prompt["id"]]))

    stats = client.get("/api/stats").json()
    assert stats["model_count"] == 1
    assert stats["prompt_count"] == 1
    assert stats["batch_count"] == 1
    assert stats["total_tasks"] == 1
    assert stats["succeed_tasks"] == 1
    assert stats["failed_tasks"] == 0
    assert stats["success_rate"] == 1.0
    assert len(stats["recent_batches"]) == 1
    assert stats["recent_batches"][0]["status"] == "COMPLETED"


def test_stats_ignores_disabled_model(client, db):
    from app.models import ModelConfig

    client.post("/api/models", json=dict(
        name="m1", provider="ollama", base_url="http://x:1/", model_name="q",
    ))
    client.post("/api/models", json=dict(
        name="m2", provider="openai", base_url="http://x:2/", model_name="q2",
    ))
    db.query(ModelConfig).filter(ModelConfig.name == "m2").update({ModelConfig.enabled: False})
    db.commit()
    assert client.get("/api/stats").json()["model_count"] == 1