# @author: ztwz
"""评测批次 API 测试：建批入队（eager）全链路、详情分组、状态筛选、重试失败、统计。"""
from app.models import EvalTask
from app.services.model_runner import ModelResult

BATCH_BASE = "/api/batches"


def _seed_models_prompts(client):
    """通过 API 造 2 个模型 + 3 个用例"""

    model_ids = []
    for i in range(2):
        resp = client.post(
            "/api/models",
            json=dict(
                name=f"模型{i}",
                provider="ollama" if i == 0 else "openai",
                base_url=f"http://127.0.0.1:{9000 + i}/",
                model_name=f"qw{i}",
            ),
        )
        model_ids.append(resp.json()["id"])

    prompt_ids = []
    for i in range(3):
        resp = client.post("/api/prompts", json=dict(content=f"问题{i}", category="通用"))
        prompt_ids.append(resp.json()["id"])
    return model_ids, prompt_ids


def test_create_and_run_batch_full_chain(client, monkeypatch):
    """建批 -> eager 消费 -> 全部成功 -> 详情对比视图完整"""

    model_ids, prompt_ids = _seed_models_prompts(client)
    monkeypatch.setattr("app.tasks.run_eval.call_model", lambda cfg, inp: ModelResult(f"回答:{inp}", 30))

    created = client.post(
        BATCH_BASE,
        json=dict(name="全链路批次", model_ids=model_ids, prompt_ids=prompt_ids),
    ).json()
    assert created["status"] == "COMPLETED"
    assert created["total_tasks"] == 6
    assert created["succeed_tasks"] == 6

    # 详情：按用例分组
    detail = client.get(f"{BATCH_BASE}/{created['id']}").json()
    assert detail["status"] == "COMPLETED"
    assert detail["matched_total"] == 3
    assert len(detail["prompts"]) == 3
    assert len(detail["prompts"][0]["results"]) == 2  # 每个用例 2 个模型
    assert [m["name"] for m in detail["models"]] == ["模型0", "模型1"]
    first = detail["prompts"][0]["results"][0]
    assert first["status"] == "SUCCESS"
    assert first["output"].startswith("回答:")


def test_batch_create_validations(client):
    model_ids, prompt_ids = _seed_models_prompts(client)

    resp = client.post(BATCH_BASE, json=dict(name="空", model_ids=model_ids, prompt_ids=[]))
    assert resp.status_code == 422

    resp = client.post(BATCH_BASE, json=dict(name="", model_ids=model_ids, prompt_ids=prompt_ids))
    assert resp.status_code == 422  # name 空串触发 pydantic min_length

    client.put(f"/api/models/{model_ids[1]}", json=dict(
        name="模型1", provider="openai", base_url="http://127.0.0.1:9001/",
        model_name="qw1", enabled=False,
    ))
    resp = client.post(BATCH_BASE, json=dict(name="禁用模型", model_ids=model_ids, prompt_ids=prompt_ids))
    assert resp.status_code == 400
    assert "已禁用" in resp.json()["detail"]

    resp = client.post(BATCH_BASE, json=dict(name="幽灵模型", model_ids=[999999], prompt_ids=prompt_ids))
    assert resp.status_code == 400


def test_batch_detail_filters_by_status(client, monkeypatch):
    model_ids, prompt_ids = _seed_models_prompts(client)

    def _flaky(cfg, inp):
        # 第二个模型（openai）在第一条问题上失败
        if cfg.provider == "openai" and "问题0" in inp:
            from app.services.model_runner import ModelCallError

            raise ModelCallError("超时重试耗尽")
        return ModelResult("ok", 10)

    monkeypatch.setattr("app.tasks.run_eval.call_model", _flaky)

    created = client.post(BATCH_BASE, json=dict(name="混合批次", model_ids=model_ids, prompt_ids=prompt_ids)).json()
    assert created["status"] == "PARTIAL"
    assert created["failed_tasks"] == 1

    # 只筛选成功的用例分组
    ok_detail = client.get(f"{BATCH_BASE}/{created['id']}", params={"status": "SUCCESS"}).json()
    assert ok_detail["matched_total"] == 3
    assert all(r["status"] == "SUCCESS" for p in ok_detail["prompts"] for r in p["results"])

    # 只筛选失败的用例分组
    bad_detail = client.get(f"{BATCH_BASE}/{created['id']}", params={"status": "FAILED"}).json()
    assert bad_detail["matched_total"] == 1
    assert bad_detail["prompts"][0]["results"][0]["error"]


def test_retry_failed_endpoint(client, monkeypatch):
    model_ids, prompt_ids = _seed_models_prompts(client)

    def _flaky(cfg, inp):
        from app.services.model_runner import ModelCallError

        raise ModelCallError("先失败一次")

    monkeypatch.setattr("app.tasks.run_eval.call_model", _flaky)
    created = client.post(BATCH_BASE, json=dict(name="重试批次", model_ids=[model_ids[0]], prompt_ids=prompt_ids)).json()
    assert created["status"] == "FAILED"
    assert created["failed_tasks"] == 3

    # 修好模型后重试失败项
    monkeypatch.setattr("app.tasks.run_eval.call_model", lambda cfg, inp: ModelResult("恢复输出", 5))
    resp = client.post(f"{BATCH_BASE}/{created['id']}/retry-failed")
    assert resp.json()["retried_count"] == 3

    refreshed = client.get(f"{BATCH_BASE}/{created['id']}").json()
    assert refreshed["status"] == "COMPLETED"
    assert refreshed["succeed_tasks"] == 3
    assert refreshed["failed_tasks"] == 0


def test_batch_list_and_detail_404(client):
    assert client.get(BATCH_BASE).json()["total"] == 0
    assert client.get(f"{BATCH_BASE}/999").status_code == 404
    assert client.post(f"{BATCH_BASE}/999/retry-failed").status_code == 404


def test_batch_list_status_filter(client, monkeypatch):
    model_ids, prompt_ids = _seed_models_prompts(client)
    monkeypatch.setattr("app.tasks.run_eval.call_model", lambda cfg, inp: ModelResult("ok", 5))
    client.post(BATCH_BASE, json=dict(name="完成批次", model_ids=model_ids, prompt_ids=prompt_ids))

    page = client.get(BATCH_BASE, params={"status": "COMPLETED"}).json()
    assert page["total"] == 1
    assert page["items"][0]["name"] == "完成批次"
    assert client.get(BATCH_BASE, params={"status": "RUNNING"}).json()["total"] == 0