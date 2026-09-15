# @author: ztwz
"""测试用例 API 测试：CRUD、查重、批量导入（文本/csv/jsonl）、批量删除、删除保护。"""
from io import BytesIO

from app.models import EvalBatch, EvalTask, ModelConfig, PromptCase

BASE = "/api/prompts"


def _payload(**kwargs):
    data = dict(content="什么是数据库索引？", category="面试")
    data.update(kwargs)
    return data


def test_prompt_crud_roundtrip(client):
    created = client.post(BASE, json=_payload()).json()
    assert created["id"] > 0
    assert created["category"] == "面试"

    page = client.get(BASE, params={"page": 1, "page_size": 10}).json()
    assert page["total"] == 1
    assert [p["id"] for p in page["items"]] == [created["id"]]

    assert client.get(BASE, params={"keyword": "索引"}).json()["total"] == 1
    assert client.get(BASE, params={"category": "面试"}).json()["total"] == 1
    assert client.get(BASE, params={"category": "其他"}).json()["total"] == 0
    assert client.get(BASE, params={"keyword": "没有的词"}).json()["total"] == 0

    updated = client.put(f"{BASE}/{created['id']}", json=_payload(content="改过的内容", category="通用")).json()
    assert updated["content"] == "改过的内容"

    assert client.delete(f"{BASE}/{created['id']}").status_code == 200
    assert client.get(BASE).json()["total"] == 0


def test_create_prompt_duplicate_content_rejected(client):
    client.post(BASE, json=_payload())
    resp = client.post(BASE, json=_payload())
    assert resp.status_code == 400


def test_prompt_delete_protected_when_referenced(client, db):
    prompt = PromptCase(content_hash="h", content="q")
    config = ModelConfig(name="m", provider="ollama", base_url="http://x:1/", model_name="x")
    db.add_all([prompt, config])
    db.flush()
    batch = EvalBatch(name="b", total_tasks=1)
    db.add(batch)
    db.flush()
    db.add(EvalTask(batch_id=batch.id, model_id=config.id, prompt_id=prompt.id, input="q"))
    db.commit()

    resp = client.delete(f"{BASE}/{prompt.id}")
    assert resp.status_code == 400
    assert "评测任务引用" in resp.json()["detail"]


def test_import_text_lines(client):
    body = {"text": "第一问\n第二问\n第一问\n\n", "default_category": "批量"}
    result = client.post(f"{BASE}/import", data=body).json()

    assert result["imported_count"] == 2  # 空行忽略，重复跳过 1 条
    assert result["skipped_count"] == 1
    assert result["invalid_count"] == 0

    # 再次导入完全重复（3 行全部命中已有内容）
    result = client.post(f"{BASE}/import", data=body).json()
    assert result["imported_count"] == 0
    assert result["skipped_count"] == 3


def test_import_csv_file(client):
    # 中间空行触发空行跳过分支
    csv_bytes = "content,category\n第一问,CSV类\n\n第二问,CSV类\n".encode("utf-8")
    result = client.post(
        f"{BASE}/import",
        files={"file": ("cases.csv", BytesIO(csv_bytes), "text/csv")},
    ).json()
    assert result["imported_count"] == 2
    page = client.get(BASE).json()
    assert {p["category"] for p in page["items"]} == {"CSV类"}


def test_import_jsonl_file(client):
    lines = [
        '{"content": "json 一问", "category": "JSON类"}',
        "",
        '{"content": ""}',
        '{"content": "json 二问"}',
    ]
    result = client.post(
        f"{BASE}/import",
        files={"file": ("cases.jsonl", BytesIO("\n".join(lines).encode("utf-8")), "application/json")},
        data={"default_category": "默认分类"},
    ).json()
    assert result["imported_count"] == 2  # 空行忽略、空内容计无效
    assert result["invalid_count"] == 1
    page = client.get(BASE).json()
    assert {p["category"] for p in page["items"]} == {"JSON类", "默认分类"}


def test_import_jsonl_invalid_line_reports_error(client):
    resp = client.post(
        f"{BASE}/import",
        files={"file": ("bad.jsonl", BytesIO(b'{"content": "ok"}\nnot-json'), "application/json")},
    )
    assert resp.status_code == 400
    assert "JSON 第 2 行" in resp.json()["detail"]


def test_import_requires_source(client):
    resp = client.post(f"{BASE}/import", data={})
    assert resp.status_code == 400


def test_batch_delete(client, db):
    p1 = PromptCase(content_hash="h1", content="c1")
    p2 = PromptCase(content_hash="h2", content="c2")
    db.add_all([p1, p2])
    db.commit()

    resp = client.post(f"{BASE}/batch-delete", json={"ids": [p1.id, p2.id]})
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 2
    assert db.query(PromptCase).count() == 0


def test_batch_delete_protected_when_referenced(client, db):
    p1 = PromptCase(content_hash="h1", content="c1")
    p2 = PromptCase(content_hash="h2", content="c2")
    config = ModelConfig(name="m", provider="ollama", base_url="http://x:1/", model_name="x")
    db.add_all([p1, p2, config])
    db.flush()
    batch = EvalBatch(name="b", total_tasks=1)
    db.add(batch)
    db.flush()
    db.add(EvalTask(batch_id=batch.id, model_id=config.id, prompt_id=p1.id, input="c1"))
    db.commit()

    resp = client.post(f"{BASE}/batch-delete", json={"ids": [p1.id, p2.id]})
    assert resp.status_code == 400
    assert "引用" in resp.json()["detail"]
    assert db.query(PromptCase).count() == 2  # 整批拒绝，未删除任何数据


def test_categories_endpoint(client, db):
    db.add_all(
        [
            PromptCase(content_hash="h1", content="a", category="X"),
            PromptCase(content_hash="h2", content="b", category="X"),
            PromptCase(content_hash="h3", content="c", category="Y"),
        ]
    )
    db.commit()
    assert sorted(client.get(f"{BASE}/categories").json()) == ["X", "Y"]


def test_prompt_update_and_delete_unknown_404(client):
    assert client.put(f"{BASE}/999", json=_payload()).status_code == 404
    assert client.delete(f"{BASE}/999").status_code == 404