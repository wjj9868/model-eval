# @author: ztwz
"""端到端验证：真实 HTTP 调用 Web + Celery worker 消费（非 eager）。
覆盖：加模型 -> 导入用例 -> 建批 -> 轮询完成 -> 详情对比 -> 失败重试。
"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode() or "null")


def main():
    # 1) 两个模型（OpenAI 兼容 -> 两个本地 mock 服务，base_url 为根地址）
    m1 = call("POST", "/api/models", {
        "name": "Qwen2.5-Mock", "provider": "openai", "base_url": "http://127.0.0.1:9100/",
        "model_name": "mock-qwen", "max_tokens": 256, "temperature": 0.7,
    })
    m2 = call("POST", "/api/models", {
        "name": "Llama3-Mock", "provider": "openai", "base_url": "http://127.0.0.1:9101/",
        "model_name": "mock-llama", "max_tokens": 256, "temperature": 0.7,
    })
    print(f"[1] 模型已创建: {m1['name']} / {m2['name']}")

    # 2) 批量导入用例（粘贴文本）
    boundary = "----e2e"
    fields = [("text", "解释一下什么是数据库索引？\n用一句话介绍 MySQL 事务。\n计算 17 * 23 的结果。\n写一首关于秋天的五言绝句。\n什么是大语言模型的 Token？"),
              ("default_category", "通用")]
    parts = []
    for k, v in fields:
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(BASE + "/api/prompts/import", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req) as resp:
        imp = json.loads(resp.read().decode())
    print(f"[2] 用例导入: 新增 {imp['imported_count']} 条, 重复跳过 {imp['skipped_count']} 条")
    prompts = call("GET", "/api/prompts?page=1&page_size=100")["items"]
    assert imp["imported_count"] == 5 and len(prompts) == 5

    # 3) 建批（2 模型 x 5 用例 = 10 任务），由真实 worker 消费
    batch = call("POST", "/api/batches", {
        "name": "E2E验证批次", "model_ids": [m1["id"], m2["id"]],
        "prompt_ids": [p["id"] for p in prompts], "remark": "mock 服务验证",
    })
    bid = batch["id"]
    print(f"[3] 批次已创建 id={bid}, 状态 {batch['status']}, 总任务 {batch['total_tasks']}")

    # 4) 轮询到终态
    deadline = time.time() + 60
    while time.time() < deadline:
        d = call("GET", f"/api/batches/{bid}")
        if d["status"] in ("COMPLETED", "PARTIAL", "FAILED"):
            break
        time.sleep(0.8)
    print(f"[4] 批次终态: {d['status']}  成功 {d['succeed_tasks']} / 失败 {d['failed_tasks']} / 共 {d['total_tasks']}")
    assert d["status"] == "COMPLETED" and d["succeed_tasks"] == 10, "E2E 全成功断言失败"

    # 5) 详情：同输入多模型输出对比
    detail = call("GET", f"/api/batches/{bid}?page=1&page_size=10")
    assert detail["matched_total"] == 5 and len(detail["prompts"]) == 5
    first = detail["prompts"][0]
    assert len(first["results"]) == 2
    for r in first["results"]:
        assert r["status"] == "SUCCESS" and r["output"] and r["response_time_ms"] > 0
        print(f"    用例 #{first['prompt_id']} | {r['model_name']}: {r['output'][:40]}... ({r['response_time_ms']}ms)")
    assert first["results"][0]["output"] != first["results"][1]["output"], "两个模型输出应当不同"

    # 6) 失败重试：对一个不可用的模型建批 -> FAILED -> 修好 -> retry
    m3 = call("POST", "/api/models", {
        "name": "Down-Model", "provider": "openai", "base_url": "http://127.0.0.1:9999/v1/",
        "model_name": "x", "max_tokens": 64,
    })
    bad = call("POST", "/api/batches", {"name": "故障批次", "model_ids": [m3["id"]], "prompt_ids": [prompts[0]["id"]]})
    deadline = time.time() + 30
    while time.time() < deadline:
        d2 = call("GET", f"/api/batches/{bad['id']}")
        if d2["status"] in ("COMPLETED", "PARTIAL", "FAILED"):
            break
        time.sleep(0.8)
    if d2["status"] == "RUNNING" and d2["completed_tasks"] == d2["total_tasks"]:
        pass
    print(f"[6] 故障批次: {d2['status']} 失败 {d2['failed_tasks']}")
    assert d2["failed_tasks"] == 1

    # 修好模型（改地址指向 mock）后重试
    call("PUT", f"/api/models/{m3['id']}", {
        "name": "Down-Model", "provider": "openai", "base_url": "http://127.0.0.1:9100/",
        "model_name": "mock-qwen", "max_tokens": 64, "temperature": 0.7, "enabled": True,
    })
    retried = call("POST", f"/api/batches/{bad['id']}/retry-failed")
    print(f"[7] 重试投递: {retried['retried_count']} 个失败任务")
    deadline = time.time() + 30
    while time.time() < deadline:
        d3 = call("GET", f"/api/batches/{bad['id']}")
        if d3["status"] in ("COMPLETED", "PARTIAL", "FAILED"):
            break
        time.sleep(0.8)
    print(f"[8] 重试后: {d3['status']} 成功 {d3['succeed_tasks']} / 失败 {d3['failed_tasks']}")
    assert d3["status"] == "COMPLETED" and d3["failed_tasks"] == 0

    # 9) 统计
    stats = call("GET", "/api/stats")
    print(f"[9] 工作台统计: 模型 {stats['model_count']}, 用例 {stats['prompt_count']}, 批次 {stats['batch_count']}, 成功率 {stats['success_rate']}")
    print("\n=== E2E 全部通过 ===")


if __name__ == "__main__":
    main()
