# @author: ztwz
"""模拟 OpenAI 兼容模型服务（本地 E2E 验证用，无需真实 GPU/模型）。

用法：python scripts/mock_llm_server.py [port=9100]
返回内容形如 "【mock:模型名】收到：用户输入"，便于肉眼比对同一输入的多模型输出。
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    model = "mock-model"

    def log_message(self, *args):
        pass  # 精简日志

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        user_input = next((m.get("content") for m in body.get("messages", []) if m.get("role") == "user"), "")
        model = body.get("model", self.model)
        time.sleep(0.15)  # 模拟生成耗时
        reply = f"【mock:{model}】收到：{user_input}"
        self._json(200, {"choices": [{"message": {"role": "assistant", "content": reply}}]})


def make_server(port: int, model: str) -> ThreadingHTTPServer:
    Handler.model = model
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--model", default="mock-model", help="模拟的模型名")
    args = parser.parse_args()
    server = make_server(args.port, args.model)
    print(f"  模拟 OpenAI 模型服务已启动： http://127.0.0.1:{args.port}/v1 （模型名 {args.model}）")
    server.serve_forever()


if __name__ == "__main__":
    main()