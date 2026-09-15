# @author: ztwz
"""Web 服务启动入口：python run.py [--port 8000]"""
import os

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    port = int(os.environ.get("MODEL_EVAL_PORT", "8000"))
    print(f"  Web 服务启动中： http://127.0.0.1:{port}   （数据库：{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_db}）")
    uvicorn.run("app.main:app", host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()