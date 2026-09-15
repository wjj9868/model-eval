# @author: ztwz
"""FastAPI 应用入口：注册路由、异常处理、启动建表、托管前端静态资源。"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.batches_router import router as batches_router
from app.api.models_router import router as models_router
from app.api.prompts_router import router as prompts_router
from app.api.stats_router import router as stats_router
from app.config import get_settings
from app.database import Base, engine
from app.exceptions import BusinessError

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时自动建表（幂等）"""

    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="开源模型对比评测平台", version="1.0.0", lifespan=lifespan)


@app.exception_handler(BusinessError)
async def business_error_handler(_: Request, exc: BusinessError) -> JSONResponse:
    """业务异常统一返回（message 直接展示给前端）"""

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


app.include_router(models_router)
app.include_router(prompts_router)
app.include_router(batches_router)
app.include_router(stats_router)

# 前端静态资源（html=True：根路径直接返回 index.html，置于最后兜底其余路径）
app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="static")