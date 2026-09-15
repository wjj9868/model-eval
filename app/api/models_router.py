# @author: ztwz
"""模型配置路由：增删改查、启停、连通性测试、删除保护。"""
import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.exceptions import BusinessError
from app.models import EvalTask, ModelConfig
from app.schemas import (
    ModelConfigCreate,
    ModelConfigEnabledUpdate,
    ModelConfigOut,
    ModelConfigUpdate,
    PageOut,
    TestConnectionResult,
)
from app.services.model_runner import ModelCallError, call_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["模型配置"])


def _get_or_404(db: Session, model_id: int) -> ModelConfig:
    config = db.get(ModelConfig, model_id)
    if config is None:
        raise BusinessError("模型配置不存在", status_code=404)
    return config


def _check_name_duplicate(db: Session, name: str, exclude_id: int | None = None) -> None:
    """配置名称唯一性校验（排除自身）"""

    stmt = select(ModelConfig.id).where(ModelConfig.name == name)
    if exclude_id is not None:
        stmt = stmt.where(ModelConfig.id != exclude_id)
    if db.scalar(stmt) is not None:
        raise BusinessError("配置名称已存在")


@router.get("", response_model=PageOut[ModelConfigOut])
def list_models(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    keyword: str | None = Query(None, max_length=100),
    db: Session = Depends(get_db),
):
    """模型配置分页列表（关键字匹配名称/模型名）"""

    stmt = select(ModelConfig)
    if keyword:
        like = f"%{keyword}%"
        stmt = stmt.where(ModelConfig.name.like(like) | ModelConfig.model_name.like(like))
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    items = db.scalars(stmt.order_by(ModelConfig.id.desc()).offset((page - 1) * page_size).limit(page_size)).all()
    return PageOut[ModelConfigOut](items=list(items), total=total, page=page, page_size=page_size)


@router.post("", response_model=ModelConfigOut)
def create_model(payload: ModelConfigCreate, db: Session = Depends(get_db)):
    """新建模型配置"""

    _check_name_duplicate(db, payload.name)
    config = ModelConfig(**payload.model_dump())
    db.add(config)
    db.commit()
    db.refresh(config)
    return config


@router.put("/{model_id}", response_model=ModelConfigOut)
def update_model(model_id: int, payload: ModelConfigUpdate, db: Session = Depends(get_db)):
    """修改模型配置"""

    config = _get_or_404(db, model_id)
    _check_name_duplicate(db, payload.name, exclude_id=model_id)
    for field, value in payload.model_dump().items():
        setattr(config, field, value)
    db.commit()
    db.refresh(config)
    return config


@router.patch("/{model_id}/enabled", response_model=ModelConfigOut)
def toggle_model_enabled(model_id: int, payload: ModelConfigEnabledUpdate, db: Session = Depends(get_db)):
    """启停模型（停用后可再启用，不影响历史批次数据）"""

    config = _get_or_404(db, model_id)
    config.enabled = payload.enabled
    db.commit()
    db.refresh(config)
    return config


@router.delete("/{model_id}")
def delete_model(model_id: int, db: Session = Depends(get_db)):
    """删除模型配置（已被评测批次引用时禁止，避免历史结果缺失）"""

    config = _get_or_404(db, model_id)
    referenced = db.scalar(select(func.count()).select_from(EvalTask).where(EvalTask.model_id == model_id))
    if referenced:
        raise BusinessError(f"该模型已被 {referenced} 条评测任务引用，不能删除，可改为停用")
    db.delete(config)
    db.commit()
    return {"deleted": True}


@router.post("/{model_id}/test", response_model=TestConnectionResult)
def test_model(model_id: int, db: Session = Depends(get_db)):
    """连通性测试：发送极简输入验证模型服务可达（短超时）"""

    config = _get_or_404(db, model_id)
    try:
        result = call_model(config, "你好", timeout_override=min(config.timeout_seconds, 30))
        return TestConnectionResult(
            success=True,
            message="连接成功",
            response_time_ms=result.response_time_ms,
            sample=result.text[:100],
        )
    except ModelCallError as exc:
        return TestConnectionResult(success=False, message=str(exc))
    except Exception as exc:  # noqa: BLE001 未知异常统一结构化返回
        logger.exception("连通性测试异常，model_id=%s", model_id)
        return TestConnectionResult(success=False, message=f"未知错误：{type(exc).__name__}")