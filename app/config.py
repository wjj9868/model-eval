# @author: ztwz
"""应用配置：环境变量（前缀 MODEL_EVAL_）或 .env 文件读取，pydantic-settings 校验。"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置项，环境变量优先于 .env 文件"""

    model_config = SettingsConfigDict(env_prefix="MODEL_EVAL_", env_file=".env", env_file_encoding="utf-8")

    # ===== 数据库（Docker MySQL 3306） =====
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "123456"
    mysql_db: str = "model_eval"
    mysql_charset: str = "utf8mb4"

    # ===== Redis =====
    redis_url: str = "redis://127.0.0.1:6379/0"
    celery_broker_url: str = "redis://127.0.0.1:6379/1"
    celery_result_backend: str = "redis://127.0.0.1:6379/2"

    # ===== 模型调用 =====
    default_timeout_seconds: int = 300
    model_connect_timeout: float = 10

    # ===== 任务 =====
    task_always_eager: bool = False

    # ===== 静态资源 =====
    static_dir: str = "app/static"

    def mysql_url_for(self, db: str | None = None) -> str:
        """构造指定数据库的连接串（默认业务库）"""

        target = db or self.mysql_db
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{target}?charset={self.mysql_charset}"
        )


@lru_cache
def get_settings() -> Settings:
    """获取全局设置（缓存，测试中可通过环境变量覆盖）"""

    return Settings()