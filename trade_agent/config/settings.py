"""Typed, environment-driven settings for the local trade-agent stack."""
from __future__ import annotations
import os
from typing import Literal
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PLACEHOLDERS = {"change-me", "replace-me", "password", "secret", ""}

def _is_placeholder(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.strip().lower()
    return normalized in _PLACEHOLDERS or normalized.startswith(("replace-", "change-", "example-"))

class MysqlSettings(BaseModel):
    host: str = "mysql"
    port: int = Field(default=3306, ge=1, le=65535)
    database: str = "foreign_trade_db"
    user: str = "trade_app"
    password: str | None = None

    @model_validator(mode="after")
    def reject_placeholder_password(self) -> "MysqlSettings":
        if _is_placeholder(self.password):
            raise ValueError("mysql password must be provided and must not be a placeholder")
        return self

class MilvusSettings(BaseModel):
    host: str = "milvus"
    port: int = Field(default=19530, ge=1, le=65535)
    database: str = "default"

class RedisSettings(BaseModel):
    host: str = "redis"
    port: int = Field(default=6379, ge=1, le=65535)
    database: int = Field(default=0, ge=0, le=15)

class ModelSettings(BaseModel):
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    generation_provider: str = "deterministic"

class RuntimeLimits(BaseModel):
    max_graph_steps: int = Field(default=12, gt=0, le=50)
    max_retries: int = Field(default=1, ge=0, le=3)
    max_llm_calls: int = Field(default=5, ge=0, le=10)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")
    environment: Literal["test", "development", "compose"] = "development"
    mysql: MysqlSettings = Field(default_factory=MysqlSettings)
    milvus: MilvusSettings = MilvusSettings()
    redis: RedisSettings = RedisSettings()
    models: ModelSettings = ModelSettings()
    limits: RuntimeLimits = RuntimeLimits()

    @classmethod
    def load(cls, runtime: Literal["test", "development", "compose"] | None = None) -> "Settings":
        values = {"environment": runtime} if runtime is not None else {}
        flat_password = os.environ.get("MYSQL_APP_PASSWORD")
        if flat_password is not None:
            values["mysql"] = {"password": flat_password}
        return cls(**values)
