"""Typed, environment-driven settings for the local trade-agent stack."""
from __future__ import annotations
import os
import re
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator
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
    # Runtime code is deliberately query-only.  Migration credentials are
    # supplied only to the one-shot migration/seed commands, never Settings.
    user: str = "trade_query"
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
    checkpoint_ttl_seconds: int = Field(default=3_600, ge=1, le=86_400)

class ModelSettings(BaseModel):
    embedding_model: str = "BAAI/bge-m3"
    embedding_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=16, gt=0, le=256)
    embedding_cache_dir: str | None = None
    embedding_offline: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    generation_provider: str = "deterministic"

    @field_validator("embedding_revision", "reranker_revision")
    @classmethod
    def immutable_model_revision(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("model revision must be an immutable 40-character lowercase SHA")
        return value

    @model_validator(mode="after")
    def pin_retrieval_models(self) -> "ModelSettings":
        if self.embedding_model != "BAAI/bge-m3" or self.embedding_revision != "5617a9f61b028005a4858fdac845db406aefb181":
            raise ValueError("production embedding settings must use pinned BAAI/bge-m3")
        if self.reranker_model != "BAAI/bge-reranker-v2-m3" or self.reranker_revision != "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e":
            raise ValueError("production reranker settings must use the pinned BGE reranker")
        if not self.embedding_device.strip():
            raise ValueError("embedding device must not be blank")
        if self.embedding_cache_dir is not None and not self.embedding_cache_dir.strip():
            raise ValueError("embedding cache directory must not be blank")
        return self

class RuntimeLimits(BaseModel):
    max_graph_steps: int = Field(default=12, gt=0, le=50)
    max_retries: int = Field(default=1, ge=0, le=3)
    max_llm_calls: int = Field(default=5, ge=0, le=10)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_nested_delimiter="__", env_ignore_empty=True, extra="ignore"
    )
    environment: Literal["test", "development", "compose"] = "development"
    mysql: MysqlSettings = Field(default_factory=MysqlSettings)
    milvus: MilvusSettings = MilvusSettings()
    redis: RedisSettings = RedisSettings()
    models: ModelSettings = ModelSettings()
    limits: RuntimeLimits = RuntimeLimits()

    @classmethod
    def load(cls, runtime: Literal["test", "development", "compose"] | None = None) -> "Settings":
        values = {"environment": runtime} if runtime is not None else {}
        # MYSQL__QUERY_PASSWORD is the canonical runtime credential.  Keep
        # MYSQL__PASSWORD and MYSQL_APP_PASSWORD as compatibility aliases for
        # older local tests and development shells.
        query_password = os.environ.get("MYSQL__QUERY_PASSWORD")
        legacy_password = os.environ.get("MYSQL__PASSWORD") or os.environ.get("MYSQL_APP_PASSWORD")
        # An explicit legacy alias wins when present so existing test and
        # development callers can override the Compose-provided query value.
        # In the normal compose/development path QUERY_PASSWORD is the only
        # value and therefore remains the default runtime credential.
        password = legacy_password if legacy_password is not None else query_password
        if runtime == "test" and legacy_password is None:
            # Tests must opt into a password explicitly; never inherit a
            # developer/Compose secret accidentally.
            password = None
        if password is not None:
            values["mysql"] = {
                "user": os.environ.get("MYSQL__QUERY_USER", "trade_query"),
                "password": password,
            }
        return cls(**values)
