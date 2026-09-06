"""Redis Stack persistence for minimal, resumable trade workflow state."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, ClassVar, cast

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from redis.asyncio import Redis
from redis.exceptions import RedisError

from trade_agent.agents.intent import QueryIntent
from trade_agent.agents.state import TradeIntelState
from trade_agent.config.settings import RedisSettings
from trade_agent.errors import GraphWorkflowError


class CheckpointUnavailableError(GraphWorkflowError):
    code = "checkpoint_unavailable"

    def __init__(self) -> None:
        super().__init__(self.code)


_SAFE_STATE_FIELDS = frozenset({
    "idempotency_key", "intent", "route_plan", "sql_evidence_refs",
    "rag_evidence_refs", "evidence_refs", "sql_report", "rag_report",
    "branch_reports", "validation", "draft_ref", "refusal_reason",
    "rewrite_count", "retry_count", "llm_calls", "step_count", "errors",
    "node_status", "terminal",
})


def _safe_intent(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    try:
        checked = QueryIntent.model_validate({**value, "question": "resume"})
    except (TypeError, ValueError):
        return {}
    return checked.model_dump(mode="json", exclude={"question"})


def _safe_errors(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [
        {key: item[key] for key in ("code", "node", "retryable") if key in item}
        for item in value if isinstance(item, Mapping)
    ]


def _checkpoint_safe(values: Mapping[str, object]) -> dict[str, object]:
    """Project only reviewed durable fields; never redact a blacklist in place."""
    safe = {key: values[key] for key in _SAFE_STATE_FIELDS & values.keys()}
    if "intent" in safe:
        safe["intent"] = _safe_intent(safe["intent"])
    if "errors" in safe:
        safe["errors"] = _safe_errors(safe["errors"])
    return safe


def _safe_write(channel: str, value: object, idempotency_key: str) -> tuple[str, object] | None:
    if channel == "idempotency_key":
        if value != idempotency_key:
            raise CheckpointUnavailableError()
        return channel, idempotency_key
    if channel in _SAFE_STATE_FIELDS:
        projected = _checkpoint_safe({channel: value})
        return (channel, projected[channel]) if channel in projected else None
    if channel.startswith("branch:"):
        return channel, value
    return None


def _run_namespace(run_id: str) -> str:
    if not isinstance(run_id, str) or not run_id.strip() or any(char.isspace() for char in run_id):
        raise ValueError("run_id must be nonblank and whitespace-free")
    return f"run:{run_id}"


def checkpoint_config(thread_id: str, run_id: str, idempotency_key: str) -> dict[str, dict[str, str]]:
    """Build the only supported checkpoint config for an idempotent graph run."""
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("thread_id must be nonblank")
    if not isinstance(idempotency_key, str) or not idempotency_key.strip():
        raise ValueError("idempotency_key must be nonblank")
    return {"configurable": {
        "thread_id": thread_id,
        "run_id": run_id,
        "checkpoint_ns": _run_namespace(run_id),
        "idempotency_key": idempotency_key,
    }}


def _scoped_config(config: Mapping[str, object], run_id: object) -> dict[str, object]:
    if not isinstance(run_id, str):
        raise CheckpointUnavailableError()
    scoped = dict(config)
    configurable = config.get("configurable")
    if not isinstance(configurable, Mapping):
        raise CheckpointUnavailableError()
    scoped["configurable"] = {
        **configurable,
        "run_id": run_id,
        "checkpoint_ns": _run_namespace(run_id),
    }
    return scoped


class _MinimalAsyncRedisSaver(AsyncRedisSaver):
    """AsyncRedisSaver which accepts only checkpoint-safe state channels."""

    async def aput(self, config, checkpoint, metadata, new_versions, stream_mode="values"):
        configurable = config.get("configurable", {})
        run_id = configurable.get("run_id", metadata.get("run_id"))
        scoped_config = _scoped_config(config, run_id)
        idempotency_key = configurable.get("idempotency_key")
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise CheckpointUnavailableError()
        safe_checkpoint = dict(checkpoint)
        values = checkpoint.get("channel_values", {})
        safe_checkpoint["channel_values"] = _checkpoint_safe(values if isinstance(values, Mapping) else {})
        persisted_key = safe_checkpoint["channel_values"].get("idempotency_key")
        if persisted_key is not None and persisted_key != idempotency_key:
            raise CheckpointUnavailableError()
        safe_checkpoint["channel_values"]["idempotency_key"] = idempotency_key
        try:
            saved = await super().aput(scoped_config, safe_checkpoint, metadata, new_versions, stream_mode)
            saved["configurable"]["run_id"] = run_id
            saved["configurable"]["idempotency_key"] = idempotency_key
            return saved
        except (OSError, RedisError, TimeoutError):
            raise CheckpointUnavailableError() from None

    async def aput_writes(self, config, writes, task_id, task_path=""):
        configurable = config.get("configurable", {})
        scoped_config = _scoped_config(config, configurable.get("run_id"))
        idempotency_key = configurable.get("idempotency_key")
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise CheckpointUnavailableError()
        safe_writes = tuple(
            item for channel, value in writes
            if (item := _safe_write(channel, value, idempotency_key)) is not None
        )
        try:
            return await super().aput_writes(scoped_config, safe_writes, task_id, task_path)
        except (OSError, RedisError, TimeoutError):
            raise CheckpointUnavailableError() from None

    async def aget_tuple(self, config):
        try:
            return await super().aget_tuple(_scoped_config(config, config.get("configurable", {}).get("run_id")))
        except (OSError, RedisError, TimeoutError):
            raise CheckpointUnavailableError() from None

    async def alist(self, config, *, filter=None, before=None, limit=None):
        try:
            scoped_config = None if config is None else _scoped_config(config, config.get("configurable", {}).get("run_id"))
            async for item in super().alist(scoped_config, filter=filter, before=before, limit=limit):
                yield item
        except (OSError, RedisError, TimeoutError):
            raise CheckpointUnavailableError() from None


class RedisCheckpointFactory:
    """Create Redis Stack checkpointers with isolated keys and an explicit TTL."""

    _default_saver: ClassVar[AsyncRedisSaver | None] = None

    @staticmethod
    async def create(settings: RedisSettings, *, ttl_seconds: int | None = None, namespace: str = "trade-agent") -> AsyncRedisSaver:
        if type(settings) is not RedisSettings:
            raise TypeError("settings must be RedisSettings")
        if not namespace or any(character.isspace() for character in namespace):
            raise ValueError("checkpoint namespace must be nonblank and whitespace-free")
        ttl = settings.checkpoint_ttl_seconds if ttl_seconds is None else ttl_seconds
        if type(ttl) is not int or ttl < 1:
            raise ValueError("checkpoint TTL must be a positive integer")
        saver = _MinimalAsyncRedisSaver(
            f"redis://{settings.host}:{settings.port}/{settings.database}",
            ttl={"default_ttl": ttl / 60, "refresh_on_read": True},
            checkpoint_prefix=f"{namespace}:checkpoint",
            checkpoint_write_prefix=f"{namespace}:checkpoint_write",
        )
        try:
            await saver.asetup()
            await saver._redis.ping()
        except Exception:
            try:
                await RedisCheckpointFactory.close(saver)
            except Exception:
                pass
            raise CheckpointUnavailableError() from None
        RedisCheckpointFactory._default_saver = saver
        return saver

    @staticmethod
    async def close(saver: AsyncRedisSaver) -> None:
        if RedisCheckpointFactory._default_saver is saver:
            RedisCheckpointFactory._default_saver = None
        if (client := getattr(saver, "_redis", None)) is not None:
            await client.aclose()

    @staticmethod
    async def dump_thread(client: Redis, thread_id: str, *, namespace: str) -> str:
        documents: list[object] = []
        async for key in client.scan_iter(match=f"{namespace}:*{thread_id}*"):
            if await client.type(key) == "ReJSON-RL":
                if (value := await client.json().get(key)) is not None:
                    documents.append(value)
        return json.dumps(documents, ensure_ascii=False, sort_keys=True)


async def _resume_from_saver(checkpointer: AsyncRedisSaver, thread_id: str, run_id: str) -> TradeIntelState:
    config = checkpoint_config(thread_id, run_id, "resume-lookup")
    try:
        checkpoints = [item async for item in checkpointer.alist(config, limit=1)]
    except CheckpointUnavailableError:
        raise
    except Exception:
        raise CheckpointUnavailableError() from None
    if not checkpoints:
        raise CheckpointUnavailableError()
    state = checkpoints[0].checkpoint.get("channel_values", {})
    if not isinstance(state, dict):
        raise CheckpointUnavailableError()
    return cast(TradeIntelState, state.copy())


async def resume_run(thread_id: str, run_id: str) -> TradeIntelState:
    """Return one exact run's safe state, or fail closed when unavailable."""
    if RedisCheckpointFactory._default_saver is None:
        raise CheckpointUnavailableError()
    return await _resume_from_saver(RedisCheckpointFactory._default_saver, thread_id, run_id)
