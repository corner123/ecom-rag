"""Redis Stack persistence for minimal, resumable trade workflow state."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from redis.asyncio import Redis
from redis.exceptions import RedisError

from trade_agent.agents.state import TradeIntelState
from trade_agent.config.settings import RedisSettings
from trade_agent.errors import GraphWorkflowError


class CheckpointUnavailableError(GraphWorkflowError):
    """A closed failure for unavailable, expired, or missing checkpoints."""

    code = "checkpoint_unavailable"

    def __init__(self) -> None:
        super().__init__(self.code)


_OMITTED_STATE_FIELDS = frozenset({"answer", "claims", "raw_rows", "draft_answer"})


def _checkpoint_safe(value: Any) -> Any:
    """Remove answer/source payloads while retaining resume metadata and refs."""
    if isinstance(value, Mapping):
        return {
            key: _checkpoint_safe(item)
            for key, item in value.items()
            if key not in _OMITTED_STATE_FIELDS
        }
    if isinstance(value, list):
        return [_checkpoint_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_checkpoint_safe(item) for item in value)
    return value


class _MinimalAsyncRedisSaver(AsyncRedisSaver):
    """AsyncRedisSaver which never persists answer text or raw source payloads."""

    async def aput(self, config, checkpoint, metadata, new_versions, stream_mode="values"):
        safe_checkpoint = dict(checkpoint)
        safe_checkpoint["channel_values"] = _checkpoint_safe(
            checkpoint.get("channel_values", {})
        )
        try:
            return await super().aput(
                config, safe_checkpoint, metadata, new_versions, stream_mode
            )
        except (OSError, RedisError, TimeoutError) as error:
            raise CheckpointUnavailableError() from None

    async def aput_writes(self, config, writes, task_id, task_path=""):
        safe_writes = tuple(
            (channel, _checkpoint_safe(value))
            for channel, value in writes
            if channel not in _OMITTED_STATE_FIELDS
        )
        try:
            return await super().aput_writes(
                config, safe_writes, task_id, task_path
            )
        except (OSError, RedisError, TimeoutError) as error:
            raise CheckpointUnavailableError() from None

    async def aget_tuple(self, config):
        try:
            return await super().aget_tuple(config)
        except (OSError, RedisError, TimeoutError) as error:
            raise CheckpointUnavailableError() from None

    async def alist(self, config, *, filter=None, before=None, limit=None):
        try:
            async for item in super().alist(
                config, filter=filter, before=before, limit=limit
            ):
                yield item
        except (OSError, RedisError, TimeoutError) as error:
            raise CheckpointUnavailableError() from None


class RedisCheckpointFactory:
    """Create Redis Stack checkpointers with isolated keys and an explicit TTL."""

    @staticmethod
    async def create(
        settings: RedisSettings,
        *,
        ttl_seconds: int | None = None,
        namespace: str = "trade-agent",
    ) -> AsyncRedisSaver:
        if type(settings) is not RedisSettings:
            raise TypeError("settings must be RedisSettings")
        if not namespace or any(character.isspace() for character in namespace):
            raise ValueError("checkpoint namespace must be nonblank and whitespace-free")
        ttl = settings.checkpoint_ttl_seconds if ttl_seconds is None else ttl_seconds
        if type(ttl) is not int or ttl < 1:
            raise ValueError("checkpoint TTL must be a positive integer")

        redis_url = f"redis://{settings.host}:{settings.port}/{settings.database}"
        saver = _MinimalAsyncRedisSaver(
            redis_url,
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
        return saver

    @staticmethod
    async def close(saver: AsyncRedisSaver) -> None:
        close = getattr(saver, "_redis", None)
        if close is not None:
            await close.aclose()

    @staticmethod
    async def dump_thread(
        client: Redis,
        thread_id: str,
        *,
        namespace: str,
    ) -> str:
        """Return RedisJSON documents for an integration-only sensitive-data check."""
        documents: list[object] = []
        async for key in client.scan_iter(match=f"{namespace}:*{thread_id}*"):
            if await client.type(key) != "ReJSON-RL":
                continue
            value = await client.json().get(key)
            if value is not None:
                documents.append(value)
        return json.dumps(documents, ensure_ascii=False, sort_keys=True)


async def resume_run(
    checkpointer: AsyncRedisSaver,
    *,
    thread_id: str,
    run_id: str,
) -> TradeIntelState:
    """Load a matching checkpoint or fail closed; never initialize a fresh run."""
    if not thread_id or not run_id:
        raise ValueError("thread_id and run_id must be nonblank")
    config = {"configurable": {"thread_id": thread_id, "run_id": run_id}}
    try:
        checkpoints = [
            item
            async for item in checkpointer.alist(config, limit=1)
        ]
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
