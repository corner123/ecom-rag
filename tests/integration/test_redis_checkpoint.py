"""Redis Stack-backed persistence tests for the durable trade graph."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date
import os
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from redis.asyncio import Redis

from trade_agent.agents.checkpoint import (
    CheckpointUnavailableError,
    RedisCheckpointFactory,
    resume_run,
)
from trade_agent.agents.graph import GraphDependencies, build_trade_graph
from trade_agent.agents.intent import IntentParser, QueryIntent
from trade_agent.agents.nodes import FileEvidenceRepository, GraphBudgets
from trade_agent.config.settings import RedisSettings
from tests.unit.test_evidence_validator import _rag, _sql


AS_OF = date(2026, 9, 4)


def _redis_port() -> int:
    return int(os.environ.get("REDIS_TEST_PORT", "6379"))


def _redis_host() -> str:
    return os.environ.get("REDIS_TEST_HOST", os.environ.get("REDIS__HOST", "127.0.0.1"))


@dataclass
class RecordingBranch:
    evidence: tuple
    calls: int = 0
    supports_finite_timeout: bool = True

    def run(
        self,
        intent: QueryIntent,
        *,
        timeout_seconds: float,
        identity_hmac_key: bytes | None = None,
        entity_bindings=None,
        candidate_limit: int = 100,
    ):
        del intent, timeout_seconds, identity_hmac_key, entity_bindings, candidate_limit
        self.calls += 1
        return self.evidence, ()


@pytest_asyncio.fixture
async def redis_client() -> Redis:
    client = Redis(host=_redis_host(), port=_redis_port(), decode_responses=True)
    try:
        await client.ping()
    except Exception as error:  # pragma: no cover - an environment prerequisite
        await client.aclose()
        pytest.fail(f"Redis Stack must be running for this integration test: {type(error).__name__}")
    yield client
    await client.aclose()


def _deps(tmp_path: Path, sql: RecordingBranch, rag: RecordingBranch) -> GraphDependencies:
    return GraphDependencies(
        intent_parser=IntentParser(as_of=AS_OF),
        sql_branch=sql,
        rag_branch=rag,
        evidence_repository=FileEvidenceRepository(tmp_path / "evidence"),
        sql_identity_hmac_key=b"checkpoint-test-key",
        budgets=GraphBudgets(node_timeout_seconds=0.2),
        as_of=AS_OF,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graph_resumes_without_repeating_completed_retrieval_or_sql(
    tmp_path: Path, redis_client: Redis
) -> None:
    namespace = f"trade-checkpoint-{uuid4().hex}"
    saver = await RedisCheckpointFactory.create(
        RedisSettings(host=_redis_host(), port=_redis_port()), ttl_seconds=60, namespace=namespace
    )
    sql = RecordingBranch((_sql(start=date(2025, 9, 4), scope="lead"),))
    rag = RecordingBranch((_rag(suffix="checkpoint"), _rag(suffix="checkpoint-2", source_type="industry_news")))
    thread_id, run_id = uuid4().hex, uuid4().hex
    config = {"configurable": {"thread_id": thread_id, "run_id": run_id}, "recursion_limit": 64}
    resumed_saver = None
    try:
        first_graph = build_trade_graph(_deps(tmp_path, sql, rag), checkpointer=saver, interrupt_before=["answer_draft"])
        paused = await first_graph.ainvoke({"question": "Acme 是否值得跟进"}, config)
        assert paused["node_status"]["sql_node"] == "completed"
        assert paused["node_status"]["rag_node"] == "completed"
        assert sql.calls == rag.calls == 1
        recovered = await resume_run(saver, thread_id=thread_id, run_id=run_id)
        assert recovered["node_status"]["sql_node"] == "completed"
        await RedisCheckpointFactory.close(saver)

        resumed_saver = await RedisCheckpointFactory.create(
            RedisSettings(host=_redis_host(), port=_redis_port()), ttl_seconds=60, namespace=namespace
        )
        resumed_graph = build_trade_graph(_deps(tmp_path, sql, rag), checkpointer=resumed_saver)
        result = await resumed_graph.ainvoke(None, config)
        assert result["terminal"] is True
        assert sql.calls == rag.calls == 1
    finally:
        cleanup_saver = resumed_saver or saver
        await cleanup_saver.adelete_thread(thread_id)
        await RedisCheckpointFactory.close(cleanup_saver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_checkpoint_payload_excludes_raw_sql_rows_and_draft_text(
    tmp_path: Path, redis_client: Redis
) -> None:
    namespace = f"trade-checkpoint-{uuid4().hex}"
    saver = await RedisCheckpointFactory.create(
        RedisSettings(host=_redis_host(), port=_redis_port()), ttl_seconds=60, namespace=namespace
    )
    thread_id, run_id = uuid4().hex, uuid4().hex
    config = {"configurable": {"thread_id": thread_id, "run_id": run_id}, "recursion_limit": 64}
    sql = RecordingBranch((_sql(start=date(2025, 9, 4), scope="lead"),))
    rag = RecordingBranch((_rag(suffix="checkpoint"), _rag(suffix="checkpoint-2", source_type="industry_news")))
    try:
        graph = build_trade_graph(_deps(tmp_path, sql, rag), checkpointer=saver)
        result = await graph.ainvoke({"question": "Acme 是否值得跟进"}, config)
        raw_payload = await RedisCheckpointFactory.dump_thread(redis_client, thread_id, namespace=namespace)
        assert "raw_rows" not in raw_payload
        assert "draft answer" not in raw_payload.casefold()
        if result["answer"] is not None:
            assert result["answer"] not in raw_payload
        assert "sql_evidence_refs" in raw_payload
        assert "payload_sha256" in raw_payload
    finally:
        await saver.adelete_thread(thread_id)
        await RedisCheckpointFactory.close(saver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_missing_checkpoint_fails_closed(redis_client: Redis) -> None:
    saver = await RedisCheckpointFactory.create(
        RedisSettings(host=_redis_host(), port=_redis_port()), ttl_seconds=1, namespace=f"trade-checkpoint-{uuid4().hex}"
    )
    try:
        with pytest.raises(CheckpointUnavailableError, match="checkpoint_unavailable"):
            await resume_run(saver, thread_id=uuid4().hex, run_id=uuid4().hex)
    finally:
        await RedisCheckpointFactory.close(saver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unavailable_redis_connection_is_a_typed_closed_failure() -> None:
    with pytest.raises(CheckpointUnavailableError, match="checkpoint_unavailable"):
        await RedisCheckpointFactory.create(
            RedisSettings(host="127.0.0.1", port=1), namespace=f"trade-checkpoint-{uuid4().hex}"
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_checkpoint_ttl_expires_thread_state(
    tmp_path: Path, redis_client: Redis
) -> None:
    namespace = f"trade-checkpoint-{uuid4().hex}"
    saver = await RedisCheckpointFactory.create(
        RedisSettings(host=_redis_host(), port=_redis_port()), ttl_seconds=1, namespace=namespace
    )
    thread_id, run_id = uuid4().hex, uuid4().hex
    config = {"configurable": {"thread_id": thread_id, "run_id": run_id}, "recursion_limit": 64}
    try:
        sql = RecordingBranch((_sql(start=date(2025, 9, 4), scope="lead"),))
        rag = RecordingBranch((_rag(suffix="checkpoint"), _rag(suffix="checkpoint-2", source_type="industry_news")))
        graph = build_trade_graph(_deps(tmp_path, sql, rag), checkpointer=saver)
        await graph.ainvoke({"question": "Acme 是否值得跟进"}, config)
        assert await RedisCheckpointFactory.dump_thread(redis_client, thread_id, namespace=namespace)
        await asyncio.sleep(1.1)
        with pytest.raises(CheckpointUnavailableError, match="checkpoint_unavailable"):
            await resume_run(saver, thread_id=thread_id, run_id=run_id)
    finally:
        await RedisCheckpointFactory.close(saver)
