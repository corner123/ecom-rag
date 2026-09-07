"""Runtime composition and guarded graph projection for the HTTP boundary."""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

import anyio

from trade_agent.agents.state import EvidenceRef, GuardProjection, RequestRef
from trade_agent.api.models import (
    PublicError,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
    RetrievalHitResponse,
    RetrieveRequest,
    RetrievalResponse,
    RetrievalTrace,
)
from trade_agent.errors import EvidenceRepositoryError
from trade_agent.evidence.models import Conflict, Evidence


class RuntimeUnavailableError(RuntimeError):
    """A typed safe error crossing from runtime composition to HTTP."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@runtime_checkable
class ApiRuntime(Protocol):
    async def readiness(self) -> ReadinessResponse: ...
    async def query(self, request: QueryRequest) -> QueryResponse: ...
    async def retrieve(self, request: RetrieveRequest) -> RetrievalResponse: ...
    async def get_run(self, run_id: str) -> QueryResponse | None: ...
    async def resume(self, run_id: str) -> QueryResponse | None: ...
    async def get_evidence(self, evidence_id: str) -> Evidence | None: ...
    async def close(self) -> None: ...


def _model_sequence(model: type[Any], values: object) -> tuple[Any, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RuntimeUnavailableError("invalid_runtime_contract")
    try:
        return tuple(
            model.model_validate_json(json.dumps(value, ensure_ascii=False))
            if isinstance(value, Mapping)
            else model.model_validate(value)
            for value in values
        )
    except Exception:
        raise RuntimeUnavailableError("invalid_runtime_contract") from None


def _public_errors(values: object) -> tuple[PublicError, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    projected: list[PublicError] = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        try:
            projected.append(
                PublicError(
                    code=value["code"],
                    node=value["node"],
                    retryable=value["retryable"],
                )
            )
        except Exception:
            projected.append(
                PublicError(code="invalid_runtime_error", node="runtime", retryable=False)
            )
    return tuple(projected)


@dataclass
class AgentRuntime:
    """Bounded adapter from compiled graphs and retrieval to public contracts."""

    graph_factory: Callable[[int], Any]
    evidence_repository: Any
    build_id: str
    readiness_probe: Callable[[], Awaitable[ReadinessResponse]]
    retrieval_service: Any | None = None
    intent_parser: Any | None = None
    resume_state_loader: Callable[[str], Awaitable[Mapping[str, object]]] | None = None
    close_callback: Callable[[], Awaitable[None]] | None = None
    max_outstanding_requests: int = 8

    def __post_init__(self) -> None:
        if not self.build_id.startswith("build_") or len(self.build_id) != 38:
            raise ValueError("runtime requires a published build identity")
        if type(self.max_outstanding_requests) is not int or not 1 <= self.max_outstanding_requests <= 128:
            raise ValueError("max outstanding requests must be from 1 through 128")
        self._limiter = anyio.CapacityLimiter(self.max_outstanding_requests)
        self._runs: dict[str, QueryResponse] = {}
        self._evidence_refs: dict[str, EvidenceRef] = {}
        self._closed = False

    async def readiness(self) -> ReadinessResponse:
        if self._closed:
            raise RuntimeUnavailableError("runtime_closed")
        result = await self.readiness_probe()
        if type(result) is not ReadinessResponse:
            raise RuntimeUnavailableError("invalid_readiness_contract")
        return ReadinessResponse.model_validate(result.model_dump(mode="python"))

    async def query(self, request: QueryRequest) -> QueryResponse:
        if type(request) is not QueryRequest:
            raise TypeError("request must be an exact QueryRequest")
        if self._closed:
            raise RuntimeUnavailableError("runtime_closed")
        async with self._limiter:
            run_id = uuid4().hex
            idempotency_key = request.idempotency_key or uuid4().hex
            graph = self.graph_factory(request.top_k)
            graph_input: dict[str, object] = {
                "question": request.question,
                "idempotency_key": idempotency_key,
            }
            if request.explicit_filters != type(request.explicit_filters)():
                graph_input["explicit_filters"] = request.explicit_filters.model_dump(mode="json")
            try:
                state = await graph.ainvoke(
                    graph_input,
                    {
                        "configurable": {
                            "thread_id": run_id,
                            "run_id": run_id,
                            "checkpoint_ns": f"run:{run_id}",
                            "idempotency_key": idempotency_key,
                        },
                        "recursion_limit": 128,
                    },
                )
            except Exception:
                raise RuntimeUnavailableError("workflow_unavailable") from None
            response = self._project_run(run_id, state)
            self._runs[run_id] = response
            return response

    def _project_run(self, run_id: str, state: object) -> QueryResponse:
        if not isinstance(state, Mapping):
            raise RuntimeUnavailableError("invalid_runtime_contract")
        refs = _model_sequence(EvidenceRef, state.get("evidence_refs", ()))
        try:
            evidence = self.evidence_repository.get_many(refs)
        except EvidenceRepositoryError:
            raise RuntimeUnavailableError("evidence_repository_error") from None
        if any(type(item) is not Evidence for item in evidence):
            raise RuntimeUnavailableError("invalid_runtime_contract")
        for ref in refs:
            self._evidence_refs[ref.evidence_id] = ref

        conflicts = _model_sequence(Conflict, state.get("conflicts", ()))
        raw_guard = state.get("guard")
        if raw_guard is not None:
            try:
                guard = GuardProjection.model_validate_json(
                    json.dumps(raw_guard, ensure_ascii=False)
                )
            except Exception:
                raise RuntimeUnavailableError("invalid_runtime_contract") from None
            claims = guard.claims if guard.accepted else ()
            answer = "\n".join(claim.text for claim in claims) if guard.accepted else None
            refusal = guard.refusal_reason
        else:
            if state.get("answer") is not None or state.get("claims"):
                raise RuntimeUnavailableError("unguarded_answer")
            claims = ()
            answer = None
            refusal_value = state.get("refusal_reason") or "workflow_failed"
            refusal = refusal_value if isinstance(refusal_value, str) else "workflow_failed"

        cited = {item for claim in claims for item in claim.evidence_ids}
        cited.update(item for conflict in conflicts for item in conflict.evidence_ids)
        public_evidence = tuple(
            sorted((item for item in evidence if item.evidence_id in cited), key=lambda item: item.evidence_id)
        )
        status = "completed" if answer is not None else "refused"
        degraded = tuple(
            sorted(
                {
                    component
                    for report in state.get("branch_reports", ())
                    if isinstance(report, Mapping)
                    for component in report.get("degraded_components", ())
                    if isinstance(component, str) and component.strip()
                }
            )
        )
        try:
            return QueryResponse(
                run_id=run_id,
                status=status,
                build_id=self.build_id,
                answer=answer,
                claims=claims,
                evidence=public_evidence,
                conflicts=conflicts,
                refusal_reason=refusal,
                errors=_public_errors(state.get("errors", ())),
                degraded_components=degraded,
            )
        except Exception:
            raise RuntimeUnavailableError("invalid_runtime_contract") from None

    async def retrieve(self, request: RetrieveRequest) -> RetrievalResponse:
        if type(request) is not RetrieveRequest:
            raise TypeError("request must be an exact RetrieveRequest")
        if self.retrieval_service is None or self.intent_parser is None:
            raise RuntimeUnavailableError("retrieval_unavailable")
        async with self._limiter:
            try:
                from trade_agent.agents.intent import to_retrieval_query_intent

                intent = self.intent_parser.parse(request.question, request.explicit_filters)
                outcome = await anyio.to_thread.run_sync(
                    lambda: self.retrieval_service.search(
                        to_retrieval_query_intent(intent),
                        top_k=request.top_k,
                        candidate_limit=100,
                        transport_timeout_seconds=5.0,
                    )
                )
                hits = tuple(
                    RetrievalHitResponse(
                        chunk_id=hit.chunk_id,
                        rank=hit.rank,
                        record=hit.record,
                        trace=RetrievalTrace(
                            build_id=hit.trace.build_id,
                            profile_id=hit.trace.profile_id,
                            profile_version=hit.trace.profile_version,
                            planner_version=hit.trace.planner_version,
                            filter_expression_version=hit.trace.filter_expression_version,
                            component_ranks={
                                name: component.rank
                                for name, component in hit.trace.components.items()
                            },
                            fusion_score=float(hit.trace.fusion_score),
                            source_prior=float(hit.trace.source_prior),
                            rerank_score=(
                                None
                                if hit.trace.rerank_score is None
                                else float(hit.trace.rerank_score)
                            ),
                            degradation=hit.trace.degradation,
                        ),
                    )
                    for hit in outcome.hits
                )
                return RetrievalResponse(
                    build_id=outcome.build_id,
                    query=outcome.query,
                    plan=outcome.plan,
                    hits=hits,
                    degradation=outcome.degradation,
                )
            except RuntimeUnavailableError:
                raise
            except Exception:
                raise RuntimeUnavailableError("retrieval_failed") from None

    async def get_run(self, run_id: str) -> QueryResponse | None:
        return self._runs.get(run_id)

    async def resume(self, run_id: str) -> QueryResponse | None:
        if (existing := self._runs.get(run_id)) is not None:
            return existing
        if self.resume_state_loader is None:
            return None
        async with self._limiter:
            try:
                saved = await self.resume_state_loader(run_id)
                if not isinstance(saved, Mapping):
                    raise ValueError("saved run is not a mapping")
                idempotency_key = saved.get("idempotency_key")
                raw_ref = saved.get("original_request_ref") or saved.get("request_ref")
                if not isinstance(idempotency_key, str) or not idempotency_key.strip():
                    raise ValueError("saved run lacks an idempotency identity")
                if not isinstance(raw_ref, Mapping):
                    raise ValueError("saved run lacks its request reference")
                request_ref = RequestRef.model_validate_json(
                    json.dumps(raw_ref, ensure_ascii=False)
                )
                question = self.evidence_repository.get_request(request_ref)
                graph = self.graph_factory(10)
                state = await graph.ainvoke(
                    None,
                    {
                        "configurable": {
                            "thread_id": run_id,
                            "run_id": run_id,
                            "checkpoint_ns": f"run:{run_id}",
                            "idempotency_key": idempotency_key,
                            "resume_question": question,
                        },
                        "recursion_limit": 128,
                    },
                )
            except RuntimeUnavailableError:
                raise
            except Exception:
                raise RuntimeUnavailableError("checkpoint_unavailable") from None
            response = self._project_run(run_id, state)
            self._runs[run_id] = response
            return response

    async def get_evidence(self, evidence_id: str) -> Evidence | None:
        ref = self._evidence_refs.get(evidence_id)
        if ref is None:
            return None
        try:
            values = self.evidence_repository.get_many((ref,))
        except EvidenceRepositoryError:
            raise RuntimeUnavailableError("evidence_repository_error") from None
        if len(values) != 1 or type(values[0]) is not Evidence:
            raise RuntimeUnavailableError("invalid_runtime_contract")
        return values[0]

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.close_callback is not None:
            await self.close_callback()


class UnavailableRuntime:
    """Keep liveness available while every product operation fails closed."""

    def __init__(self, code: str = "runtime_initialization_failed") -> None:
        self.code = code

    async def readiness(self) -> ReadinessResponse:
        from trade_agent.api.models import ReadinessCheck

        return ReadinessResponse(
            ready=False,
            build_id=None,
            schema_fingerprint=None,
            checks={"runtime": ReadinessCheck(ok=False, code=self.code)},
        )

    async def query(self, request: QueryRequest) -> QueryResponse:
        del request
        raise RuntimeUnavailableError(self.code)

    async def retrieve(self, request: RetrieveRequest) -> RetrievalResponse:
        del request
        raise RuntimeUnavailableError(self.code)

    async def get_run(self, run_id: str) -> QueryResponse | None:
        del run_id
        raise RuntimeUnavailableError(self.code)

    async def resume(self, run_id: str) -> QueryResponse | None:
        del run_id
        raise RuntimeUnavailableError(self.code)

    async def get_evidence(self, evidence_id: str) -> Evidence | None:
        del evidence_id
        raise RuntimeUnavailableError(self.code)

    async def close(self) -> None:
        return None


@asynccontextmanager
async def production_runtime_lifespan():
    """Create service clients at startup, never at module import time."""
    runtime: ApiRuntime
    try:
        runtime = await build_production_runtime()
    except Exception:
        runtime = UnavailableRuntime()
    try:
        yield runtime
    finally:
        await runtime.close()


async def build_production_runtime() -> AgentRuntime:
    """Build the live runtime after verifying schema and immutable index state."""
    # Imports stay local so process liveness does not load models or open sockets.
    import hashlib
    import os
    from datetime import date

    from pymilvus import MilvusClient
    from sqlalchemy import create_engine, text

    from scripts.index_trade_corpus import _manager
    from trade_agent.agents.checkpoint import RedisCheckpointFactory, resume_run
    from trade_agent.agents.graph import GraphDependencies, build_trade_graph
    from trade_agent.agents.intent import IntentParser
    from trade_agent.agents.nodes import (
        ContractRetrievalBranch,
        ContractSqlBranch,
        FileEvidenceRepository,
        GraphBudgets,
    )
    from trade_agent.config.settings import Settings
    from trade_agent.db.registry import SchemaRegistry
    from trade_agent.db.session import database_url_from_environment
    from trade_agent.db.sql_renderer import SqlDataScope
    from trade_agent.index.builder import TradeIndexBundle, validate_sparse_build
    from trade_agent.index.milvus_store import TradeMilvusStore
    from trade_agent.retrieval.profiles import load_retrieval_profile
    from trade_agent.retrieval.service import RetrievalService

    settings = Settings.load()
    timeout = 5.0
    engine = create_engine(
        database_url_from_environment(role="query"),
        pool_pre_ping=True,
        pool_size=4,
        max_overflow=0,
        pool_timeout=timeout,
        connect_args={
            "connect_timeout": int(timeout),
            "read_timeout": timeout,
            "write_timeout": timeout,
        },
    )
    milvus_client = MilvusClient(
        uri=f"http://{settings.milvus.host}:{settings.milvus.port}",
        db_name=settings.milvus.database,
        timeout=timeout,
    )
    checkpointer = None
    try:
        registry = SchemaRegistry()
        with engine.connect() as connection:
            snapshot = registry.refresh(connection)
            bounds = connection.execute(
                text("SELECT MIN(trade_date), MAX(trade_date) FROM trade_records")
            ).one()
        if not isinstance(bounds[0], date) or not isinstance(bounds[1], date):
            raise ValueError("trade data scope is empty")

        bundle_path_value = os.environ.get("TRADE_AGENT_INDEX_BUNDLE", "").strip()
        if not bundle_path_value:
            raise ValueError("TRADE_AGENT_INDEX_BUNDLE is required")
        bundle_path = Path(bundle_path_value)
        if not bundle_path.is_absolute():
            bundle_path = (Path.cwd() / bundle_path).resolve()
        manager = _manager()
        store = TradeMilvusStore(client=milvus_client, embedding_manager=manager)
        bundle = TradeIndexBundle.load(bundle_path, milvus=store, embedding_manager=manager)
        retrieval = RetrievalService(
            build=bundle.build,
            bm25=bundle.bm25,
            milvus=store,
            embedding_manager=manager,
            profile=load_retrieval_profile(),
        )
        checkpointer = await RedisCheckpointFactory.create(settings.redis)
        repository_root = Path(
            os.environ.get("TRADE_AGENT_EVIDENCE_ROOT", "/tmp/trade-agent-evidence")
        ).resolve()
        repository = FileEvidenceRepository(repository_root)
        runtime_date = date.today()
        parser = IntentParser(as_of=runtime_date)
        sql_scope = SqlDataScope(
            dataset_id="trade-seed-v1",
            synthetic=True,
            start_date=bounds[0],
            end_date=bounds[1],
        )

        def connection_factory(*, timeout_seconds: float):
            if timeout_seconds > timeout:
                raise TimeoutError("requested timeout exceeds connection policy")
            return engine.connect()

        sql_branch = ContractSqlBranch(
            registry=snapshot,
            scope=sql_scope,
            connection_factory=connection_factory,
        )

        def graph_factory(top_k: int):
            return build_trade_graph(
                GraphDependencies(
                    intent_parser=parser,
                    evidence_repository=repository,
                    as_of=runtime_date,
                    sql_branch=sql_branch,
                    rag_branch=ContractRetrievalBranch(
                        service=retrieval,
                        published_manifest=bundle.build,
                        top_k=top_k,
                    ),
                    sql_identity_hmac_key=hashlib.sha256(
                        ("trade-agent-sql:" + settings.mysql.password).encode("utf-8")
                    ).digest(),
                    budgets=GraphBudgets(
                        max_steps=settings.limits.max_graph_steps,
                        max_retries=settings.limits.max_retries,
                        max_llm_calls=settings.limits.max_llm_calls,
                        node_timeout_seconds=timeout,
                        max_outstanding_calls=8,
                    ),
                ),
                checkpointer=checkpointer,
            )

        async def readiness() -> ReadinessResponse:
            from trade_agent.api.models import ReadinessCheck

            checks: dict[str, ReadinessCheck] = {}
            current_fingerprint = None
            try:
                with engine.connect() as connection:
                    current_fingerprint = registry.refresh(connection).fingerprint
                schema_matches = current_fingerprint == snapshot.fingerprint
                checks["mysql_schema"] = ReadinessCheck(
                    ok=schema_matches,
                    code=None if schema_matches else "schema_contract_mismatch",
                )
            except Exception:
                checks["mysql_schema"] = ReadinessCheck(ok=False, code="schema_contract_mismatch")
            try:
                validate_sparse_build(bundle.build, bundle.bm25)
                store.validate(bundle.descriptor.collection_contract)
                checks["milvus_index"] = ReadinessCheck(ok=True)
            except Exception:
                checks["milvus_index"] = ReadinessCheck(ok=False, code="index_contract_mismatch")
            try:
                await checkpointer._redis.ping()
                checks["redis_checkpoint"] = ReadinessCheck(ok=True)
            except Exception:
                checks["redis_checkpoint"] = ReadinessCheck(ok=False, code="checkpoint_unavailable")
            ready = all(item.ok for item in checks.values())
            return ReadinessResponse(
                ready=ready,
                build_id=bundle.build.build_id if ready else None,
                schema_fingerprint=current_fingerprint if ready else None,
                checks=checks,
            )

        async def close() -> None:
            if checkpointer is not None:
                await RedisCheckpointFactory.close(checkpointer)
            milvus_client.close()
            engine.dispose()

        return AgentRuntime(
            graph_factory=graph_factory,
            evidence_repository=repository,
            build_id=bundle.build.build_id,
            readiness_probe=readiness,
            retrieval_service=retrieval,
            intent_parser=parser,
            resume_state_loader=lambda run_id: resume_run(run_id, run_id),
            close_callback=close,
            max_outstanding_requests=8,
        )
    except BaseException:
        if checkpointer is not None:
            await RedisCheckpointFactory.close(checkpointer)
        milvus_client.close()
        engine.dispose()
        raise
