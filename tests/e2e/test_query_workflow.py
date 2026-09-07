from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date

from fastapi.testclient import TestClient

from trade_agent.agents.graph import GraphDependencies, build_trade_graph
from trade_agent.agents.intent import IntentParser
from trade_agent.agents.nodes import FileEvidenceRepository, GraphBudgets
from trade_agent.api.app import create_app
from trade_agent.api.dependencies import AgentRuntime
from trade_agent.api.models import ReadinessCheck, ReadinessResponse
from trade_agent.evidence.models import Evidence
from tests.integration.test_graph import StaticEvidenceBranch
from tests.unit.test_evidence_validator import _rag, _sql
from tests.unit.test_retrieval_service import _service


AS_OF = date(2026, 9, 4)
BUILD_ID = "build_0123456789abcdef0123456789abcdef"


def _runtime(tmp_path, *, rag_error: Exception | None = None) -> AgentRuntime:
    repository = FileEvidenceRepository((tmp_path / "evidence").resolve())
    sql = StaticEvidenceBranch((_sql(start=date(2025, 9, 4), scope="lead"),))
    rag_values: tuple[Evidence, ...] = (
        _rag(suffix="api-official"),
        _rag(suffix="api-news", source_type="industry_news"),
    )
    rag = StaticEvidenceBranch(rag_values, error=rag_error)

    def graph_factory(_top_k: int):
        return build_trade_graph(
            GraphDependencies(
                intent_parser=IntentParser(as_of=AS_OF),
                sql_branch=sql,
                rag_branch=rag,
                evidence_repository=repository,
                entity_bindings={"company:acme": 17},
                sql_identity_hmac_key=b"e2e-api-hmac-key",
                budgets=GraphBudgets(node_timeout_seconds=0.2),
                as_of=AS_OF,
            )
        )

    async def ready() -> ReadinessResponse:
        return ReadinessResponse(
            ready=True,
            build_id=BUILD_ID,
            schema_fingerprint="c" * 64,
            checks={
                "mysql_schema": ReadinessCheck(ok=True),
                "milvus_index": ReadinessCheck(ok=True),
                "redis_checkpoint": ReadinessCheck(ok=True),
            },
        )

    return AgentRuntime(
        graph_factory=graph_factory,
        evidence_repository=repository,
        build_id=BUILD_ID,
        readiness_probe=ready,
        max_outstanding_requests=2,
    )


def _client(runtime: AgentRuntime) -> TestClient:
    @asynccontextmanager
    async def lifespan_factory():
        try:
            yield runtime
        finally:
            await runtime.close()

    return TestClient(create_app(runtime_lifespan=lifespan_factory))


def test_mixed_query_returns_only_guarded_sql_and_rag_claims(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    with _client(runtime) as client:
        response = client.post(
            "/v1/query",
            json={
                "question": "Acme 是否值得跟进",
                "top_k": 10,
                "explicit_filters": {"entity_ids": ["company:acme"]},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert {item["locator"]["branch"] for item in body["evidence"]} == {"sql", "rag"}
    known_ids = {item["evidence_id"] for item in body["evidence"]}
    assert body["claims"]
    assert all(claim["status"] == "supported" for claim in body["claims"])
    assert all(set(claim["evidence_ids"]) <= known_ids for claim in body["claims"])
    assert body["answer"] == "\n".join(claim["text"] for claim in body["claims"])


def test_required_branch_failure_returns_machine_readable_refusal_without_draft(tmp_path) -> None:
    runtime = _runtime(tmp_path, rag_error=TimeoutError("provider-secret"))
    with _client(runtime) as client:
        response = client.post(
            "/v1/query",
            json={
                "question": "Acme 是否值得跟进",
                "top_k": 10,
                "explicit_filters": {"entity_ids": ["company:acme"]},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "refused"
    assert body["answer"] is None
    assert body["claims"] == []
    assert body["refusal_reason"] == "evidence_retryable"
    assert all(set(error) == {"code", "node", "retryable"} for error in body["errors"])
    assert "provider-secret" not in response.text


def test_evidence_endpoint_reloads_the_exact_stored_contract(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    with _client(runtime) as client:
        query = client.post(
            "/v1/query",
            json={
                "question": "Acme 是否值得跟进",
                "top_k": 10,
                "explicit_filters": {"entity_ids": ["company:acme"]},
            },
        )
        expected = query.json()["evidence"][0]
        response = client.get(f"/v1/evidence/{expected['evidence_id']}")

    assert response.status_code == 200
    assert response.json() == expected


def test_retrieve_exposes_audit_trace_without_a_truth_status(tmp_path) -> None:
    service, _manifest, _store = _service(tmp_path)
    repository = FileEvidenceRepository((tmp_path / "retrieve-evidence").resolve())

    async def ready() -> ReadinessResponse:
        return ReadinessResponse(
            ready=True,
            build_id=service.build_id,
            schema_fingerprint="d" * 64,
            checks={"runtime": ReadinessCheck(ok=True)},
        )

    runtime = AgentRuntime(
        graph_factory=lambda _top_k: None,
        evidence_repository=repository,
        build_id=service.build_id,
        readiness_probe=ready,
        retrieval_service=service,
        intent_parser=IntentParser(as_of=AS_OF),
    )
    with _client(runtime) as client:
        response = client.post(
            "/v1/retrieve",
            json={"question": "Verified synthetic trade evidence 官网", "top_k": 3},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["ranking_only"] is True
    assert body["hits"]
    assert all("source_prior" in hit["trace"] for hit in body["hits"])
    assert all("status" not in hit for hit in body["hits"])
