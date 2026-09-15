from __future__ import annotations

from contextlib import asynccontextmanager
import json
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from trade_agent.agents.nodes import FileEvidenceRepository
from trade_agent.agents.state import GuardProjection
from trade_agent.api.app import create_app
from trade_agent.api.dependencies import AgentRuntime, RuntimeUnavailableError
from trade_agent.api.models import (
    QueryResponse,
    ReadinessCheck,
    ReadinessResponse,
    RetrieveRequest,
)
from trade_agent.cli import main as cli_main
from trade_agent.config.settings import RedisSettings
from trade_agent.evidence.models import Claim
from tests.unit.test_evidence_validator import _rag


BUILD_ID = "build_0123456789abcdef0123456789abcdef"


def _guarded_response() -> QueryResponse:
    evidence = _rag(suffix="api")
    claim = Claim(
        claim_id="claim_" + "a" * 64,
        text=evidence.content,
        status="supported",
        evidence_ids=(evidence.evidence_id,),
        entity_id=evidence.entity_id,
        fact_type=evidence.fact_type,
        period_start=evidence.valid_from,
        period_end=evidence.valid_to,
        confidence=0.9,
    )
    return QueryResponse(
        run_id="run-contract",
        status="completed",
        build_id=BUILD_ID,
        answer=claim.text,
        claims=(claim,),
        evidence=(evidence,),
        conflicts=(),
        refusal_reason=None,
        errors=(),
        degraded_components=(),
    )


class FakeRuntime:
    def __init__(self, *, ready: bool = True) -> None:
        self.response = _guarded_response()
        self._ready = ready
        self.closed = False

    async def readiness(self) -> ReadinessResponse:
        index = ReadinessCheck(ok=self._ready, code=None if self._ready else "index_contract_missing")
        return ReadinessResponse(
            ready=self._ready,
            build_id=BUILD_ID if self._ready else None,
            schema_fingerprint="b" * 64 if self._ready else None,
            checks={
                "mysql_schema": ReadinessCheck(ok=True),
                "milvus_index": index,
                "redis_checkpoint": ReadinessCheck(ok=True),
            },
        )

    async def query(self, _request):
        return self.response

    async def retrieve(self, _request):
        raise RuntimeUnavailableError("retrieval_unavailable")

    async def get_run(self, run_id: str):
        if run_id != self.response.run_id:
            return None
        return self.response

    async def resume(self, run_id: str):
        return await self.get_run(run_id)

    async def get_evidence(self, evidence_id: str):
        if evidence_id != self.response.evidence[0].evidence_id:
            return None
        return self.response.evidence[0]

    async def close(self) -> None:
        self.closed = True


def _client(runtime: FakeRuntime) -> TestClient:
    @asynccontextmanager
    async def lifespan_factory():
        try:
            yield runtime
        finally:
            await runtime.close()

    return TestClient(create_app(runtime_lifespan=lifespan_factory))


def test_health_is_process_liveness_even_when_dependencies_are_unready() -> None:
    runtime = FakeRuntime(ready=False)
    with _client(runtime) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}
    assert runtime.closed is True


def test_ready_is_false_when_milvus_contract_missing() -> None:
    with _client(FakeRuntime(ready=False)) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert response.json()["checks"]["milvus_index"] == {
        "ok": False,
        "code": "index_contract_missing",
    }


def test_query_response_separates_claims_and_evidence() -> None:
    with _client(FakeRuntime()) as client:
        response = client.post(
            "/v1/query", json={"question": "ABC 是否值得跟进", "top_k": 10}
        )

    assert response.status_code == 200
    body = response.json()
    assert all(
        claim["evidence_ids"]
        for claim in body["claims"]
        if claim["status"] == "supported"
    )
    assert {item["evidence_id"] for item in body["evidence"]} == {
        evidence_id for claim in body["claims"] for evidence_id in claim["evidence_ids"]
    }
    assert body["answer"] == "\n".join(claim["text"] for claim in body["claims"])
    assert body["build_id"] == BUILD_ID
    assert "draft" not in body


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "ABC 是否值得跟进", "top_k": 10, "stream": True},
        {"question": " ", "top_k": 10},
        {"question": "q", "top_k": 0},
        {"question": "q", "top_k": 101},
        {"question": "q", "top_k": True},
        {"question": "q", "top_k": 10, "explicit_filters": {"unknown": "x"}},
    ],
)
def test_query_rejects_extra_unbounded_or_coerced_input(payload: dict[str, object]) -> None:
    with _client(FakeRuntime()) as client:
        response = client.post("/v1/query", json=payload)

    assert response.status_code == 422


def test_api_refuses_to_serialize_an_untyped_runtime_draft() -> None:
    runtime = FakeRuntime()
    runtime.response = {"answer": "UNGUARDED PROVIDER DRAFT"}  # type: ignore[assignment]
    with _client(runtime) as client:
        response = client.post("/v1/query", json={"question": "ABC", "top_k": 10})

    assert response.status_code == 502
    assert "UNGUARDED PROVIDER DRAFT" not in response.text
    assert response.json() == {"detail": {"code": "invalid_runtime_contract"}}


def test_query_contract_rejects_uncited_evidence_payloads() -> None:
    response = _guarded_response()
    uncited = _rag(suffix="uncited")

    with pytest.raises(ValidationError, match="exactly the cited set"):
        QueryResponse.model_validate(
            {
                **response.model_dump(mode="python"),
                "evidence": (*response.evidence, uncited),
            }
        )


def test_run_resume_and_evidence_routes_use_the_typed_runtime_boundary() -> None:
    runtime = FakeRuntime()
    evidence_id = runtime.response.evidence[0].evidence_id
    with _client(runtime) as client:
        run = client.get("/v1/runs/run-contract")
        resumed = client.post("/v1/runs/run-contract/resume", json={})
        evidence = client.get(f"/v1/evidence/{evidence_id}")
        missing = client.get("/v1/runs/run-missing")

    assert run.status_code == resumed.status_code == evidence.status_code == 200
    assert run.json() == resumed.json() == runtime.response.model_dump(mode="json")
    assert evidence.json() == runtime.response.evidence[0].model_dump(mode="json")
    assert missing.status_code == 404


def test_evidence_path_rejects_non_hex_identity_before_repository_lookup() -> None:
    invalid = "rag_" + "z" * 64
    with _client(FakeRuntime()) as client:
        response = client.get(f"/v1/evidence/{invalid}")

    assert response.status_code == 422


def test_retrieve_failure_is_machine_readable() -> None:
    with _client(FakeRuntime()) as client:
        response = client.post("/v1/retrieve", json={"question": "ABC 官网", "top_k": 5})

    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "retrieval_unavailable"}}


def test_cli_help_exposes_the_complete_product_command_surface(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        cli_main(["--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    for command in (
        "bootstrap-demo",
        "db",
        "ingest",
        "index",
        "query",
        "eval",
        "smoke",
        "verify-report",
    ):
        assert command in help_text


def test_eval_cli_delegates_argument_validation_to_trade_runner(capsys) -> None:
    assert cli_main([
        "eval",
        "--dataset",
        "unused.jsonl",
        "--output",
        "unused-output",
        "--max-cases",
        "0",
    ]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "max-cases must be positive" in captured.err


def test_verify_report_cli_delegates_argument_validation_to_report_verifier(
    tmp_path, capsys
) -> None:
    assert cli_main(["verify-report", "--latest", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--latest requires --kind" in captured.err


def test_eval_cli_help_comes_from_trade_runner(capsys) -> None:
    assert cli_main(["eval", "--help"]) == 0
    help_text = capsys.readouterr().out
    assert "--dataset" in help_text
    assert "--consume-holdout" in help_text


def test_verify_report_cli_help_comes_from_report_verifier(capsys) -> None:
    assert cli_main(["verify-report", "--help"]) == 0
    help_text = capsys.readouterr().out
    assert "--latest" in help_text
    assert "--kind" in help_text


def test_bootstrap_demo_cli_composes_the_existing_generator(tmp_path, capsys) -> None:
    output = tmp_path / "demo"

    assert cli_main(["bootstrap-demo", "--output", str(output), "--clean"]) == 0

    captured = capsys.readouterr()
    assert '"synthetic_only": true' in captured.out
    assert (output / json.loads(captured.out)["manifest"]).is_file()


def test_product_server_exports_the_strict_fastapi_app() -> None:
    from trade_agent.server import app as server_app

    paths = set(server_app.openapi()["paths"])
    assert paths == {
        "/health",
        "/ready",
        "/v1/query",
        "/v1/retrieve",
        "/v1/runs/{run_id}",
        "/v1/runs/{run_id}/resume",
        "/v1/evidence/{evidence_id}",
    }
    assert all("stream" not in path for path in paths)


@pytest.mark.asyncio
async def test_api_checkpoint_client_has_finite_connect_and_rpc_timeouts(monkeypatch) -> None:
    from trade_agent.agents import checkpoint

    captured = {}

    class RedisClient:
        async def ping(self):
            return True

        async def aclose(self):
            return None

    class Saver:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            self._redis = RedisClient()

        async def asetup(self):
            return None

    monkeypatch.setattr(checkpoint, "_MinimalAsyncRedisSaver", Saver)

    saver = await checkpoint.RedisCheckpointFactory.create(
        RedisSettings(host="127.0.0.1", port=6389)
    )
    try:
        assert captured["connection_args"]["socket_connect_timeout"] == 5.0
        assert captured["connection_args"]["socket_timeout"] == 5.0
        assert captured["connection_args"]["max_connections"] == 16
    finally:
        await checkpoint.RedisCheckpointFactory.close(saver)


@pytest.mark.asyncio
async def test_resume_uses_only_the_stored_request_scope(tmp_path) -> None:
    repository = FileEvidenceRepository((tmp_path / "evidence").resolve())
    request_ref = repository.put_request("Acme 是否值得跟进")
    guard_ref = repository.put_guard(
        GuardProjection(
            accepted=False,
            claims=(),
            refusal_reason="evidence_retryable",
        )
    )
    captured = {}

    class ResumeGraph:
        async def ainvoke(self, graph_input, config):
            captured["graph_input"] = graph_input
            captured["config"] = config
            return {
                "guard_ref": guard_ref.model_dump(mode="json"),
                "evidence_refs": [],
                "conflicts": [],
                "errors": [],
                "branch_reports": [],
            }

    async def load_state(run_id: str):
        assert run_id == "durable-run"
        return {
            "idempotency_key": "durable-key",
            "original_request_ref": request_ref.model_dump(mode="json"),
            "request_top_k": 37,
        }

    async def ready():
        return ReadinessResponse(
            ready=True,
            build_id=BUILD_ID,
            schema_fingerprint="e" * 64,
            checks={"runtime": ReadinessCheck(ok=True)},
        )

    def graph_factory(top_k):
        captured["top_k"] = top_k
        return ResumeGraph()

    runtime = AgentRuntime(
        graph_factory=graph_factory,
        evidence_repository=repository,
        build_id=BUILD_ID,
        readiness_probe=ready,
        resume_state_loader=load_state,
    )

    response = await runtime.resume("durable-run")

    assert response is not None and response.refusal_reason == "evidence_retryable"
    assert captured["graph_input"] is None
    assert captured["top_k"] == 37
    assert captured["config"]["configurable"]["resume_question"] == "Acme 是否值得跟进"
    assert captured["config"]["configurable"]["idempotency_key"] == "durable-key"


@pytest.mark.asyncio
async def test_retrieve_uses_runtime_candidate_budget_beyond_response_top_k(tmp_path) -> None:
    from types import SimpleNamespace

    from trade_agent.agents.intent import QueryIntent as BusinessIntent
    from trade_agent.retrieval.filters import RetrievalFilter
    from trade_agent.retrieval.planner import RetrievalPlan

    repository = FileEvidenceRepository((tmp_path / "evidence").resolve())
    captured: dict[str, object] = {}

    async def ready():
        return ReadinessResponse(
            ready=True,
            build_id=BUILD_ID,
            schema_fingerprint="e" * 64,
            checks={"runtime": ReadinessCheck(ok=True)},
        )

    class Parser:
        def parse(self, question, explicit_filters):
            captured["explicit_filters"] = explicit_filters
            return BusinessIntent(
                question=question,
                kind="external_intelligence",
                need_external_intel=True,
            )

    class Retrieval:
        def search(self, intent, *, top_k, candidate_limit, transport_timeout_seconds):
            captured["intent"] = intent
            captured["top_k"] = top_k
            captured["candidate_limit"] = candidate_limit
            captured["transport_timeout_seconds"] = transport_timeout_seconds
            return SimpleNamespace(
                build_id=BUILD_ID,
                query=intent.query,
                plan=RetrievalPlan(
                    query=intent.query,
                    filter=RetrievalFilter(),
                    extraction_confidence=1.0,
                    applied_constraints=(),
                    unapplied_constraints=(),
                ),
                hits=(),
                degradation=(),
            )

    runtime = AgentRuntime(
        graph_factory=lambda _top_k: None,
        evidence_repository=repository,
        build_id=BUILD_ID,
        readiness_probe=ready,
        retrieval_service=Retrieval(),
        intent_parser=Parser(),
        max_retrieval_candidates=10,
    )

    response = await runtime.retrieve(
        RetrieveRequest(question="HS850440 charger procurement", top_k=3)
    )

    assert response.ranking_only is True
    assert captured["top_k"] == 3
    assert captured["candidate_limit"] == 10
    assert captured["transport_timeout_seconds"] == 5.0
    assert captured["explicit_filters"] == RetrievalFilter()


def test_resume_maps_a_clean_missing_checkpoint_to_not_found(tmp_path) -> None:
    repository = FileEvidenceRepository((tmp_path / "evidence").resolve())

    async def load_missing(_run_id: str):
        return None

    async def ready():
        return ReadinessResponse(
            ready=True,
            build_id=BUILD_ID,
            schema_fingerprint="e" * 64,
            checks={"runtime": ReadinessCheck(ok=True)},
        )

    runtime = AgentRuntime(
        graph_factory=lambda _top_k: None,
        evidence_repository=repository,
        build_id=BUILD_ID,
        readiness_probe=ready,
        resume_state_loader=load_missing,
    )

    with _client(runtime) as client:
        response = client.post("/v1/runs/missing-run/resume", json={})

    assert response.status_code == 404
    assert response.json() == {"detail": {"code": "run_not_found"}}


def test_resume_keeps_checkpoint_transport_failure_unavailable(tmp_path) -> None:
    repository = FileEvidenceRepository((tmp_path / "evidence").resolve())

    async def load_unavailable(_run_id: str):
        raise RuntimeUnavailableError("checkpoint_unavailable")

    async def ready():
        return ReadinessResponse(
            ready=True,
            build_id=BUILD_ID,
            schema_fingerprint="e" * 64,
            checks={"runtime": ReadinessCheck(ok=True)},
        )

    runtime = AgentRuntime(
        graph_factory=lambda _top_k: None,
        evidence_repository=repository,
        build_id=BUILD_ID,
        readiness_probe=ready,
        resume_state_loader=load_unavailable,
    )

    with _client(runtime) as client:
        response = client.post("/v1/runs/unavailable-run/resume", json={})

    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "checkpoint_unavailable"}}
