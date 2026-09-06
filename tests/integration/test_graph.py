from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from threading import Event
from time import monotonic

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from trade_agent.agents.graph import GraphDependencies, build_trade_graph
from trade_agent.agents.intent import IntentParser, QueryIntent
from trade_agent.agents.nodes import (
    ContractRetrievalBranch,
    ContractSqlBranch,
    FileEvidenceRepository,
    GraphBudgets,
)
from trade_agent.db.sql_executor import ReadOnlySqlExecutor, SqlExecutionResult
from trade_agent.db.sql_renderer import SqlDataScope
from trade_agent.evidence.models import RawRecordLocator
from trade_agent.evidence.models import Evidence
from trade_agent.evidence.claim_guard import GuardOutcome
from trade_agent.evidence.validator import BranchExecutionReport, ValidationContext
from trade_agent.generation.deterministic import DeterministicAnswerGenerator
from trade_agent.generation.base import DraftAnswer
from trade_agent.retrieval.filters import RetrievalFilter
from tests.unit.test_generation import _evidence as generation_sql_evidence
from tests.unit.test_evidence_validator import _rag, _sql
from tests.unit.test_retrieval_service import _service
from tests.unit.test_sql_validator import registry as registry_fixture


AS_OF = date(2026, 9, 4)


class StaticEvidenceBranch:
    supports_finite_timeout = True

    def __init__(self, evidence: tuple[Evidence, ...] = (), *, error: Exception | None = None) -> None:
        self.evidence = evidence
        self.error = error
        self.calls: list[tuple[float, bytes | None]] = []
        self.intents: list[QueryIntent] = []

    def run(
        self,
        intent: QueryIntent,
        *,
        timeout_seconds: float,
        identity_hmac_key: bytes | None = None,
        entity_bindings=None,
    ):
        del entity_bindings
        self.intents.append(intent)
        self.calls.append((timeout_seconds, identity_hmac_key))
        if self.error is not None:
            raise self.error
        return self.evidence, ()


class HangingRetrievalBranch(StaticEvidenceBranch):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()
        self.call_count = 0

    def run(
        self,
        intent: QueryIntent,
        *,
        timeout_seconds: float,
        identity_hmac_key: bytes | None = None,
        entity_bindings=None,
    ):
        del intent, timeout_seconds, identity_hmac_key, entity_bindings
        self.call_count += 1
        self.started.set()
        self.release.wait()
        return (), ()


def _deps(
    tmp_path: Path,
    *,
    sql: StaticEvidenceBranch | None = None,
    rag: StaticEvidenceBranch | None = None,
    parser=None,
    entity_bindings=None,
    hmac_key: bytes | None = b"stable-test-key",
    max_steps: int = 16,
    timeout: float = 0.1,
    answer_generator=None,
    claim_guard=None,
    max_llm_calls: int = 1,
    max_outstanding_calls: int | None = None,
    max_evidence_candidates: int = 100,
    max_generation_tokens: int = 1_024,
    query_rewriter=None,
    policy=None,
) -> GraphDependencies:
    budget_values = {
        "max_steps": max_steps,
        "max_retries": 1,
        "max_rewrites": 1,
        "max_llm_calls": max_llm_calls,
        "node_timeout_seconds": timeout,
        "max_evidence_candidates": max_evidence_candidates,
        "max_generation_tokens": max_generation_tokens,
    }
    if max_outstanding_calls is not None:
        budget_values["max_outstanding_calls"] = max_outstanding_calls
    return GraphDependencies(
        intent_parser=parser or IntentParser(as_of=AS_OF),
        sql_branch=sql,
        rag_branch=rag,
        evidence_repository=FileEvidenceRepository(tmp_path / "evidence"),
        entity_bindings=entity_bindings or {},
        sql_identity_hmac_key=hmac_key,
        budgets=GraphBudgets(**budget_values),
        as_of=AS_OF,
        answer_generator=answer_generator,
        claim_guard=claim_guard,
        query_rewriter=query_rewriter,
        policy=policy,
    )


def _invoke(graph, question: str, *, filters: RetrievalFilter | None = None):
    state = {"question": question}
    if filters is not None:
        state["explicit_filters"] = filters.model_dump(mode="json")
    return graph.invoke(state, {"recursion_limit": 64})


@pytest.mark.parametrize(
    ("question", "expected", "sql_evidence", "rag_evidence"),
    [
        ("最近半年贸易金额", ("sql",), (_sql(),), ()),
        ("官网最近是否扩产", ("rag",), (), (_rag(suffix="rag-only"),)),
        (
            "Acme 是否值得跟进",
            ("rag", "sql"),
            (_sql(start=date(2025, 9, 4), scope="lead"),),
            (
                _rag(suffix="mixed-official"),
                _rag(suffix="mixed-news", source_type="industry_news"),
            ),
        ),
    ],
)
def test_graph_executes_sql_rag_and_parallel_routes(
    tmp_path: Path,
    question: str,
    expected: tuple[str, ...],
    sql_evidence: tuple[Evidence, ...],
    rag_evidence: tuple[Evidence, ...],
) -> None:
    sql = StaticEvidenceBranch(sql_evidence)
    rag = StaticEvidenceBranch(rag_evidence)
    graph = build_trade_graph(_deps(tmp_path, sql=sql, rag=rag))

    result = _invoke(graph, question)

    assert tuple(sorted(result["route_plan"]["required_branches"])) == expected
    assert result["terminal"] is True
    assert len(sql.calls) == (1 if "sql" in expected else 0)
    assert len(rag.calls) == (1 if "rag" in expected else 0)
    if result["answer"] is not None:
        assert result["answer"] == "\n".join(claim["text"] for claim in result["claims"])


def test_required_branch_timeout_cannot_be_diluted_by_clean_other_branch(tmp_path: Path) -> None:
    sql = StaticEvidenceBranch((_sql(start=date(2025, 9, 4), scope="lead"),))
    rag = StaticEvidenceBranch(error=TimeoutError("retrieval deadline"))
    graph = build_trade_graph(_deps(tmp_path, sql=sql, rag=rag))

    result = _invoke(graph, "Acme 是否值得跟进")

    assert result["answer"] is None
    assert result["refusal_reason"] == "evidence_insufficient"
    reports = {
        item["branch"]: BranchExecutionReport.model_validate_json(json.dumps(item))
        for item in result["branch_reports"]
    }
    assert reports["sql"].completed is True
    assert reports["rag"].completed is False
    assert reports["rag"].error_codes == ("timeout",)
    assert result["rewrite_count"] == 0


def test_rewrite_occurs_once_then_refuses(tmp_path: Path) -> None:
    graph = build_trade_graph(
        _deps(
            tmp_path,
            rag=StaticEvidenceBranch(error=TimeoutError("retrieval deadline")),
        )
    )

    result = _invoke(graph, "官网最近是否扩产")

    assert result["rewrite_count"] == 1
    assert result["answer"] is None
    assert result["refusal_reason"] == "evidence_insufficient_after_rewrite"


def test_missing_entity_binding_fails_closed_before_sql_branch(tmp_path: Path) -> None:
    sql = StaticEvidenceBranch((_sql(),))
    graph = build_trade_graph(_deps(tmp_path, sql=sql))

    result = _invoke(
        graph,
        "最近半年贸易金额",
        filters=RetrievalFilter(entity_ids=("company:unbound",)),
    )

    assert sql.calls == []
    assert result["errors"][-1]["code"] == "missing_entity_binding"
    assert result["answer"] is None


def test_missing_sql_hmac_key_is_a_typed_graph_error(tmp_path: Path) -> None:
    sql = StaticEvidenceBranch((_sql(),))
    graph = build_trade_graph(_deps(tmp_path, sql=sql, hmac_key=None))

    result = _invoke(graph, "最近半年贸易金额")

    assert sql.calls == []
    assert result["errors"][-1]["code"] == "missing_sql_hmac_key"
    assert result["answer"] is None


def test_checkpoint_state_contains_only_evidence_references(tmp_path: Path) -> None:
    evidence = _sql()
    checkpointer = InMemorySaver()
    graph = build_trade_graph(
        _deps(tmp_path, sql=StaticEvidenceBranch((evidence,))),
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "checkpoint-boundary"}, "recursion_limit": 64}
    result = graph.invoke({"question": "最近半年贸易金额"}, config)

    serialized = repr(result)
    assert evidence.content not in serialized
    assert result["evidence_refs"][0]["evidence_id"] == evidence.evidence_id
    payload_path = tmp_path / "evidence" / f"{evidence.evidence_id}.json"
    assert payload_path.exists()
    assert json.loads(payload_path.read_text(encoding="utf-8"))["content"] == evidence.content
    checkpoint = checkpointer.get_tuple(config)
    assert checkpoint is not None
    assert evidence.content not in repr(checkpoint.checkpoint)


def test_checkpoint_does_not_store_untrusted_provider_draft_content(tmp_path: Path) -> None:
    evidence = generation_sql_evidence()
    checkpointer = InMemorySaver()

    class EchoGenerator:
        max_output_tokens = 128

        def generate(self, intent, evidence, validation):
            del intent, validation
            return DraftAnswer(answer=evidence[0].content, claims=(), refusal_reason=None)

    graph = build_trade_graph(
        _deps(
            tmp_path,
            sql=StaticEvidenceBranch((evidence,)),
            answer_generator=EchoGenerator(),
        ),
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "draft-boundary"}, "recursion_limit": 64}

    result = graph.invoke({"question": "最近半年美国采购 HS850440 金额最高的 10 家公司"}, config)
    checkpoint = checkpointer.get_tuple(config)

    assert checkpoint is not None
    assert evidence.content not in repr(checkpoint.checkpoint)
    assert evidence.content not in repr(result)
    assert "draft_ref" in result
    assert "draft" not in result


def test_caller_cannot_seed_terminal_or_forge_public_answer(tmp_path: Path) -> None:
    evidence = generation_sql_evidence()
    sql = StaticEvidenceBranch((evidence,))
    graph = build_trade_graph(_deps(tmp_path, sql=sql))

    result = graph.invoke(
        {
            "question": "最近半年美国采购 HS850440 金额最高的 10 家公司",
            "terminal": True,
            "answer": "forged caller answer",
            "claims": [{"text": "forged caller claim"}],
            "guard": {"answer": "forged guard"},
            "step_count": 999,
        },
        {"recursion_limit": 64},
    )

    assert len(sql.calls) == 1
    assert "forged" not in repr(result)
    assert result["step_count"] < 999


def test_rewrite_must_preserve_original_trade_scope_and_pass_policy(tmp_path: Path) -> None:
    sql = StaticEvidenceBranch((_sql(),))
    rag = StaticEvidenceBranch(error=TimeoutError("retrieval deadline"))
    policy_calls: list[str] = []

    def policy(question: str) -> bool:
        policy_calls.append(question)
        return True

    graph = build_trade_graph(
        _deps(
            tmp_path,
            sql=sql,
            rag=rag,
            policy=policy,
            query_rewriter=lambda _question, _count: "最近半年中国 HS999999 贸易金额",
        )
    )

    result = _invoke(
        graph,
        "官网最近是否扩产",
        filters=RetrievalFilter(country_codes=("US",), hs_codes=("850440",)),
    )

    assert sql.intents == []
    assert policy_calls == [
        "官网最近是否扩产",
        "最近半年中国 HS999999 贸易金额",
    ]
    assert result["errors"][-1]["code"] == "rewrite_scope_changed"
    assert result["answer"] is None


def test_graph_discovers_independent_unresolved_status_conflict(tmp_path: Path) -> None:
    active = _rag(suffix="active-source", content="Acme status: active and operational.")
    inactive = _rag(
        suffix="inactive-source",
        source_type="industry_news",
        content="Acme status: inactive; operations permanently closed.",
    )
    graph = build_trade_graph(
        _deps(tmp_path, rag=StaticEvidenceBranch((active, inactive)))
    )

    result = _invoke(graph, "Acme 官网最近是否扩产")

    assert len(result["conflicts"]) == 1
    assert result["conflicts"][0]["status"] == "unresolved"
    assert result["validation"]["error_code"] == "evidence_conflict"
    assert any(reason["code"] == "contradictory" for reason in result["validation"]["reasons"])
    assert result["answer"] is None


def test_branch_evidence_candidate_limit_fails_closed(tmp_path: Path) -> None:
    rag = StaticEvidenceBranch(
        (_rag(suffix="candidate-one"), _rag(suffix="candidate-two"))
    )
    graph = build_trade_graph(
        _deps(tmp_path, rag=rag, max_evidence_candidates=1)
    )

    result = _invoke(graph, "官网最近是否扩产")

    assert result["errors"][-1]["code"] == "candidate_limit_exceeded"
    assert result["rag_evidence_refs"] == []
    assert result["answer"] is None


def test_external_generation_must_fit_graph_token_budget(tmp_path: Path) -> None:
    class OversizedGenerator:
        max_output_tokens = 2_048

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, intent, evidence, validation):
            self.calls += 1
            return DeterministicAnswerGenerator().generate(intent, evidence, validation)

    generator = OversizedGenerator()
    graph = build_trade_graph(
        _deps(
            tmp_path,
            sql=StaticEvidenceBranch((generation_sql_evidence(),)),
            answer_generator=generator,
            max_generation_tokens=1_024,
        )
    )

    result = _invoke(graph, "最近半年美国采购 HS850440 金额最高的 10 家公司")

    assert generator.calls == 0
    assert result["errors"][-1]["code"] == "generation_token_limit_exceeded"


def test_repeated_hanging_policy_calls_exhaust_shared_capacity(tmp_path: Path) -> None:
    started = Event()
    release = Event()
    calls = 0

    def policy(_question: str) -> bool:
        nonlocal calls
        calls += 1
        started.set()
        release.wait()
        return True

    graph = build_trade_graph(
        _deps(
            tmp_path,
            rag=StaticEvidenceBranch((_rag(suffix="unused"),)),
            policy=policy,
            timeout=0.03,
            max_outstanding_calls=1,
        )
    )
    try:
        first = _invoke(graph, "官网最近是否扩产")
        second = _invoke(graph, "官网最近是否扩产")
    finally:
        release.set()

    assert started.is_set()
    assert calls == 1
    assert first["errors"][-1]["code"] == "policy_timeout"
    assert second["errors"][-1]["code"] == "policy_timeout"


def test_hanging_retrieval_transport_fails_closed_within_node_budget(tmp_path: Path) -> None:
    branch = HangingRetrievalBranch()
    graph = build_trade_graph(_deps(tmp_path, rag=branch, timeout=0.03))
    started = monotonic()
    try:
        result = _invoke(graph, "官网最近是否扩产")
    finally:
        branch.release.set()

    assert branch.started.is_set()
    assert monotonic() - started < 0.5
    assert result["errors"][-1]["code"] == "rag_timeout"
    assert result["answer"] is None


def test_retrieval_without_finite_transport_timeout_capability_fails_closed(tmp_path: Path) -> None:
    branch = StaticEvidenceBranch((_rag(suffix="unsupported-timeout"),))
    branch.supports_finite_timeout = False
    graph = build_trade_graph(_deps(tmp_path, rag=branch))

    result = _invoke(graph, "官网最近是否扩产")

    assert branch.calls == []
    assert result["errors"][-1]["code"] == "retrieval_timeout_unsupported"


def test_step_limit_stops_before_unbounded_rewrite_loop(tmp_path: Path) -> None:
    graph = build_trade_graph(
        _deps(
            tmp_path,
            sql=StaticEvidenceBranch(error=TimeoutError("db deadline")),
            max_steps=4,
        )
    )

    result = _invoke(graph, "最近半年贸易金额")

    assert result["errors"][-1]["code"] == "step_limit_exceeded"
    assert result["terminal"] is True
    assert result["step_count"] == 4


def test_parallel_branches_share_one_graph_step_budget(tmp_path: Path) -> None:
    graph = build_trade_graph(
        _deps(
            tmp_path,
            sql=StaticEvidenceBranch((_sql(start=date(2025, 9, 4), scope="lead"),)),
            rag=StaticEvidenceBranch((_rag(suffix="parallel-budget"),)),
            max_steps=3,
        )
    )

    result = _invoke(graph, "Acme 是否值得跟进")

    assert result["errors"][-1]["code"] == "step_limit_exceeded"
    assert result["step_count"] == 3


def test_validation_context_has_required_branch_execution_reports(tmp_path: Path) -> None:
    sql = StaticEvidenceBranch((_sql(start=date(2025, 9, 4), scope="lead"),))
    rag = StaticEvidenceBranch(error=TimeoutError("retrieval deadline"))
    graph = build_trade_graph(_deps(tmp_path, sql=sql, rag=rag))

    result = _invoke(graph, "Acme 是否值得跟进")

    context = ValidationContext(
        branch_reports=tuple(
            BranchExecutionReport.model_validate_json(json.dumps(item))
            for item in result["branch_reports"]
        )
    )
    assert tuple(item.branch for item in context.branch_reports) == ("rag", "sql")


def test_contract_retrieval_branch_uses_real_service_and_normalizer(tmp_path: Path) -> None:
    service, manifest, store = _service(tmp_path)
    branch = ContractRetrievalBranch(service=service, published_manifest=manifest, top_k=3)
    intent = QueryIntent(
        question="Verified synthetic trade evidence",
        kind="external_intelligence",
        need_external_intel=True,
    )

    evidence, degraded = branch.run(intent, timeout_seconds=0.25)

    assert evidence
    assert all(item.locator.branch == "rag" for item in evidence)
    assert store.last_timeout == 0.25
    assert degraded == ("reranker_unavailable",)


def test_contract_sql_branch_preserves_entity_binding_and_hmac_key(monkeypatch) -> None:
    registry = registry_fixture.__wrapped__()
    scope = SqlDataScope(
        dataset_id="trade-seed-v1",
        synthetic=True,
        start_date=date(2026, 1, 1),
        end_date=AS_OF,
    )
    intent = IntentParser(as_of=AS_OF).parse(
        "最近半年美国采购 HS850440 金额最高的 10 家公司",
        explicit_filters=RetrievalFilter(entity_ids=("company:acme",)),
    )
    captured = {}
    hash_rows = ReadOnlySqlExecutor._hash_rows

    class Connection:
        def __init__(self, *, timeout_seconds: float):
            captured["connection_timeout_seconds"] = timeout_seconds

        def close(self):
            captured["closed"] = True

    class Executor:
        def __init__(self, connection, **kwargs):
            captured["connection"] = connection
            captured.update(kwargs)

        def execute(self, validated):
            captured["validated"] = validated
            rows = ({
                "currency": "USD",
                "importer_company": "Acme",
                "trade_amount": "12.30",
            },)
            return SqlExecutionResult(
                query_id="query-contract-test",
                normalized_sql=validated.sql,
                bound_filter_names=validated.bound_filter_names,
                schema_fingerprint=validated.schema_fingerprint,
                dataset_id=validated.dataset_id,
                is_synthetic=validated.is_synthetic,
                effective_start_date=validated.effective_start_date,
                effective_end_date=validated.effective_end_date,
                aggregation_grain=validated.aggregation_grain,
                time_grain=validated.time_grain,
                metric_names=validated.metric_names,
                rows=rows,
                row_count=1,
                result_hash=hash_rows(rows),
                raw_record_locators=(RawRecordLocator(source_id=1, raw_record_id="ROW-1"),),
                raw_record_locators_truncated=False,
                estimated_scan_rows=1,
                execution_ms=1.0,
                max_execution_time_ms=250,
                client_timeout_ms=250,
            )

    monkeypatch.setattr("trade_agent.agents.nodes.ReadOnlySqlExecutor", Executor)
    branch = ContractSqlBranch(
        registry=registry, scope=scope, connection_factory=Connection
    )

    evidence, degraded = branch.run(
        intent,
        timeout_seconds=0.25,
        identity_hmac_key=b"stable-runtime-key",
        entity_bindings={"company:acme": 17},
    )

    assert evidence and evidence[0].locator.branch == "sql"
    assert captured["identity_hmac_key"] == b"stable-runtime-key"
    assert captured["connection_timeout_seconds"] == 0.25
    assert "17" in captured["validated"].params.values()
    assert captured["closed"] is True
    assert degraded == ()


def test_sql_connection_factory_without_deadline_seam_fails_closed(tmp_path: Path) -> None:
    calls = 0

    def unsupported_factory():
        nonlocal calls
        calls += 1
        return object()

    branch = ContractSqlBranch(
        registry=registry_fixture.__wrapped__(),
        scope=SqlDataScope(
            dataset_id="trade-seed-v1",
            synthetic=True,
            start_date=date(2026, 1, 1),
            end_date=AS_OF,
        ),
        connection_factory=unsupported_factory,
    )
    graph = build_trade_graph(_deps(tmp_path, sql=branch))

    result = _invoke(graph, "最近半年贸易金额")

    assert calls == 0
    assert result["errors"][-1]["code"] == "sql_connection_timeout_unsupported"


def test_hanging_sql_connection_creation_is_bounded(tmp_path: Path) -> None:
    started = Event()
    release = Event()
    calls = 0

    def hanging_factory(*, timeout_seconds: float):
        nonlocal calls
        assert timeout_seconds == 0.03
        calls += 1
        started.set()
        release.wait()
        return object()

    branch = ContractSqlBranch(
        registry=registry_fixture.__wrapped__(),
        scope=SqlDataScope(
            dataset_id="trade-seed-v1",
            synthetic=True,
            start_date=date(2026, 1, 1),
            end_date=AS_OF,
        ),
        connection_factory=hanging_factory,
    )
    graph = build_trade_graph(
        _deps(
            tmp_path,
            sql=branch,
            timeout=0.03,
            max_outstanding_calls=1,
        )
    )
    try:
        result = _invoke(graph, "最近半年贸易金额")
    finally:
        release.set()

    assert started.is_set()
    assert calls == 1
    assert result["errors"][-1]["code"] == "sql_timeout"
    assert result["answer"] is None


def test_answer_generator_exception_becomes_a_typed_terminal_error(tmp_path: Path) -> None:
    class ExplodingGenerator:
        max_output_tokens = 128

        def generate(self, intent, evidence, validation):
            del intent, evidence, validation
            raise RuntimeError("provider secret detail")

    graph = build_trade_graph(
        _deps(
            tmp_path,
            sql=StaticEvidenceBranch((_sql(),)),
            answer_generator=ExplodingGenerator(),
        )
    )

    result = _invoke(graph, "最近半年贸易金额")

    assert result["terminal"] is True
    assert result["answer"] is None
    assert result["errors"][-1]["code"] == "internal_error"
    assert "provider secret detail" not in repr(result)


def test_finalizer_rebuilds_public_text_from_guard_retained_claims(tmp_path: Path) -> None:
    class ForgedAnswerGuard:
        def guard(self, draft, evidence, intent, validation):
            del evidence, intent, validation
            return GuardOutcome(
                accepted=True,
                answer="forged provider text",
                claims=draft.claims,
                refusal_reason=None,
            )

    graph = build_trade_graph(
        _deps(
            tmp_path,
            sql=StaticEvidenceBranch((generation_sql_evidence(),)),
            claim_guard=ForgedAnswerGuard(),
        )
    )

    result = _invoke(
        graph, "最近半年美国采购 HS850440 金额最高的 10 家公司"
    )

    expected = "\n".join(claim["text"] for claim in result["claims"])
    assert result["answer"] == expected
    assert "forged provider text" not in result["answer"]


def test_provider_answer_generator_obeys_zero_llm_budget(tmp_path: Path) -> None:
    class ProviderGenerator:
        def __init__(self) -> None:
            self.calls = 0
            self.max_output_tokens = 128

        def generate(self, intent, evidence, validation):
            self.calls += 1
            return DeterministicAnswerGenerator().generate(intent, evidence, validation)

    generator = ProviderGenerator()
    graph = build_trade_graph(
        _deps(
            tmp_path,
            sql=StaticEvidenceBranch((generation_sql_evidence(),)),
            answer_generator=generator,
            max_llm_calls=0,
        )
    )

    result = _invoke(
        graph, "最近半年美国采购 HS850440 金额最高的 10 家公司"
    )

    assert generator.calls == 0
    assert result["errors"][-1]["code"] == "llm_limit_exceeded"
    assert result["node_status"]["answer_draft"] == "failed"


def test_hanging_calls_cannot_accumulate_past_graph_capacity(tmp_path: Path) -> None:
    branch = HangingRetrievalBranch()
    graph = build_trade_graph(
        _deps(
            tmp_path,
            rag=branch,
            timeout=0.03,
            max_outstanding_calls=1,
        )
    )
    try:
        first = _invoke(graph, "官网最近是否扩产")
        second = _invoke(graph, "官网最近是否扩产")
    finally:
        branch.release.set()

    assert first["errors"][-1]["code"] == "rag_timeout"
    assert second["errors"][-1]["code"] == "rag_timeout"
    assert branch.call_count == 1


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (TimeoutError("provider timeout detail"), "rewrite_timeout"),
        (RuntimeError("provider secret detail"), "internal_error"),
    ],
)
def test_query_rewriter_failures_are_counted_and_classified(
    tmp_path: Path, failure: Exception, expected_code: str
) -> None:
    calls = 0

    def rewriter(question: str, count: int) -> str:
        nonlocal calls
        del question, count
        calls += 1
        raise failure

    graph = build_trade_graph(
        _deps(
            tmp_path,
            rag=StaticEvidenceBranch(error=TimeoutError("retrieval deadline")),
            query_rewriter=rewriter,
        )
    )

    result = _invoke(graph, "官网最近是否扩产")

    assert calls == 1
    assert result["llm_calls"] == 1
    assert result["terminal"] is True
    assert result["errors"][-1]["code"] == expected_code
    assert str(failure) not in repr(result)
