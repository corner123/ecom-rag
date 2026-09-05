from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from hashlib import sha256

import pytest
from pydantic import ValidationError

from trade_agent.db.sql_executor import SqlExecutionResult
from trade_agent.evidence.models import (
    Claim,
    Conflict,
    Evidence,
    IntelligenceAnswer,
    PublicTrace,
    RawRecordLocator,
)
from trade_agent.evidence.normalize import normalize_retrieval
from trade_agent.evidence.sql import build_sql_evidence
from trade_agent.entities.resolver import ResolutionOutcome
from trade_agent.retrieval.fusion import ComponentRank
from trade_agent.retrieval.planner import QueryIntent, RetrievalPlanner
from trade_agent.retrieval.profiles import load_retrieval_profile
from trade_agent.retrieval.service import HitTrace, RetrievalHit, RetrievalOutcome
from trade_agent.schemas.source import ChunkMetadata, ChunkRecord, FactType, FileType, SourceLocator, SourceType


NOW = datetime(2026, 8, 30, tzinfo=timezone.utc)
BUILD_ID = "build_0123456789abcdef0123456789abcdef"


def _sql_result() -> SqlExecutionResult:
    return SqlExecutionResult(
        query_id="sqlq_" + "a" * 64,
        normalized_sql="SELECT SUM(tr.trade_amount) AS trade_amount, tr.currency AS currency FROM trade_records AS tr LIMIT 50",
        bound_filter_names=("country_code_0",),
        schema_fingerprint="b" * 64,
        dataset_id="trade-seed-v1",
        is_synthetic=True,
        effective_start_date=date(2026, 1, 1),
        effective_end_date=date(2026, 6, 30),
        aggregation_grain=("currency",),
        time_grain="total",
        metric_names=("trade_amount",),
        rows=({"trade_amount": Decimal("12.30"), "currency": "USD"},),
        row_count=1,
        result_hash="c" * 64,
        raw_record_locators=(RawRecordLocator(source_id=7, raw_record_id="ROW-9"),),
        raw_record_locators_truncated=False,
        estimated_scan_rows=3,
        execution_ms=2.5,
        max_execution_time_ms=2_000,
        client_timeout_ms=3_000,
    )


def _retrieval_outcome(
    *,
    rank: int = 1,
    fusion_score: float = 0.031,
    source_type: SourceType = SourceType.INDUSTRY_NEWS,
    fact_type: FactType = FactType.COMPANY_STATUS,
    duplicate: bool = False,
) -> RetrievalOutcome:
    content = "Example Components opened a new audited production line."
    metadata = ChunkMetadata(
        chunk_id="chunk_0123456789abcdef0123456789abcdef",
        chunk_index=2,
        document_id="doc_0123456789abcdef0123456789abcdef",
        entity_id="company-42",
        company_name="Example Components",
        country_code="US",
        hs_code="850440",
        file_type=FileType.JSON,
        source_type=source_type,
        source_weight=0.8,
        fact_type=fact_type,
        publish_time=NOW,
        valid_from=NOW,
        ingested_at=NOW,
        source_url="https://news.example/items/NEWS-9",
        canonical_url="https://news.example/items/NEWS-9",
        source_locator=SourceLocator(
            row=9,
            raw={
                "story_id": "NEWS-9",
                "claim_id": "GOLD-DO-NOT-LEAK",
                "manifest_claim_ids": ["REF-DO-NOT-LEAK"],
                "token": "secret-do-not-leak",
                "path": "/Users/private/corpus/news.json",
            },
        ),
        raw_record_id="NEWS-9",
        aggregation_info={"aggregation_grain": "document", "currency": "USD", "unit": "count"},
        content_hash=sha256(content.encode()).hexdigest(),
        parent_document_hash="d" * 64,
        language="en",
        is_synthetic=True,
        dedupe_cluster_id="story-family-9",
    )
    record = ChunkRecord(content=content, metadata=metadata)
    component = ComponentRank(rank=4, raw_score=12.5, retriever_weight=0.6, relevance_contribution=0.01)
    trace = HitTrace(
        build_id=BUILD_ID,
        profile_id="balanced-v1",
        profile_version="trade-source-profiles-v1",
        planner_version="trade-retrieval-planner-v1",
        filter_expression_version="trade-filter-v1",
        components={"bm25": component},
        fusion_score=fusion_score,
        source_prior=0.8,
        rerank_score=9.5,
        pre_rerank_rank=3,
        entity_resolution=ResolutionOutcome(
            status="resolved", entity_id="company-42", reason="frozen_metadata", confidence=1.0
        ),
        dedupe_cluster_id="dedupe-9",
        duplicate_chunk_ids=(metadata.chunk_id,),
        dedupe_reasons=frozenset({"content_hash"}),
        degradation=("dense_unavailable",),
    )
    hit = RetrievalHit(chunk_id=metadata.chunk_id, record=record, rank=rank, trace=trace)
    plan = RetrievalPlanner().plan(QueryIntent(query="latest production status"))
    hits = (hit, hit) if duplicate else (hit,)
    return RetrievalOutcome(
        build_id=BUILD_ID,
        query=plan.query,
        plan=plan,
        filter_expression="",
        profile=load_retrieval_profile(),
        dense_hits=(),
        sparse_hits=(),
        fused_hits=(),
        reranked_hits=None,
        rerank_contract=None,
        rerank_degraded=False,
        rerank_error_code=None,
        rerank_latency_ms=0.0,
        hits=hits,
        degradation=("dense_unavailable",),
        diversity_dropped_chunk_ids=(),
        source_diversity_cap=3,
    )


def test_sql_and_rag_evidence_share_contract() -> None:
    sql_evidence = build_sql_evidence(_sql_result())[0]
    rag_evidence = normalize_retrieval(_retrieval_outcome())[0]

    assert type(sql_evidence) is type(rag_evidence) is Evidence
    assert sql_evidence.locator.scope == "bounded_predicate_population"
    assert sql_evidence.locator.raw_record_locators
    assert rag_evidence.locator.chunk_id == "chunk_0123456789abcdef0123456789abcdef"
    assert rag_evidence.locator.collection_name == "trade_intel_chunks_0123456789abcdef0123"
    assert rag_evidence.locator.document_id.startswith("doc_")


def test_retrieval_identity_ignores_rank_and_float_scores_but_scopes_semantics() -> None:
    first = normalize_retrieval(_retrieval_outcome(rank=1, fusion_score=0.1))[0]
    later = normalize_retrieval(_retrieval_outcome(rank=7, fusion_score=99.125))[0]
    other_fact = normalize_retrieval(_retrieval_outcome(fact_type=FactType.RISK))[0]
    other_source = normalize_retrieval(_retrieval_outcome(source_type=SourceType.B2B))[0]

    assert first.evidence_id == later.evidence_id
    assert first.evidence_id != other_fact.evidence_id
    assert first.evidence_id != other_source.evidence_id
    assert later.retrieval_provenance.rank == 7
    assert later.retrieval_provenance.fusion_score == 99.125
    assert later.confidence == 1.0
    assert later.retrieval_provenance.score_meaning == "ranking_only"


def test_duplicate_retrieval_evidence_is_deduplicated() -> None:
    assert len(normalize_retrieval(_retrieval_outcome(duplicate=True))) == 1


def test_duplicate_selection_is_independent_of_input_order() -> None:
    outcome = _retrieval_outcome()
    first = outcome.hits[0]
    other = first.model_copy(update={
        "trace": first.trace.model_copy(update={"fusion_score": 20.0, "rerank_score": -1.0})
    })
    forward = normalize_retrieval(outcome.model_copy(update={"hits": (first, other)}))
    reversed_ = normalize_retrieval(outcome.model_copy(update={"hits": (other, first)}))

    assert forward == reversed_


def test_generation_evidence_projection_excludes_evaluation_labels_secrets_and_paths() -> None:
    evidence = normalize_retrieval(_retrieval_outcome())[0]
    payload = evidence.model_dump_json()

    for sentinel in (
        "GOLD-DO-NOT-LEAK",
        "REF-DO-NOT-LEAK",
        "secret-do-not-leak",
        "/Users/private",
        "claim_id",
        "reference_claim",
    ):
        assert sentinel not in payload
    assert evidence.raw_record_id == "NEWS-9"
    assert evidence.locator.source_record_id == "NEWS-9"
    assert evidence.company_name == "Example Components"
    assert evidence.country_code == "US"
    assert evidence.hs_code == "850440"


def test_sql_evidence_exposes_units_currency_time_and_branch_without_raw_rows() -> None:
    evidence = build_sql_evidence(_sql_result())[0]

    assert evidence.currencies == ("USD",)
    assert evidence.units == ()
    assert evidence.aggregation_grain == ("currency",)
    assert evidence.sql_provenance.branch == "sql"
    assert evidence.sql_provenance.effective_start_date == date(2026, 1, 1)
    assert "ROW-9" not in evidence.content


def test_claim_answer_and_conflict_enforce_citations_and_validity() -> None:
    evidence = normalize_retrieval(_retrieval_outcome())[0]
    claim = Claim(
        claim_id="claim_" + "1" * 64,
        text="Example Components opened a production line.",
        status="supported",
        evidence_ids=(evidence.evidence_id,),
        entity_id="company-42",
        fact_type="company_status",
        confidence=0.9,
    )
    conflict = Conflict(
        conflict_id="conflict_" + "2" * 64,
        entity_id="company-42",
        fact_type="company_status",
        evidence_ids=(evidence.evidence_id, "rag_" + "3" * 64),
        status="unresolved",
        valid_from=date(2026, 1, 1),
        valid_to=date(2026, 12, 31),
        unit=None,
        aggregation_grain=("document",),
        explanation="Two current sources disagree.",
    )
    answer = IntelligenceAnswer(
        answer="The company reports a new production line.",
        claims=(claim,),
        evidence=(evidence,),
        conflicts=(),
        refusal_reason=None,
        degraded_components=("dense_unavailable",),
        public_trace=PublicTrace(
            branches=("rag",), evidence_ids=(evidence.evidence_id,), conflict_ids=()
        ),
    )

    assert answer.claims[0].evidence_ids == (evidence.evidence_id,)
    assert conflict.status == "unresolved"
    with pytest.raises(ValidationError, match="generation evidence"):
        answer.model_copy(
            update={
                "claims": (
                    claim.model_copy(update={"evidence_ids": ("rag_" + "f" * 64,)}),
                )
            }
        )
    with pytest.raises(ValidationError):
        claim.model_copy(update={"confidence": 1.1})
    with pytest.raises(ValidationError, match="valid_from"):
        conflict.model_copy(update={"valid_from": date(2027, 1, 1)})


def test_supported_claim_needs_evidence_and_models_are_frozen_extra_forbid() -> None:
    with pytest.raises(ValidationError, match="supported claim"):
        Claim(
            claim_id="claim_" + "1" * 64,
            text="Unsupported assertion",
            status="supported",
            evidence_ids=(),
            confidence=0.5,
        )
    with pytest.raises(ValidationError):
        Evidence.model_validate({**build_sql_evidence(_sql_result())[0].model_dump(), "gold_label": "yes"})
    evidence = normalize_retrieval(_retrieval_outcome())[0]
    assert Evidence.model_validate_json(evidence.model_dump_json()) == evidence
    with pytest.raises(ValidationError, match="content-addressed"):
        evidence.model_copy(update={"evidence_id": "rag_" + "0" * 64})
    with pytest.raises(ValidationError, match="content hash"):
        evidence.model_copy(update={"content": "tampered chunk"})
    with pytest.raises(ValidationError, match="source identity"):
        evidence.model_copy(update={"source_id": "https://other.example/source"})
    with pytest.raises(ValidationError):
        evidence.content = "mutated"  # type: ignore[misc]


def test_normalizer_fails_closed_on_foreign_build_trace() -> None:
    outcome = _retrieval_outcome()
    hit = outcome.hits[0]
    forged = hit.model_copy(
        update={"trace": hit.trace.model_copy(update={"build_id": "build_" + "f" * 32})}
    )
    with pytest.raises(ValueError, match="build"):
        normalize_retrieval(outcome.model_copy(update={"hits": (forged,)}))


def test_retrieval_locator_rejects_nonreplayable_build_and_private_path() -> None:
    locator = normalize_retrieval(_retrieval_outcome())[0].locator
    with pytest.raises(ValidationError, match="collection"):
        locator.model_copy(update={"collection_name": "trade_intel_chunks_forged"})
    with pytest.raises(ValidationError, match="source_identity"):
        locator.model_copy(update={"source_identity": "/Users/private/corpus.json"})
