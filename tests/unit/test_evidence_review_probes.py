from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from trade_agent.data.manifest import BuildManifest
from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog
from trade_agent.data.router import DocumentRouter, SourceInput
from trade_agent.db.sql_executor import ReadOnlySqlExecutor, SqlExecutionResult
from trade_agent.evidence.models import (
    Claim,
    IntelligenceAnswer,
    PublicTrace,
    RawRecordLocator,
    canonical_public_url,
    sql_evidence_id,
)
from trade_agent.evidence.normalize import normalize_retrieval
from trade_agent.evidence.sql import build_sql_evidence
from trade_agent.entities.resolver import ResolutionOutcome
from trade_agent.retrieval.fusion import ComponentRank
from trade_agent.retrieval.planner import QueryIntent, RetrievalPlanner
from trade_agent.retrieval.profiles import load_retrieval_profile
from trade_agent.retrieval.service import HitTrace, RetrievalHit, RetrievalOutcome
from trade_agent.schemas.source import FileType, SourceType


NOW = datetime(2026, 8, 30, tzinfo=timezone.utc)


def _sql_result(**updates: object) -> SqlExecutionResult:
    rows = ({"trade_amount": Decimal("12.30"), "currency": "USD", "unit": "kg"},)
    values: dict[str, object] = {
        "query_id": "sqlq_" + "a" * 64,
        "normalized_sql": "SELECT SUM(tr.trade_amount) AS trade_amount, tr.currency AS currency, 'kg' AS unit FROM trade_records AS tr LIMIT 50",
        "bound_filter_names": ("country_code_0",),
        "schema_fingerprint": "b" * 64,
        "dataset_id": "trade-seed-v1",
        "is_synthetic": True,
        "effective_start_date": date(2026, 1, 1),
        "effective_end_date": date(2026, 6, 30),
        "aggregation_grain": ("currency", "unit"),
        "time_grain": "total",
        "metric_names": ("trade_amount",),
        "rows": rows,
        "row_count": 1,
        "result_hash": ReadOnlySqlExecutor._hash_rows(rows),
        "raw_record_locators": (RawRecordLocator(source_id=7, raw_record_id="ROW-9"),),
        "raw_record_locators_truncated": False,
        "estimated_scan_rows": 3,
        "execution_ms": 2.5,
        "max_execution_time_ms": 2_000,
        "client_timeout_ms": 3_000,
    }
    values.update(updates)
    return SqlExecutionResult.model_validate(values)


@pytest.fixture(scope="module")
def published_build(tmp_path_factory: pytest.TempPathFactory) -> BuildManifest:
    tmp_path = tmp_path_factory.mktemp("published-evidence-build")
    root = tmp_path / "corpus"
    (root / "manifests").mkdir(parents=True)
    source = root / "products.json"
    source.write_text(
        json.dumps(
            {
                "products": [
                    {
                        "product_id": "P-1",
                        "product_name": "Audited charger",
                        "sku": "CH-1",
                        "supplier": "Example Components",
                        "hs_code": "850440",
                        "url": "https://marketplace.example/items/P-1?signature=private#offer",
                        "description": "An audited production line makes this charger.",
                        "complianceLabel": "CE",
                        "packaging-labels": ["FRAGILE", "KEEP DRY"],
                        "goldAnswer": "evaluation-only",
                        "nested": [{"Reference-Claim-ID": "REF-DO-NOT-LEAK"}],
                    }
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    profile = root / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "aggregation_grain": "company_country_hs_calendar_month",
                "aggregation_window": {
                    "calendar_month": "2026-08",
                    "start": "2026-08-01",
                    "GoldAnswer": "PROFILE-NESTED-LEAK",
                },
                "calendar_month": "2026-08",
                "company": "Example Components",
                "company_id": 42,
                "country_code": "US",
                "currency": "USD",
                "export_amount_usd": "12.30",
                "export_quantity_kg": "4.00",
                "hs_code": "850440",
                "import_amount_usd": "0",
                "import_quantity_kg": "0",
                "raw_record_summary": {
                    "record_id_hash": "d" * 64,
                    "record_ids": ["ROW-9"],
                    "goldAnswer": "PROFILE-SUMMARY-LEAK",
                },
                "roles_included": ["export"],
                "source_record_count": 1,
                "synthetic_notice": "test-only",
                "total_amount_usd": "12.30",
                "total_quantity_kg": "4.00",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "manifests" / "corpus_manifest.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "path": "products.json",
                        "content_hash": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "entity_id": "company-42",
                        "fact_type": "company_status",
                        "file_type": "json",
                        "source_type": "b2b",
                        "ingested_at": NOW.isoformat(),
                        "publish_time": NOW.isoformat(),
                        "valid_from": NOW.isoformat(),
                        "is_synthetic": True,
                        "locator": {"row": 1, "expectedOutput": "do-not-index"},
                        "reference_claim_ids": ["CLAIM-1"],
                    },
                    {
                        "path": "profile.json",
                        "content_hash": hashlib.sha256(profile.read_bytes()).hexdigest(),
                        "entity_id": "company-42",
                        "fact_type": "trade_activity",
                        "file_type": "generated_profile",
                        "source_type": "customs_profile",
                        "ingested_at": NOW.isoformat(),
                        "publish_time": NOW.isoformat(),
                        "valid_from": NOW.isoformat(),
                        "is_synthetic": True,
                        "locator": {"profile": "monthly_company_hs"},
                    }
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        "\n".join(
            [
                "catalog_version: 1",
                f"root: {root.as_posix()}",
                "synthetic_notice: test-only",
                "sources:",
                "  - source_type: b2b",
                "    file_types: [json]",
                "    paths: [products.json]",
                "    url_pattern: https://marketplace.example/*",
                "  - source_type: customs_profile",
                "    file_types: [generated_profile]",
                "    paths: [profile.json]",
                "    url_pattern: https://profiles.example/*",
            ]
        ),
        encoding="utf-8",
    )
    return IngestionPipeline().run(SourceCatalog.from_yaml(catalog), tmp_path / "build.json")


def _outcome(
    build: BuildManifest,
    *,
    duplicate: bool = False,
    source_type: SourceType = SourceType.B2B,
) -> RetrievalOutcome:
    matching = tuple(
        snapshot.restore()
        for snapshot in build.chunks
        if snapshot.restore().metadata.source_type is source_type
    )
    assert matching, build.quarantined
    record = matching[0]
    component = ComponentRank(rank=4, raw_score=12.5, retriever_weight=0.6, relevance_contribution=0.01)
    trace = HitTrace(
        build_id=build.build_id,
        profile_id="balanced-v1",
        profile_version="trade-source-profiles-v1",
        planner_version="trade-retrieval-planner-v1",
        filter_expression_version="trade-filter-v1",
        components={"bm25": component},
        fusion_score=0.031,
        source_prior=record.metadata.source_weight,
        rerank_score=9.5,
        pre_rerank_rank=3,
        entity_resolution=ResolutionOutcome(
            status="resolved", entity_id="company-42", reason="frozen_metadata", confidence=1.0
        ),
        dedupe_cluster_id=record.metadata.dedupe_cluster_id or record.metadata.content_hash,
        duplicate_chunk_ids=(record.metadata.chunk_id,),
        dedupe_reasons=frozenset({"content_hash"}),
        degradation=("dense_unavailable",),
    )
    hit = RetrievalHit(chunk_id=record.metadata.chunk_id, record=record, rank=1, trace=trace)
    plan = RetrievalPlanner().plan(QueryIntent(query="latest production status"))
    return RetrievalOutcome(
        build_id=build.build_id,
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
        hits=(hit, hit) if duplicate else (hit,),
        degradation=("dense_unavailable",),
        diversity_dropped_chunk_ids=(),
        source_diversity_cap=3,
    )


def test_sql_evidence_rejects_forged_execution_result_and_public_semantic_tamper() -> None:
    with pytest.raises(ValueError, match="result hash"):
        build_sql_evidence(_sql_result(result_hash="c" * 64))
    evidence = build_sql_evidence(_sql_result())[0]
    for update in (
        {"content": evidence.content.replace("12.30", "999.00")},
        {"source_type": "regulator"},
        {"fact_type": "risk"},
        {"confidence": 0.2},
        {"currencies": ("EUR",)},
        {"units": ("items",)},
        {"time_grain": "month"},
    ):
        with pytest.raises(ValidationError):
            evidence.model_copy(update=update)


def test_sql_identity_uses_typed_canonical_json_instead_of_delimiters() -> None:
    assert sql_evidence_id(identity={"dataset": "a:b", "schema": "c"}) != sql_evidence_id(
        identity={"dataset": "a", "schema": "b:c"}
    )
    assert sql_evidence_id(identity={"score": 0.1}) == sql_evidence_id(identity={"score": 0.1})


def test_normalizer_requires_published_membership_and_rejects_forged_hit(
    published_build: BuildManifest,
) -> None:
    outcome = _outcome(published_build)
    assert normalize_retrieval(outcome, published_manifest=published_build)
    hit = outcome.hits[0]
    forged_content = "fabricated unpublished content"
    forged_record = hit.record.model_copy(
        update={
            "content": forged_content,
            "metadata": hit.record.metadata.model_copy(
                update={"content_hash": hashlib.sha256(forged_content.encode()).hexdigest()}
            ),
        }
    )
    forged = hit.model_copy(update={"record": forged_record})
    with pytest.raises(ValueError, match="published manifest"):
        normalize_retrieval(
            outcome.model_copy(update={"hits": (forged,)}),
            published_manifest=published_build,
        )
    with pytest.raises(TypeError, match="published"):
        normalize_retrieval(outcome)


def test_public_url_is_canonical_and_never_contains_query_or_fragment(
    published_build: BuildManifest,
) -> None:
    evidence = normalize_retrieval(_outcome(published_build), published_manifest=published_build)[0]
    assert evidence.source_url == "https://marketplace.example/items/P-1"
    assert evidence.canonical_url == "https://marketplace.example/items/P-1"
    assert evidence.source_id == evidence.source_url
    payload = evidence.model_dump_json()
    assert "signature" not in payload and "private" not in payload and "#offer" not in payload
    with pytest.raises(ValueError, match="credential"):
        canonical_public_url("https://user:password@marketplace.example/items/P-1")
    with pytest.raises(ValidationError, match="canonical"):
        evidence.model_copy(update={"source_url": "https://marketplace.example/items/P-1?token=secret"})


def test_router_strips_normalized_supervision_keys_but_preserves_business_labels(tmp_path: Path) -> None:
    path = tmp_path / "products.json"
    path.write_text(
        json.dumps(
            {
                "products": [
                    {
                        "product_id": "P-1", "product_name": "Pump", "sku": "PUMP-1",
                        "supplier": "Maker", "hs_code": "841370", "url": "https://marketplace.example/p-1",
                        "label": "GOLD", "labels": ["REFERENCE"],
                        "complianceLabel": "RoHS", "packaging-labels": ["KEEP DRY"],
                        "productLabels": ["industrial", "export"],
                        "referencePrice": "12.30", "expectedDeliveryDate": "2026-10-01",
                        "goldPurity": "99.9%",
                        "goldAnswer": "secret-a", "REFERENCE-label": "secret-b",
                        "nested": [{"Expected_Output": "secret-c", "referenceClaimId": "secret-d"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    source = SourceInput(
        path=path, file_type=FileType.JSON, source_type=SourceType.B2B,
        source_id="source-review", title="products", language="en", fetched_at=NOW,
        is_synthetic=True, source_url="https://ingest.example/source",
        manifest_attributes={"locator": {"ExpectedAnswer": "secret-e", "claimId": "secret-f"}},
    )
    document = DocumentRouter().load(source)[0]
    payload = json.dumps(document.model_dump(mode="json"), sort_keys=True)
    for sentinel in ("secret-a", "secret-b", "secret-c", "secret-d", "secret-e", "secret-f"):
        assert sentinel not in payload
    source_payload = document.attributes["source_payload"]
    assert "label" not in source_payload and "labels" not in source_payload
    assert source_payload["complianceLabel"] == "RoHS"
    assert source_payload["packaging-labels"] == ["KEEP DRY"]
    assert source_payload["productLabels"] == ["industrial", "export"]
    assert source_payload["referencePrice"] == "12.30"
    assert source_payload["expectedDeliveryDate"] == "2026-10-01"
    assert source_payload["goldPurity"] == "99.9%"
    assert "GOLD" not in payload and "REFERENCE" not in payload


def test_profile_uses_one_safe_payload_for_content_attributes_locator_and_currency(
    published_build: BuildManifest,
) -> None:
    outcome = _outcome(published_build, source_type=SourceType.CUSTOMS_PROFILE)
    record = outcome.hits[0].record
    serialized_record = record.model_dump_json()
    assert "PROFILE-NESTED-LEAK" not in serialized_record
    assert "PROFILE-SUMMARY-LEAK" not in serialized_record
    assert record.metadata.aggregation_info["currency"] == "USD"

    evidence = normalize_retrieval(outcome, published_manifest=published_build)[0]
    assert evidence.currencies == ("USD",)
    claim = Claim(
        claim_id="claim_" + "9" * 64,
        text="The profile reports USD aggregates.",
        status="supported",
        evidence_ids=(evidence.evidence_id,),
        entity_id="company-42",
        fact_type="trade_activity",
        currency="USD",
        period_start=date(2026, 8, 1),
        period_end=date(2026, 8, 31),
        confidence=0.9,
    )
    assert IntelligenceAnswer(
        answer="Supported.", claims=(claim,), evidence=(evidence,), conflicts=(),
        refusal_reason=None, degraded_components=(),
        public_trace=PublicTrace(branches=("rag",), evidence_ids=(evidence.evidence_id,), conflict_ids=()),
    )


def test_answer_requires_semantically_compatible_evidence_and_answer_xor_refusal(
    published_build: BuildManifest,
) -> None:
    evidence = normalize_retrieval(_outcome(published_build), published_manifest=published_build)[0]
    claim = Claim(
        claim_id="claim_" + "1" * 64,
        text="Example Components operated an audited production line.",
        status="supported",
        evidence_ids=(evidence.evidence_id,),
        entity_id="company-42",
        fact_type=evidence.fact_type,
        period_start=date(2026, 8, 1),
        period_end=date(2026, 8, 31),
        confidence=0.9,
    )
    values = {
        "answer": "Supported.", "claims": (claim,), "evidence": (evidence,), "conflicts": (),
        "refusal_reason": None, "degraded_components": (),
        "public_trace": PublicTrace(branches=("rag",), evidence_ids=(evidence.evidence_id,), conflict_ids=()),
    }
    assert IntelligenceAnswer(**values)
    for update in (
        {"entity_id": "company-other"},
        {"fact_type": "risk"},
        {"currency": "EUR"},
        {"unit": "kg"},
        {"period_start": date(2026, 1, 1), "period_end": date(2026, 1, 31)},
    ):
        with pytest.raises(ValidationError, match="semantic"):
            IntelligenceAnswer(**{**values, "claims": (claim.model_copy(update=update),)})
    with pytest.raises(ValidationError, match="exactly one"):
        IntelligenceAnswer(**{**values, "refusal_reason": "also refusing"})
    incompatible = build_sql_evidence(_sql_result())[0]
    with pytest.raises(ValidationError, match="semantic"):
        IntelligenceAnswer(
            **{
                **values,
                "claims": (
                    claim.model_copy(
                        update={"evidence_ids": (evidence.evidence_id, incompatible.evidence_id)}
                    ),
                ),
                "evidence": (evidence, incompatible),
                "public_trace": PublicTrace(
                    branches=("rag", "sql"),
                    evidence_ids=(evidence.evidence_id, incompatible.evidence_id),
                    conflict_ids=(),
                ),
            }
        )


def test_same_identity_duplicate_must_differ_only_by_ranking(
    published_build: BuildManifest,
) -> None:
    outcome = _outcome(published_build, duplicate=True)
    first = outcome.hits[0]
    ranking_only = first.model_copy(
        update={
            "rank": 7,
            "trace": first.trace.model_copy(update={"fusion_score": 99.0, "pre_rerank_rank": 8}),
        }
    )
    forward = normalize_retrieval(
        outcome.model_copy(update={"hits": (first, ranking_only)}),
        published_manifest=published_build,
    )
    reverse = normalize_retrieval(
        outcome.model_copy(update={"hits": (ranking_only, first)}),
        published_manifest=published_build,
    )
    assert forward == reverse
    semantic_mismatch = first.model_copy(
        update={"trace": first.trace.model_copy(update={"profile_version": "different-profile"})}
    )
    with pytest.raises(ValueError, match="duplicate"):
        normalize_retrieval(
            outcome.model_copy(update={"hits": (first, semantic_mismatch)}),
            published_manifest=published_build,
        )
