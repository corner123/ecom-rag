from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json

import pytest
from pydantic import ValidationError

from trade_agent.agents.intent import IntentParser, QueryIntent
from trade_agent.db.sql_executor import ReadOnlySqlExecutor, SqlExecutionResult
from trade_agent.evidence.models import (
    Conflict,
    Evidence,
    RawRecordLocator,
    RetrievalComponentProvenance,
    RetrievalEvidenceLocator,
    RetrievalProvenance,
    _evidence_identity,
    retrieval_evidence_id,
)
from trade_agent.evidence.sql import build_sql_evidence
from trade_agent.retrieval.filters import RetrievalFilter


AS_OF = date(2026, 9, 4)
BUILD_ID = "build_" + "1" * 32
MANIFEST_HASH = "2" * 64
COLLECTION = "trade_intel_chunks_" + BUILD_ID[6:26]


def _intent(question: str, *, filters: RetrievalFilter | None = None) -> QueryIntent:
    return IntentParser(as_of=AS_OF).parse(question, explicit_filters=filters)


def _sql(
    *,
    metric: str = "trade_amount",
    start: date = date(2026, 3, 4),
    end: date = AS_OF,
    time_grain: str = "total",
    aggregation_grain: tuple[str, ...] | None = None,
    currency: str | None = "USD",
    unit: str | None = None,
    rows: bool = True,
    truncated: bool = False,
    scope: str = "top",
) -> Evidence:
    dimensions: dict[str, object] = {}
    if currency is not None:
        dimensions["currency"] = currency
    if unit is not None:
        dimensions["unit"] = unit
    if time_grain == "month":
        dimensions["trade_date"] = "2026-08"
    row = {"importer_company": "Acme", **dimensions, metric: Decimal("12.30")}
    result_rows = (row,) if rows else ()
    grain = aggregation_grain
    if grain is None:
        grain = ("importer_company",) + (("trade_date",) if time_grain == "month" else ())
        if currency is not None:
            grain += ("currency",)
        if unit is not None:
            grain += ("unit",)
    if scope == "lead":
        where = "importer.company_name = :filter_0_0 AND importer.id = :filter_1_0"
        filter_names = ("filter_0_0", "filter_1_0")
    elif scope == "quantity":
        where = "export_country.country_code = :filter_0_0 AND hs.hs_code = :filter_1_0"
        filter_names = ("filter_0_0", "filter_1_0")
    else:
        where = (
            "import_country.country_code = :filter_0_0 AND hs.hs_code = :filter_1_0 "
            "AND tr.trade_date BETWEEN :filter_2_start AND :filter_2_end"
        )
        filter_names = ("filter_0_0", "filter_1_0", "filter_2_end", "filter_2_start")
    result = SqlExecutionResult(
        query_id="sqlq_" + "a" * 64,
        normalized_sql=(
            f"SELECT importer.company_name AS importer_company, {metric}, "
            "import_country.country_code, hs.hs_code FROM trade_records AS tr "
            "JOIN companies AS importer ON tr.importer_id = importer.id "
            "JOIN countries AS import_country ON tr.import_country_id = import_country.id "
            "JOIN hs_codes AS hs ON tr.hs_code_id = hs.id "
            f"WHERE {where} LIMIT 50"
        ),
        bound_filter_names=(*filter_names,
            "policy_end_date", "policy_is_synthetic", "policy_start_date",
        ),
        schema_fingerprint="b" * 64,
        dataset_id="trade-seed-v1",
        is_synthetic=True,
        effective_start_date=start,
        effective_end_date=end,
        aggregation_grain=grain,
        time_grain=time_grain,
        metric_names=(metric,),
        rows=result_rows,
        row_count=len(result_rows),
        result_hash=ReadOnlySqlExecutor._hash_rows(result_rows),
        raw_record_locators=(RawRecordLocator(source_id=7, raw_record_id="ROW-9"),) if rows else (),
        raw_record_locators_truncated=truncated,
        estimated_scan_rows=3,
        execution_ms=2.5,
        max_execution_time_ms=2_000,
        client_timeout_ms=3_000,
    )
    return build_sql_evidence(result)[0]


def _rag(
    *,
    suffix: str,
    content: str = "Acme opened a new audited production line and remains operational.",
    fact_type: str = "company_status",
    source_type: str = "official_website",
    entity_id: str | None = "company:acme",
    company_name: str | None = "Acme",
    country_code: str | None = None,
    hs_code: str | None = None,
    publish_time: datetime | None = datetime(2026, 8, 20, tzinfo=timezone.utc),
    valid_from: date | datetime | None = date(2026, 8, 20),
    valid_to: date | datetime | None = None,
    degradation: tuple[str, ...] = (),
    source_id: str | None = None,
    cluster: str | None = None,
    fusion_score: float = 0.01,
    rerank_score: float | None = 0.2,
    source_weight: float = 0.8,
) -> Evidence:
    source = source_id or f"https://source-{suffix}.example/status"
    digest = sha256(content.encode("utf-8")).hexdigest()
    locator = RetrievalEvidenceLocator(
        manifest_id=BUILD_ID,
        manifest_fingerprint=MANIFEST_HASH,
        build_id=BUILD_ID,
        collection_name=COLLECTION,
        source_identity=source,
        document_id=f"doc-{suffix}",
        chunk_id=f"chunk-{suffix}",
        chunk_index=0,
        content_hash=digest,
        source_record_id=f"record-{suffix}",
        section="status",
    )
    provenance = RetrievalProvenance(
        build_id=BUILD_ID,
        manifest_fingerprint=MANIFEST_HASH,
        collection_name=COLLECTION,
        profile_id="balanced-v1",
        profile_version="profiles-v1",
        planner_version="planner-v1",
        filter_expression_version="filter-v1",
        rank=1,
        pre_rerank_rank=1,
        components=(
            RetrievalComponentProvenance(
                retriever="bm25", rank=1, raw_score=1.0,
                retriever_weight=0.5, relevance_contribution=0.5,
            ),
        ),
        fusion_score=fusion_score,
        source_prior=1.0,
        rerank_score=rerank_score,
        entity_resolution_status="resolved" if entity_id else "unresolved",
        entity_resolution_id=entity_id,
        entity_resolution_reason="frozen_metadata" if entity_id else "no_safe_match",
        entity_resolution_confidence=1.0 if entity_id else 0.0,
        dedupe_cluster_id=cluster or f"cluster-{suffix}",
        duplicate_chunk_ids=(f"chunk-{suffix}",),
        dedupe_reasons=("content_hash",),
        degraded_components=degradation,
    )
    values = dict(
        entity_id=entity_id,
        company_name=company_name,
        country_code=country_code,
        hs_code=hs_code,
        fact_type=fact_type,
        source_type=source_type,
        source_id=source,
        source_weight=source_weight,
        content=content,
        source_url=source,
        canonical_url=source,
        locator=locator,
        raw_record_id=f"record-{suffix}",
        publish_time=publish_time,
        valid_from=valid_from,
        valid_to=valid_to,
        time_grain="document",
        currencies=(),
        units=(),
        aggregation_grain=(),
        confidence=1.0,
        confidence_basis="content_hash_verified",
        is_synthetic=True,
        retrieval_provenance=provenance,
        sql_provenance=None,
    )
    provisional = Evidence.model_construct(evidence_id="rag_" + "0" * 64, **values)
    return Evidence(evidence_id=retrieval_evidence_id(identity=_evidence_identity(provisional)), **values)


@pytest.fixture
def validator():
    from trade_agent.evidence.validator import EvidenceValidator

    return EvidenceValidator()


def _codes(outcome) -> set[str]:
    return {reason.code for reason in outcome.reasons}


def test_mixed_lead_requires_trade_and_two_current_independent_status_sources(validator) -> None:
    intent = _intent(
        "Acme 是否值得跟进",
        filters=RetrievalFilter(entity_ids=("company:acme",), is_synthetic=True),
    )
    evidence = (_sql(start=date(2025, 9, 4), scope="lead"), _rag(suffix="official"), _rag(suffix="news", source_type="industry_news"))

    outcome = validator.validate(intent, evidence, (), AS_OF)

    assert outcome.can_answer is True
    assert outcome.decision == "answer"
    assert outcome.error_code is None
    assert {item.branch for item in outcome.satisfied_requirements} == {"rag", "sql"}


def test_sql_only_amount_and_quantity_contracts_are_answerable(validator) -> None:
    amount_intent = _intent("最近半年美国采购 HS850440 金额最高的 10 家公司")
    quantity_intent = _intent("中国出口 HS850440 的月度数量")
    quantity = _sql(
        metric="quantity", currency=None, unit="kg", time_grain="month",
        aggregation_grain=("trade_date", "unit"), scope="quantity",
    )

    amount_outcome = validator.validate(amount_intent, (_sql(),), (), AS_OF)
    quantity_outcome = validator.validate(quantity_intent, (quantity,), (), AS_OF)

    assert amount_outcome.can_answer is True
    assert quantity_outcome.can_answer is True


def test_one_mixed_branch_cannot_pass_and_social_status_is_low_authority(validator) -> None:
    intent = _intent("Acme 是否值得跟进", filters=RetrievalFilter(entity_ids=("company:acme",)))

    only_rag = validator.validate(intent, (_rag(suffix="official"), _rag(suffix="news", source_type="industry_news")), (), AS_OF)
    only_sql = validator.validate(intent, (_sql(start=date(2025, 9, 4)),), (), AS_OF)
    social = validator.validate(intent, (_sql(start=date(2025, 9, 4), scope="lead"), _rag(suffix="social", source_type="social")), (), AS_OF)

    assert not only_rag.can_answer and "missing" in _codes(only_rag)
    assert not only_sql.can_answer and "missing" in _codes(only_sql)
    assert only_rag.decision == only_sql.decision == "refuse"
    assert not social.can_answer and "low_authority" in _codes(social)


def test_duplicate_or_syndicated_sources_do_not_satisfy_diversity(validator) -> None:
    intent = _intent("Acme 是否值得跟进", filters=RetrievalFilter(entity_ids=("company:acme",)))
    first = _rag(suffix="one", cluster="same-story")
    syndicated = _rag(suffix="two", source_type="industry_news", cluster="same-story")

    outcome = validator.validate(intent, (_sql(start=date(2025, 9, 4), scope="lead"), first, syndicated), (), AS_OF)

    assert outcome.can_answer is False
    assert "insufficient_diversity" in _codes(outcome)


def test_repeated_pages_from_one_source_do_not_satisfy_diversity(validator) -> None:
    intent = _intent("Acme 是否值得跟进", filters=RetrievalFilter(entity_ids=("company:acme",)))
    first = _rag(suffix="page-1", source_id="https://acme.example/status", cluster="story-1")
    second = _rag(suffix="page-2", source_id="https://acme.example/status", cluster="story-2")

    outcome = validator.validate(
        intent, (_sql(start=date(2025, 9, 4), scope="lead"), first, second), (), AS_OF
    )

    assert outcome.can_answer is False
    assert "insufficient_diversity" in _codes(outcome)


def test_fact_label_without_factual_status_content_is_not_sufficient(validator) -> None:
    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    generic = _rag(suffix="generic", content="Acme synthetic demonstration information page.")

    outcome = validator.validate(intent, (generic,), (), AS_OF)

    assert outcome.can_answer is False
    assert "unsupported_fact_content" in _codes(outcome)


def test_entity_country_hs_and_requested_fact_scope_are_enforced(validator) -> None:
    intent = _intent(
        "Acme 官网最近状态",
        filters=RetrievalFilter(
            entity_ids=("company:acme",), country_codes=("US",), hs_codes=("850440",),
            fact_types=("company_status",),
        ),
    )
    wrong = _rag(
        suffix="wrong", entity_id="company:other", company_name="Other",
        country_code="CN", hs_code="999999", fact_type="market_signal",
        content="Other reports market demand growth.",
    )

    outcome = validator.validate(intent, (wrong,), (), AS_OF)

    assert outcome.can_answer is False
    assert "dimension_mismatch" in _codes(outcome)
    assert "fact_type_mismatch" in _codes(outcome)


def test_unresolved_entity_unknown_fact_and_future_evidence_fail_closed(validator) -> None:
    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    unresolved = _rag(suffix="unresolved", entity_id=None)
    unknown = _rag(suffix="unknown", fact_type="unknown")
    future = _rag(
        suffix="future", publish_time=datetime(2026, 10, 1, tzinfo=timezone.utc),
        valid_from=date(2026, 10, 1),
    )

    outcome = validator.validate(intent, (unresolved, unknown, future), (), AS_OF)

    assert outcome.can_answer is False
    assert {"unresolved_entity", "unknown_fact", "out_of_scope"} <= _codes(outcome)


def test_stale_current_status_and_sql_interval_gap_are_typed(validator) -> None:
    lead = _intent("Acme 是否值得跟进", filters=RetrievalFilter(entity_ids=("company:acme",)))
    stale_one = _rag(
        suffix="stale-1", publish_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        valid_from=date(2025, 1, 1), source_type="official_website",
    )
    stale_two = _rag(
        suffix="stale-2", publish_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
        valid_from=date(2025, 1, 2), source_type="industry_news",
    )
    stale = validator.validate(lead, (_sql(start=date(2025, 9, 4), scope="lead"), stale_one, stale_two), (), AS_OF)
    interval = _intent("最近半年美国采购 HS850440 金额最高的 10 家公司")
    gap = validator.validate(interval, (_sql(start=date(2026, 4, 1)),), (), AS_OF)

    assert not stale.can_answer and stale.error_code == "evidence_insufficient"
    assert "stale" in _codes(stale)
    assert not gap.can_answer and "temporal_gap" in _codes(gap)


def test_versioned_policy_age_boundaries_are_explicit_and_inclusive() -> None:
    from trade_agent.evidence.requirements import EvidenceRequirements
    from trade_agent.evidence.validator import EvidenceValidator

    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    requirements = EvidenceRequirements.for_intent(intent)
    requirement = requirements.items[0]
    exact = AS_OF - timedelta(days=requirement.maximum_age_days)
    old = exact - timedelta(days=1)

    accepted = EvidenceValidator().validate(
        intent,
        (_rag(suffix="exact-age", publish_time=datetime.combine(exact, datetime.min.time(), timezone.utc), valid_from=exact),),
        (),
        AS_OF,
    )
    rejected = EvidenceValidator().validate(
        intent,
        (_rag(suffix="too-old", publish_time=datetime.combine(old, datetime.min.time(), timezone.utc), valid_from=old),),
        (),
        AS_OF,
    )

    assert requirements.policy_version == "trade-evidence-sufficiency-v1"
    assert requirement.maximum_age_days == 180
    assert accepted.can_answer is True
    assert not rejected.can_answer and "stale" in _codes(rejected)


def test_requested_historical_publication_window_uses_interval_not_current_freshness(validator) -> None:
    historical_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    historical_end = datetime(2024, 12, 31, tzinfo=timezone.utc)
    intent = _intent(
        "查询历史法规",
        filters=RetrievalFilter(
            fact_types=("regulation",),
            source_types=("regulator",),
            published_after=historical_start,
            published_before=historical_end,
        ),
    )
    evidence = _rag(
        suffix="historical-regulation",
        fact_type="regulation",
        source_type="regulator",
        entity_id=None,
        company_name=None,
        content="The 2024 regulation requires documented compliance.",
        publish_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        valid_from=date(2024, 6, 1),
        valid_to=date(2024, 12, 31),
    )

    outcome = validator.validate(intent, (evidence,), (), AS_OF)

    assert outcome.requirements.items[0].maximum_age_days is None
    assert outcome.can_answer is True


def test_validator_uses_the_injected_versioned_source_policy() -> None:
    from trade_agent.evidence.requirements import FactSourcePolicy, SufficiencyPolicy
    from trade_agent.evidence.validator import EvidenceValidator

    base = SufficiencyPolicy()
    sources = tuple(
        FactSourcePolicy(fact_type=item.fact_type, allowed_source_types=("social",))
        if item.fact_type == "company_status"
        else item
        for item in base.fact_sources
    )
    policy = base.model_copy(update={"fact_sources": sources})
    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    social = _rag(suffix="policy-social", source_type="social")

    outcome = EvidenceValidator(policy).validate(intent, (social,), (), AS_OF)

    assert outcome.can_answer is True
    assert outcome.requirements.policy_version == policy.policy_version


def test_lead_sql_365_day_recency_boundary_is_inclusive(validator) -> None:
    intent = _intent("Acme 是否值得跟进", filters=RetrievalFilter(entity_ids=("company:acme",)))
    rag = (_rag(suffix="one"), _rag(suffix="two", source_type="industry_news"))
    exact = AS_OF - timedelta(days=365)
    old = exact - timedelta(days=1)

    accepted = validator.validate(intent, (_sql(start=date(2024, 1, 1), end=exact, scope="lead"), *rag), (), AS_OF)
    rejected = validator.validate(intent, (_sql(start=date(2024, 1, 1), end=old, scope="lead"), *rag), (), AS_OF)

    assert accepted.can_answer is True
    assert not rejected.can_answer and "stale" in _codes(rejected)


@pytest.mark.parametrize(
    ("evidence", "reason"),
    [
        (_sql(metric="quantity", currency=None, unit="kg"), "metric_mismatch"),
        (_sql(currency=None), "currency_missing"),
        (_sql(aggregation_grain=("importer_company",)), "currency_missing"),
        (_sql(metric="quantity", currency=None, unit=None), "metric_mismatch"),
    ],
)
def test_sql_metric_currency_unit_and_grain_must_match_intent(validator, evidence, reason) -> None:
    intent = _intent("最近半年美国采购 HS850440 金额最高的 10 家公司")
    outcome = validator.validate(intent, (evidence,), (), AS_OF)

    assert outcome.can_answer is False
    assert reason in _codes(outcome)


def test_quantity_requires_unit_and_requested_month_grain(validator) -> None:
    intent = _intent("中国出口 HS850440 的月度数量")
    wrong = _sql(metric="quantity", currency=None, unit=None, time_grain="total", aggregation_grain=(), scope="quantity")

    outcome = validator.validate(intent, (wrong,), (), AS_OF)

    assert {"unit_missing", "grain_mismatch"} <= _codes(outcome)


def test_empty_sql_result_and_truncated_locator_do_not_count(validator) -> None:
    intent = _intent("最近半年美国采购 HS850440 金额最高的 10 家公司")

    empty = validator.validate(intent, (_sql(rows=False),), (), AS_OF)
    truncated = validator.validate(intent, (_sql(truncated=True),), (), AS_OF)

    assert "empty_result" in _codes(empty)
    assert "truncated_locator" in _codes(truncated)
    assert not empty.can_answer and not truncated.can_answer


def test_unresolved_relevant_conflict_blocks_generation(validator) -> None:
    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    left = _rag(suffix="left")
    right = _rag(suffix="right", source_type="industry_news", content="Acme permanently closed operations.")
    conflict = Conflict(
        conflict_id="conflict_" + "3" * 64,
        entity_id="company:acme",
        fact_type="company_status",
        evidence_ids=(left.evidence_id, right.evidence_id),
        status="unresolved",
        valid_from=date(2026, 8, 1),
        explanation="Current sources disagree.",
    )

    outcome = validator.validate(intent, (left, right), (conflict,), AS_OF)

    assert outcome.can_answer is False
    assert outcome.decision == "refuse"
    assert outcome.error_code == "evidence_conflict"
    assert "contradictory" in _codes(outcome)


def test_resolved_conflict_uses_only_selected_support_and_is_order_independent(validator) -> None:
    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    selected = _rag(suffix="selected")
    rejected = _rag(suffix="rejected", source_type="industry_news", content="Acme permanently closed operations.")
    conflict = Conflict(
        conflict_id="conflict_" + "4" * 64,
        entity_id="company:acme",
        fact_type="company_status",
        evidence_ids=tuple(sorted((selected.evidence_id, rejected.evidence_id))),
        status="resolved",
        valid_from=date(2026, 8, 1),
        explanation="The official correction supersedes the report.",
        selected_evidence_ids=(selected.evidence_id,),
    )

    forward = validator.validate(intent, (selected, rejected), (conflict,), AS_OF)
    reverse = validator.validate(intent, (rejected, selected), (conflict,), AS_OF)

    assert forward == reverse
    assert forward.can_answer is True
    assert forward.eligible_evidence_ids == (selected.evidence_id,)
    assert forward.conflict_ids == (conflict.conflict_id,)


def test_fatal_degraded_required_branch_retries_but_partial_degradation_is_audited(validator) -> None:
    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    fatal = validator.validate(intent, (_rag(suffix="fatal", degradation=("retrieval_timeout",)),), (), AS_OF)
    partial = validator.validate(intent, (_rag(suffix="partial", degradation=("dense_unavailable",)),), (), AS_OF)

    assert fatal.can_answer is False
    assert fatal.decision == "rewrite_once"
    assert fatal.error_code == "evidence_retryable"
    assert "degraded_branch" in _codes(fatal)
    assert partial.can_answer is True
    assert partial.degraded_components == ("dense_unavailable",)


def test_ranking_and_weight_cannot_turn_bad_evidence_into_support(validator) -> None:
    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    malicious = _rag(
        suffix="malicious", content="Acme synthetic demonstration information page.",
        source_type="social", fusion_score=1e30, rerank_score=1e30, source_weight=1.0,
    )

    outcome = validator.validate(intent, (malicious,), (), AS_OF)

    assert outcome.can_answer is False
    assert {"low_authority", "unsupported_fact_content"} <= _codes(outcome)


def test_construct_bypass_and_extra_policy_fields_cannot_raise_sufficiency(validator) -> None:
    from trade_agent.evidence.requirements import SufficiencyPolicy

    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    valid = _rag(suffix="construct")
    forged_values = {name: getattr(valid, name) for name in Evidence.model_fields}
    forged_values["content"] = "Acme synthetic demonstration page."
    forged_values["source_weight"] = 999.0
    forged = Evidence.model_construct(**forged_values)

    outcome = validator.validate(intent, (forged,), (), AS_OF)

    assert outcome.can_answer is False
    assert "invalid_contract" in _codes(outcome)
    with pytest.raises(ValidationError):
        SufficiencyPolicy.model_validate(
            {**SufficiencyPolicy().model_dump(), "gold_label": "accept"}
        )
    with pytest.raises(ValidationError):
        SufficiencyPolicy().model_copy(update={"lead_status_independent_sources": 0})


def test_validation_is_order_independent_strict_frozen_and_json_round_trippable(validator) -> None:
    from trade_agent.evidence.validator import ValidationOutcome

    intent = _intent("Acme 是否值得跟进", filters=RetrievalFilter(entity_ids=("company:acme",)))
    evidence = (_sql(start=date(2025, 9, 4), scope="lead"), _rag(suffix="one"), _rag(suffix="two", source_type="industry_news"))
    forward = validator.validate(intent, evidence, (), AS_OF)
    reverse = validator.validate(intent, tuple(reversed(evidence)), (), AS_OF)

    assert forward == reverse
    assert ValidationOutcome.model_validate_json(forward.model_dump_json()) == forward
    with pytest.raises(ValidationError):
        ValidationOutcome.model_validate({**forward.model_dump(), "gold_label": "answer"})
    with pytest.raises(ValidationError):
        forward.can_answer = False  # type: ignore[misc]
    serialized = forward.model_dump_json()
    for forbidden in ("gold_label", "reference_claim", "expected_answer"):
        assert forbidden not in serialized


def test_out_of_scope_intent_refuses_without_considering_evidence(validator) -> None:
    intent = _intent("写一首诗")
    outcome = validator.validate(intent, (_rag(suffix="irrelevant"),), (), AS_OF)

    assert outcome.can_answer is False
    assert outcome.decision == "refuse"
    assert outcome.error_code == "out_of_scope"
    assert _codes(outcome) == {"out_of_scope"}
