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
    metric_value: object = Decimal("12.30"),
    result_rows: tuple[dict[str, object], ...] | None = None,
    where_override: str | None = None,
    filter_names_override: tuple[str, ...] | None = None,
    select_operation_modifier: str = "",
    trade_records_hint: str = "",
    company_join_keyword: str = "JOIN",
) -> Evidence:
    dimensions: dict[str, object] = {}
    if currency is not None:
        dimensions["currency"] = currency
    if unit is not None:
        dimensions["unit"] = unit
    if time_grain == "month":
        dimensions["trade_date"] = "2026-08"
    row = {"importer_company": "Acme", **dimensions, metric: metric_value}
    effective_rows = result_rows if result_rows is not None else (row,) if rows else ()
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
    where = where_override or where
    filter_names = filter_names if filter_names_override is None else filter_names_override
    result = SqlExecutionResult(
        query_id="sqlq_" + "a" * 64,
        normalized_sql=(
            f"SELECT {select_operation_modifier}importer.company_name AS importer_company, {metric}, "
            f"import_country.country_code, hs.hs_code FROM trade_records AS tr{trade_records_hint} "
            f"{company_join_keyword} companies AS importer ON tr.importer_id = importer.id "
            "JOIN countries AS import_country ON tr.import_country_id = import_country.id "
            "JOIN hs_codes AS hs ON tr.hs_code_id = hs.id "
            "JOIN data_sources AS data_scope ON tr.source_id = data_scope.id "
            "WHERE data_scope.is_synthetic = :policy_is_synthetic "
            "AND tr.trade_date BETWEEN :policy_start_date AND :policy_end_date "
            f"AND {where} LIMIT 50"
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
        rows=effective_rows,
        row_count=len(effective_rows),
        result_hash=ReadOnlySqlExecutor._hash_rows(effective_rows),
        raw_record_locators=(RawRecordLocator(source_id=7, raw_record_id="ROW-9"),) if effective_rows else (),
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
    content: str | None = None,
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
    conflict_group_id: str | None = None,
    currencies: tuple[str, ...] = (),
    units: tuple[str, ...] = (),
    aggregation_grain: tuple[str, ...] = (),
) -> Evidence:
    if content is None:
        content = f"Acme opened audited production line {suffix} and remains operational."
    domains = {
        "b2b": "marketplace.example",
        "customs_profile": "profiles.example",
        "industry_news": "news.example",
        "official_website": "official.example",
        "regulator": "regulator.example",
        "social": "social.example",
    }
    source = source_id or f"https://{domains[source_type]}/status/{suffix}"
    public_url = source if source.startswith(("http://", "https://")) else None
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
        source_url=public_url,
        canonical_url=public_url,
        locator=locator,
        raw_record_id=f"record-{suffix}",
        publish_time=publish_time,
        valid_from=valid_from,
        valid_to=valid_to,
        time_grain="document",
        currencies=currencies,
        units=units,
        aggregation_grain=aggregation_grain,
        confidence=1.0,
        confidence_basis="content_hash_verified",
        is_synthetic=True,
        retrieval_provenance=provenance,
        sql_provenance=None,
        conflict_group_id=conflict_group_id,
    )
    provisional = Evidence.model_construct(evidence_id="rag_" + "0" * 64, **values)
    return Evidence(evidence_id=retrieval_evidence_id(identity=_evidence_identity(provisional)), **values)


@pytest.fixture
def validator():
    from trade_agent.evidence.validator import EvidenceValidator

    class ContextualValidator(EvidenceValidator):
        def validate(self, intent, evidence, conflicts, as_of, *, context=None):
            return super().validate(
                intent, evidence, conflicts, as_of,
                context=context if context is not None else _context_for(intent, evidence),
            )

    return ContextualValidator()


def _context_for(intent: QueryIntent, evidence: tuple[Evidence, ...]):
    from trade_agent.evidence.validator import BranchExecutionReport, ValidationContext

    branches = []
    if intent.need_external_intel:
        branches.append("rag")
    if intent.need_trade_data:
        branches.append("sql")
    reports = []
    for branch in sorted(branches):
        branch_evidence = tuple(item for item in evidence if item.locator.branch == branch)
        observed = {component for item in branch_evidence if item.retrieval_provenance for component in item.retrieval_provenance.degraded_components}
        fatal = observed & {
            "all_retrievers_unavailable", "branch_timeout", "required_branch_timeout",
            "retrieval_timeout", "retrieval_unavailable", "sql_timeout", "sql_unavailable",
        }
        degraded = tuple(sorted(observed - fatal))
        errors = ()
        if any("timeout" in item for item in fatal):
            errors = ("timeout",)
        elif fatal:
            errors = ("unavailable",)
        reports.append(BranchExecutionReport(branch=branch, attempted=True, completed=True,
            zero_hits=not branch_evidence, degraded_components=degraded, error_codes=errors))
    return ValidationContext(branch_reports=tuple(reports))


def _validated(intent, evidence, conflicts=(), as_of=AS_OF, *, policy=None):
    from trade_agent.evidence.validator import EvidenceValidator

    validator = EvidenceValidator() if policy is None else EvidenceValidator(policy)
    return validator.validate(intent, evidence, conflicts, as_of, context=_context_for(intent, evidence))


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

    accepted = _validated(
        intent,
        (_rag(suffix="exact-age", publish_time=datetime.combine(exact, datetime.min.time(), timezone.utc), valid_from=exact),),
        (),
        AS_OF,
    )
    rejected = _validated(
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
    from trade_agent.evidence.requirements import FactSourcePolicy, SourceAuthorityRule, SufficiencyPolicy
    from trade_agent.evidence.validator import EvidenceValidator

    base = SufficiencyPolicy()
    sources = tuple(
        FactSourcePolicy(fact_type=item.fact_type, allowed_source_types=("social",))
        if item.fact_type == "company_status"
        else item
        for item in base.fact_sources
    )
    policy = base.model_copy(update={
        "fact_sources": sources,
        "authority_rules": tuple(sorted((*base.authority_rules, SourceAuthorityRule(
            rule_id="social.reviewed", publisher_id="social-demo", source_type="social",
            domain_suffixes=("social.example",), directness="secondary",
        )), key=lambda item: item.rule_id)),
    })
    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    social = _rag(suffix="policy-social", source_type="social")

    outcome = _validated(intent, (social,), (), AS_OF, policy=policy)

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
    conflict_id = "conflict_" + "3" * 64
    left = _rag(suffix="left", conflict_group_id=conflict_id)
    right = _rag(suffix="right", source_type="industry_news", content="Acme permanently closed operations.", conflict_group_id=conflict_id)
    conflict = Conflict(
        conflict_id=conflict_id,
        entity_id="company:acme",
        fact_type="company_status",
        evidence_ids=(left.evidence_id, right.evidence_id),
        status="unresolved",
        valid_from=date(2026, 8, 20),
        explanation="Current sources disagree.",
    )

    outcome = validator.validate(intent, (left, right), (conflict,), AS_OF)

    assert outcome.can_answer is False
    assert outcome.decision == "refuse"
    assert outcome.error_code == "evidence_conflict"
    assert "contradictory" in _codes(outcome)


def test_resolved_conflict_uses_only_selected_support_and_is_order_independent(validator) -> None:
    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    conflict_id = "conflict_" + "4" * 64
    selected = _rag(suffix="selected", conflict_group_id=conflict_id)
    rejected = _rag(suffix="rejected", source_type="industry_news", content="Acme permanently closed operations.", conflict_group_id=conflict_id)
    conflict = Conflict(
        conflict_id=conflict_id,
        entity_id="company:acme",
        fact_type="company_status",
        evidence_ids=tuple(sorted((selected.evidence_id, rejected.evidence_id))),
        status="resolved",
        valid_from=date(2026, 8, 20),
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


def test_review_lead_explicit_risk_cannot_replace_mandatory_status(validator) -> None:
    intent = _intent(
        "Acme 是否值得跟进",
        filters=RetrievalFilter(
            entity_ids=("company:acme",), fact_types=("risk",),
        ),
    )
    risk = (
        _rag(suffix="risk-one", fact_type="risk", content="Acme reports a current risk."),
        _rag(
            suffix="risk-two", fact_type="risk", source_type="industry_news",
            content="Acme faces a current sanction risk.",
        ),
    )

    outcome = validator.validate(
        intent, (_sql(start=date(2025, 9, 4), scope="lead"), *risk), (), AS_OF
    )

    assert outcome.can_answer is False
    assert any(item.requirement_id == "rag.company_status" for item in outcome.missing_requirements)


def test_review_forged_route_flags_and_zero_requirements_fail_closed(validator) -> None:
    valid = _intent("Acme 官网最近状态")
    forged = QueryIntent.model_construct(
        **{
            **{name: getattr(valid, name) for name in QueryIntent.model_fields},
            "need_external_intel": False,
        }
    )

    outcome = validator.validate(forged, (), (), AS_OF)

    assert outcome.can_answer is False
    assert outcome.error_code == "invalid_intent"
    assert "invalid_intent" in _codes(outcome)


def test_review_all_requested_entities_and_fact_types_need_coverage(validator) -> None:
    entities = _intent(
        "官网最近状态",
        filters=RetrievalFilter(entity_ids=("company:a", "company:b")),
    )
    only_a = _rag(
        suffix="only-a", entity_id="company:a", company_name="A",
        content="A opened a production line and remains operational.",
    )
    entity_outcome = validator.validate(entities, (only_a,), (), AS_OF)

    facts = _intent(
        "查询风险和状态",
        filters=RetrievalFilter(fact_types=("company_status", "risk")),
    )
    only_status = _rag(suffix="only-status")
    fact_outcome = validator.validate(facts, (only_status,), (), AS_OF)

    assert entity_outcome.can_answer is False
    assert fact_outcome.can_answer is False
    assert len(entity_outcome.missing_requirements) >= 1
    assert any(item.requirement_id.startswith("rag.risk") for item in fact_outcome.missing_requirements)


@pytest.mark.parametrize("value", [None, "NaN", "Infinity", True])
def test_review_sql_metric_value_must_be_nonnull_finite_numeric(validator, value) -> None:
    intent = _intent("最近半年美国采购 HS850440 金额最高的 10 家公司")

    outcome = validator.validate(intent, (_sql(metric_value=value),), (), AS_OF)

    assert outcome.can_answer is False
    assert "invalid_metric" in _codes(outcome)


def test_review_unsupported_or_multiple_metrics_are_invalid_intents(validator) -> None:
    from trade_agent.db.contracts import QueryConstraints

    base = _intent("最近半年美国采购 HS850440 金额最高的 10 家公司")
    unsupported = QueryIntent.model_construct(
        **{
            **{name: getattr(base, name) for name in QueryIntent.model_fields},
            "constraints": QueryConstraints(metrics=["profit_margin"], dimensions=[], filters=[]),
        }
    )
    multiple = QueryIntent.model_construct(
        **{
            **{name: getattr(base, name) for name in QueryIntent.model_fields},
            "constraints": QueryConstraints(
                metrics=["trade_amount", "quantity"], dimensions=[], filters=[]
            ),
        }
    )

    assert validator.validate(unsupported, (), (), AS_OF).error_code == "invalid_intent"
    assert validator.validate(multiple, (), (), AS_OF).error_code == "invalid_intent"


def test_review_branch_reports_prevent_clean_hit_dilution_and_describe_zero_hit_retry() -> None:
    from trade_agent.evidence.validator import (
        BranchExecutionReport,
        EvidenceValidator,
        ValidationContext,
    )

    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    fatal_context = ValidationContext(
        branch_reports=(
            BranchExecutionReport(
                branch="rag", attempted=True, completed=False, zero_hits=False,
                degraded_components=(), error_codes=("timeout",),
            ),
        )
    )
    clean_hit = EvidenceValidator().validate(
        intent, (_rag(suffix="clean-but-timeout"),), (), AS_OF,
        context=fatal_context,
    )
    zero_context = ValidationContext(
        branch_reports=(
            BranchExecutionReport(
                branch="rag", attempted=True, completed=True, zero_hits=True,
                degraded_components=(),
                error_codes=("unavailable",),
            ),
        )
    )
    zero = EvidenceValidator().validate(intent, (), (), AS_OF, context=zero_context)
    missing_report = EvidenceValidator().validate(intent, (_rag(suffix="no-report"),), (), AS_OF)

    assert clean_hit.decision == "rewrite_once"
    assert zero.decision == "rewrite_once"
    assert missing_report.decision == "refuse"
    assert "branch_report_missing" in _codes(missing_report)


def test_review_future_subday_publish_and_future_valid_from_are_rejected(validator) -> None:
    as_of = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    future_publish = _rag(
        suffix="future-hour",
        publish_time=datetime(2026, 9, 4, 12, 0, 1, tzinfo=timezone.utc),
        valid_from=date(2026, 9, 4),
    )
    future_valid = _rag(
        suffix="future-valid",
        publish_time=datetime(2026, 9, 4, 11, 0, tzinfo=timezone.utc),
        valid_from=date(2026, 9, 5),
    )

    assert not validator.validate(intent, (future_publish,), (), as_of).can_answer
    valid_outcome = validator.validate(intent, (future_valid,), (), as_of)
    assert not valid_outcome.can_answer
    assert "out_of_scope" in _codes(valid_outcome)


def test_review_same_content_with_forged_clusters_is_not_independent(validator) -> None:
    intent = _intent("Acme 是否值得跟进", filters=RetrievalFilter(entity_ids=("company:acme",)))
    content = "Acme opened a new audited production line and remains operational."
    first = _rag(suffix="copy-one", content=content, cluster="claimed-one")
    second = _rag(
        suffix="copy-two", content=content, cluster="claimed-two",
        source_type="industry_news",
    )

    outcome = validator.validate(
        intent, (_sql(start=date(2025, 9, 4), scope="lead"), first, second), (), AS_OF
    )

    assert outcome.can_answer is False
    assert "insufficient_diversity" in _codes(outcome)


def test_review_source_type_alone_and_official_urn_are_not_authority(validator) -> None:
    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    unreviewed = _rag(
        suffix="unreviewed", source_id="https://attacker.invalid/status",
        source_type="official_website",
    )
    official_urn = _rag(
        suffix="urn", source_id="urn:trade-agent:document:doc-urn",
        source_type="official_website",
    )

    assert "low_authority" in _codes(validator.validate(intent, (unreviewed,), (), AS_OF))
    assert "low_authority" in _codes(validator.validate(intent, (official_urn,), (), AS_OF))


def test_review_conflict_group_must_match_conflict_and_selected_support(validator) -> None:
    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    evidence = _rag(suffix="conflict-mismatch")
    conflict = Conflict(
        conflict_id="conflict_" + "5" * 64,
        entity_id="company:acme", fact_type="company_status",
        evidence_ids=(evidence.evidence_id, "rag_" + "6" * 64),
        status="resolved", explanation="Claims to resolve unrelated Evidence.",
        selected_evidence_ids=(evidence.evidence_id,),
    )

    outcome = validator.validate(intent, (evidence,), (conflict,), AS_OF)

    assert outcome.can_answer is False
    assert "conflict_mismatch" in _codes(outcome)


def test_review_validation_outcome_rejects_forged_partition_and_eligible_ids(validator) -> None:
    from trade_agent.evidence.validator import ValidationOutcome

    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    outcome = validator.validate(intent, (_rag(suffix="outcome"),), (), AS_OF)
    forged = outcome.model_dump(mode="python")
    forged["satisfied_requirements"] = ()

    with pytest.raises(ValidationError):
        ValidationOutcome.model_validate(forged)
    forged = outcome.model_dump(mode="python")
    forged["eligible_evidence_ids"] = ()
    with pytest.raises(ValidationError):
        ValidationOutcome.model_validate(forged)


def test_review_policy_fingerprint_changes_with_semantics_and_rejects_tamper() -> None:
    from trade_agent.evidence.requirements import SufficiencyPolicy

    original = SufficiencyPolicy()
    changed = original.model_copy(update={"lead_status_independent_sources": 3})

    assert original.policy_version == changed.policy_version
    assert original.policy_fingerprint != changed.policy_fingerprint
    forged = changed.model_dump(mode="python")
    forged["policy_fingerprint"] = original.policy_fingerprint
    with pytest.raises(ValidationError):
        SufficiencyPolicy.model_validate(forged)


def test_round2_branch_taxonomy_is_closed_coherent_and_hard_errors_refuse() -> None:
    from trade_agent.evidence.validator import BranchExecutionReport, EvidenceValidator, ValidationContext

    with pytest.raises(ValidationError):
        BranchExecutionReport(
            branch="rag", attempted=True, completed=False, zero_hits=False,
            error_codes=("invented_error",),
        )
    with pytest.raises(ValidationError):
        BranchExecutionReport(
            branch="rag", attempted=False, completed=False, zero_hits=False,
            error_codes=(),
        )
    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    context = ValidationContext(branch_reports=(BranchExecutionReport(
        branch="rag", attempted=True, completed=False, zero_hits=False,
        error_codes=("policy_denied",),
    ),))
    outcome = EvidenceValidator().validate(
        intent, (_rag(suffix="hard-error"),), (), AS_OF, context=context,
    )

    assert outcome.decision == "refuse"
    assert "branch_hard_error" in _codes(outcome)


def test_round2_sql_scope_requires_exact_columns_placeholders_and_no_unused() -> None:
    intent = _intent(
        "最近半年美国采购 HS850440 金额最高的 10 家公司",
        filters=RetrievalFilter(entity_ids=("company:one", "company:two")),
    )
    wrong_alias = _sql(
        where_override=(
            "exporter.id IN (:filter_0_0, :filter_0_1) "
            "AND import_country.country_code = :filter_1_0 "
            "AND hs.hs_code = :filter_2_0 "
            "AND tr.trade_date BETWEEN :filter_3_start AND :filter_3_end"
        ),
        filter_names_override=(
            "filter_0_0", "filter_0_1", "filter_1_0", "filter_2_0",
            "filter_3_end", "filter_3_start",
        ),
    )
    unused = _sql(filter_names_override=(
        "filter_0_0", "filter_1_0", "filter_2_end", "filter_2_start", "filter_9_0",
    ))
    disguised_unused = _sql(
        where_override=(
            "import_country.country_code = :filter_0_0 "
            "AND hs.hs_code = :filter_1_0 "
            "AND tr.trade_date BETWEEN :filter_2_start AND :filter_2_end "
            "AND importer.id = :misc_scope"
        ),
        filter_names_override=(
            "filter_0_0", "filter_1_0", "filter_2_end", "filter_2_start",
            "misc_scope",
        ),
    )

    assert "dimension_mismatch" in _codes(validator := _validated(intent, (wrong_alias,)))
    assert not validator.can_answer
    assert "invalid_contract" in _codes(_validated(
        _intent("最近半年美国采购 HS850440 金额最高的 10 家公司"), (unused,),
    ))
    assert "invalid_contract" in _codes(_validated(
        _intent("最近半年美国采购 HS850440 金额最高的 10 家公司"),
        (disguised_unused,),
    ))


def test_round2_sql_scope_accepts_all_atomic_values_on_canonical_columns() -> None:
    intent = _intent(
        "最近半年美国采购 HS850440 金额最高的 10 家公司",
        filters=RetrievalFilter(entity_ids=("company:one", "company:two")),
    )
    evidence = _sql(
        where_override=(
            "importer.id IN (:filter_0_0, :filter_0_1) "
            "AND import_country.country_code = :filter_1_0 "
            "AND hs.hs_code = :filter_2_0 "
            "AND tr.trade_date BETWEEN :filter_3_start AND :filter_3_end"
        ),
        filter_names_override=(
            "filter_0_0", "filter_0_1", "filter_1_0", "filter_2_0",
            "filter_3_end", "filter_3_start",
        ),
    )

    assert _validated(intent, (evidence,)).can_answer is True


def test_round2_date_as_of_is_end_of_day_but_datetime_is_exact(validator) -> None:
    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    same_day = _rag(
        suffix="same-day-evening",
        publish_time=datetime(2026, 9, 4, 23, 59, 59, tzinfo=timezone.utc),
        valid_from=datetime(2026, 9, 4, 23, 59, 59, tzinfo=timezone.utc),
    )

    assert validator.validate(intent, (same_day,), (), AS_OF).can_answer is True
    exact_noon = validator.validate(
        intent, (same_day,), (), datetime(2026, 9, 4, 12, tzinfo=timezone.utc),
    )
    assert not exact_noon.can_answer and "out_of_scope" in _codes(exact_noon)


def test_round2_same_publisher_pages_are_not_independent_and_policy_axes_execute(validator) -> None:
    from trade_agent.evidence.requirements import SufficiencyPolicy

    intent = _intent("Acme 是否值得跟进", filters=RetrievalFilter(entity_ids=("company:acme",)))
    pages = (
        _rag(suffix="publisher-page-one", source_id="https://official.example/page/one"),
        _rag(suffix="publisher-page-two", source_id="https://official.example/page/two"),
    )
    evidence = (_sql(start=date(2025, 9, 4), scope="lead"), *pages)
    strict = validator.validate(intent, evidence, (), AS_OF)
    cluster_only = SufficiencyPolicy().model_copy(
        update={"independence_dimensions": ("dedupe_cluster_id",)}
    )
    relaxed = _validated(intent, evidence, policy=cluster_only)

    assert not strict.can_answer and "insufficient_diversity" in _codes(strict)
    assert relaxed.can_answer is True
    assert strict.policy_fingerprint != relaxed.policy_fingerprint


def test_round2_conflict_scope_currency_unit_grain_and_validity_must_close(validator) -> None:
    conflict_id = "conflict_" + "7" * 64
    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    first = _rag(suffix="scope-one", conflict_group_id=conflict_id)
    second = _rag(
        suffix="scope-two", source_type="industry_news", conflict_group_id=conflict_id,
    )
    conflict = Conflict(
        conflict_id=conflict_id, entity_id="company:acme", fact_type="company_status",
        evidence_ids=tuple(sorted((first.evidence_id, second.evidence_id))),
        status="resolved", currency="USD", unit="kg",
        aggregation_grain=("month",), valid_from=date(2026, 8, 1),
        explanation="Scope does not match either Evidence item.",
        selected_evidence_ids=(first.evidence_id,),
    )

    outcome = validator.validate(intent, (first, second), (conflict,), AS_OF)

    assert not outcome.can_answer
    assert "conflict_mismatch" in _codes(outcome)

    hidden = _rag(suffix="hidden-status", conflict_group_id=conflict_id)
    risk_left = _rag(
        suffix="risk-left", fact_type="risk", content="Acme reported an active recall risk.",
        conflict_group_id=conflict_id,
    )
    risk_right = _rag(
        suffix="risk-right", fact_type="risk", content="Acme reported no recall risk.",
        source_type="industry_news", conflict_group_id=conflict_id,
    )
    unrelated = Conflict(
        conflict_id=conflict_id, entity_id="company:acme", fact_type="risk",
        evidence_ids=tuple(sorted((risk_left.evidence_id, risk_right.evidence_id))),
        status="resolved", valid_from=date(2026, 8, 20),
        explanation="The risk report does not cover the hidden status Evidence.",
        selected_evidence_ids=(risk_left.evidence_id,),
    )
    hidden_outcome = validator.validate(
        intent, (hidden, risk_left, risk_right), (unrelated,), AS_OF,
    )

    assert not hidden_outcome.can_answer
    assert "conflict_mismatch" in _codes(hidden_outcome)


def test_round2_requirements_and_result_branches_reject_joint_forgery(validator) -> None:
    from trade_agent.evidence.requirements import EvidenceRequirements
    from trade_agent.evidence.validator import ValidationOutcome

    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    outcome = validator.validate(intent, (_rag(suffix="fingerprinted"),), (), AS_OF)
    forged_requirements = outcome.requirements.model_dump(mode="python")
    forged_requirements["items"][0]["maximum_age_days"] = 9_999
    with pytest.raises(ValidationError):
        EvidenceRequirements.model_validate(forged_requirements)
    forged_outcome = outcome.model_dump(mode="python")
    forged_outcome["satisfied_requirements"][0]["branch"] = "sql"
    with pytest.raises(ValidationError):
        ValidationOutcome.model_validate(forged_outcome)
    insufficient = validator.validate(intent, (), (), AS_OF)
    retryable_refusal = insufficient.model_dump(mode="python")
    retryable_refusal["error_code"] = "evidence_retryable"
    with pytest.raises(ValidationError):
        ValidationOutcome.model_validate(retryable_refusal)


def test_round2_sql_rows_require_complete_currency_or_unit_scope(validator) -> None:
    amount = _intent("最近半年美国采购 HS850440 金额最高的 10 家公司")
    amount_rows = (
        {"importer_company": "Acme", "currency": "USD", "trade_amount": "12.30"},
        {"importer_company": "Beta", "currency": None, "trade_amount": "10.00"},
    )
    quantity = _intent("中国出口 HS850440 的月度数量")
    quantity_rows = (
        {"importer_company": "Acme", "trade_date": "2026-08", "unit": "kg", "quantity": "12"},
        {"importer_company": "Beta", "trade_date": "2026-08", "unit": None, "quantity": "10"},
    )

    amount_outcome = validator.validate(
        amount, (_sql(result_rows=amount_rows),), (), AS_OF,
    )
    quantity_outcome = validator.validate(
        quantity,
        (_sql(metric="quantity", currency=None, unit="kg", time_grain="month",
              aggregation_grain=("trade_date", "unit"), scope="quantity",
              result_rows=quantity_rows),),
        (), AS_OF,
    )

    assert "currency_missing" in _codes(amount_outcome)
    assert "unit_missing" in _codes(quantity_outcome)


def test_round2_partial_degradation_mode_changes_fingerprint_and_result() -> None:
    from trade_agent.evidence.requirements import SufficiencyPolicy

    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    evidence = (_rag(suffix="partial-policy"),)
    audit_policy = SufficiencyPolicy()
    block_policy = audit_policy.model_copy(update={"partial_degradation_mode": "block"})
    from trade_agent.evidence.validator import BranchExecutionReport, EvidenceValidator, ValidationContext
    context = ValidationContext(branch_reports=(BranchExecutionReport(
        branch="rag", attempted=True, completed=True, zero_hits=False,
        degraded_components=("dense_unavailable",), error_codes=(),
    ),))

    audit = EvidenceValidator(audit_policy).validate(
        intent, evidence, (), AS_OF, context=context,
    )
    blocked = EvidenceValidator(block_policy).validate(
        intent, evidence, (), AS_OF, context=context,
    )

    assert audit.can_answer is True
    assert blocked.decision == "refuse"
    assert "partial_degradation" in _codes(blocked)
    assert audit.policy_fingerprint != blocked.policy_fingerprint


@pytest.mark.parametrize("where", (
    (
        "import_country.country_code = :filter_0_0 "
        "AND 1 = 1 /* hs.hs_code = :filter_1_0 */ "
        "AND tr.trade_date BETWEEN :filter_2_start AND :filter_2_end"
    ),
    (
        "import_country.country_code = :filter_0_0 "
        "AND 'hs.hs_code = :filter_1_0' "
        "AND tr.trade_date BETWEEN :filter_2_start AND :filter_2_end"
    ),
    (
        "import_country.country_code = :filter_0_0 "
        "OR hs.hs_code = :filter_1_0 "
        "AND tr.trade_date BETWEEN :filter_2_start AND :filter_2_end"
    ),
))
def test_round3_sql_scope_rejects_comment_string_and_or_pseudo_predicates(where: str) -> None:
    evidence = _sql(where_override=where)

    outcome = _validated(
        _intent("最近半年美国采购 HS850440 金额最高的 10 家公司"),
        (evidence,),
    )

    assert not outcome.can_answer
    assert "invalid_contract" in _codes(outcome)


def test_round3_sql_scope_rejects_in_literal_even_with_exact_placeholder_count() -> None:
    intent = _intent(
        "最近半年美国采购 HS850440 金额最高的 10 家公司",
        filters=RetrievalFilter(entity_ids=("company:one", "company:two")),
    )
    evidence = _sql(
        where_override=(
            "importer.id IN (:filter_0_0, :filter_0_1, 999) "
            "AND import_country.country_code = :filter_1_0 "
            "AND hs.hs_code = :filter_2_0 "
            "AND tr.trade_date BETWEEN :filter_3_start AND :filter_3_end"
        ),
        filter_names_override=(
            "filter_0_0", "filter_0_1", "filter_1_0", "filter_2_0",
            "filter_3_end", "filter_3_start",
        ),
    )

    outcome = _validated(intent, (evidence,))

    assert not outcome.can_answer
    assert "invalid_contract" in _codes(outcome)

    duplicate_eq = _sql(
        where_override=(
            "importer.id = :filter_0_0 AND importer.id = :filter_0_1 "
            "AND import_country.country_code = :filter_1_0 "
            "AND hs.hs_code = :filter_2_0 "
            "AND tr.trade_date BETWEEN :filter_3_start AND :filter_3_end"
        ),
        filter_names_override=(
            "filter_0_0", "filter_0_1", "filter_1_0", "filter_2_0",
            "filter_3_end", "filter_3_start",
        ),
    )
    duplicate_outcome = _validated(intent, (duplicate_eq,))

    assert not duplicate_outcome.can_answer
    assert "invalid_contract" in _codes(duplicate_outcome)


@pytest.mark.parametrize("sql_options", (
    {"trade_records_hint": " USE INDEX (idx_trade_date)"},
    {"trade_records_hint": " FORCE INDEX (idx_trade_date)"},
    {"select_operation_modifier": "STRAIGHT_JOIN "},
    {"company_join_keyword": "STRAIGHT_JOIN"},
))
def test_round4_sql_scope_rejects_table_hints_and_operation_modifiers(
    sql_options: dict[str, str],
) -> None:
    evidence = _sql(**sql_options)

    outcome = _validated(
        _intent("最近半年美国采购 HS850440 金额最高的 10 家公司"),
        (evidence,),
    )

    assert not outcome.can_answer
    assert "invalid_contract" in _codes(outcome)


def test_round3_policy_cannot_reclassify_deterministic_errors_as_retryable() -> None:
    from trade_agent.evidence.requirements import SufficiencyPolicy
    from trade_agent.evidence.validator import BranchExecutionReport, EvidenceValidator, ValidationContext

    with pytest.raises(ValidationError):
        SufficiencyPolicy().model_copy(update={
            "fatal_error_codes": ("policy_denied", "timeout", "transport", "unavailable"),
        })
    timeout_only = SufficiencyPolicy().model_copy(update={"fatal_error_codes": ("timeout",)})
    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    transport = ValidationContext(branch_reports=(BranchExecutionReport(
        branch="rag", attempted=True, completed=False, zero_hits=False,
        error_codes=("transport",),
    ),))
    outcome = EvidenceValidator(timeout_only).validate(
        intent, (_rag(suffix="transport-not-retryable"),), (), AS_OF, context=transport,
    )

    assert outcome.decision == "refuse"
    assert "branch_hard_error" in _codes(outcome)


def test_round3_requirements_fingerprint_is_mandatory_on_json_input() -> None:
    from trade_agent.evidence.requirements import EvidenceRequirements

    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    payload = EvidenceRequirements.for_intent(intent).model_dump(mode="python")
    payload.pop("requirements_fingerprint")

    with pytest.raises(ValidationError):
        EvidenceRequirements.model_validate(payload)


def test_round3_outcome_requires_exact_missing_reasons_and_closed_source_count(validator) -> None:
    from trade_agent.evidence.validator import ValidationOutcome

    intent = _intent("Acme 官网最近状态", filters=RetrievalFilter(entity_ids=("company:acme",)))
    insufficient = validator.validate(intent, (), (), AS_OF)
    missing_forgery = insufficient.model_dump(mode="python")
    assert len(missing_forgery["missing_requirements"][0]["reason_codes"]) > 1
    missing_forgery["missing_requirements"][0]["reason_codes"] = (
        missing_forgery["missing_requirements"][0]["reason_codes"][0],
    )
    with pytest.raises(ValidationError):
        ValidationOutcome.model_validate(missing_forgery)

    answer = validator.validate(intent, (_rag(suffix="count-proof"),), (), AS_OF)
    count_forgery = answer.model_dump(mode="python")
    count_forgery["satisfied_requirements"][0]["independent_source_count"] = 999
    with pytest.raises(ValidationError):
        ValidationOutcome.model_validate(count_forgery)
