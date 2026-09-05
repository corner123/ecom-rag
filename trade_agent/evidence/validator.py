"""Deterministic, score-independent validation of evidence coverage."""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import json
import re
from typing import Literal, Mapping, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from trade_agent.agents.intent import QueryIntent
from trade_agent.evidence.models import Conflict, Evidence
from trade_agent.evidence.requirements import (
    DEFAULT_SUFFICIENCY_POLICY,
    EvidenceRequirement,
    EvidenceRequirements,
    SufficiencyPolicy,
)


ReasonCode = Literal[
    "contradictory",
    "currency_missing",
    "degraded_branch",
    "dimension_mismatch",
    "empty_result",
    "fact_type_mismatch",
    "grain_mismatch",
    "insufficient_diversity",
    "invalid_contract",
    "low_authority",
    "metric_mismatch",
    "missing",
    "missing_locator",
    "out_of_scope",
    "stale",
    "temporal_gap",
    "truncated_locator",
    "unit_missing",
    "unknown_fact",
    "unresolved_entity",
    "unsupported_fact_content",
]


_REASON_ORDER = {
    code: index
    for index, code in enumerate(
        (
            "out_of_scope",
            "contradictory",
            "invalid_contract",
            "missing",
            "degraded_branch",
            "empty_result",
            "missing_locator",
            "truncated_locator",
            "unresolved_entity",
            "unknown_fact",
            "fact_type_mismatch",
            "unsupported_fact_content",
            "low_authority",
            "dimension_mismatch",
            "metric_mismatch",
            "grain_mismatch",
            "currency_missing",
            "unit_missing",
            "temporal_gap",
            "stale",
            "insufficient_diversity",
        )
    )
}


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def model_copy(self, *, update: Mapping[str, object] | None = None, deep: bool = False) -> Self:
        if update is None:
            return super().model_copy(deep=deep)
        values = self.model_dump(mode="python")
        if deep:
            values = deepcopy(values)
        values.update(update)
        return type(self).model_validate(values)


class ValidationReason(_Contract):
    code: ReasonCode
    requirement_id: StrictStr | None
    branch: Literal["sql", "rag"] | None
    evidence_ids: tuple[StrictStr, ...] = ()
    blocking: StrictBool
    detail: StrictStr

    @model_validator(mode="after")
    def stable(self) -> Self:
        if self.evidence_ids != tuple(sorted(set(self.evidence_ids))):
            raise ValueError("reason evidence IDs must be sorted and unique")
        if not self.detail.strip():
            raise ValueError("reason detail must not be blank")
        return self


class SatisfiedRequirement(_Contract):
    requirement_id: StrictStr
    branch: Literal["sql", "rag"]
    evidence_ids: tuple[StrictStr, ...]
    independent_source_count: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def stable(self) -> Self:
        if self.evidence_ids != tuple(sorted(set(self.evidence_ids))) or not self.evidence_ids:
            raise ValueError("satisfied evidence IDs must be nonempty, sorted and unique")
        return self


class MissingRequirement(_Contract):
    requirement_id: StrictStr
    branch: Literal["sql", "rag"]
    reason_codes: tuple[ReasonCode, ...]
    considered_evidence_ids: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def stable(self) -> Self:
        if self.reason_codes != tuple(sorted(set(self.reason_codes), key=_REASON_ORDER.__getitem__)):
            raise ValueError("missing reason codes must be sorted and unique")
        if self.considered_evidence_ids != tuple(sorted(set(self.considered_evidence_ids))):
            raise ValueError("considered evidence IDs must be sorted and unique")
        return self


class ValidationOutcome(_Contract):
    can_answer: StrictBool
    decision: Literal["answer", "rewrite_once", "refuse"]
    error_code: Literal[
        "evidence_conflict", "evidence_insufficient", "evidence_retryable", "out_of_scope"
    ] | None
    requirements: EvidenceRequirements
    satisfied_requirements: tuple[SatisfiedRequirement, ...]
    missing_requirements: tuple[MissingRequirement, ...]
    reasons: tuple[ValidationReason, ...]
    eligible_evidence_ids: tuple[StrictStr, ...]
    excluded_evidence_ids: tuple[StrictStr, ...]
    conflict_ids: tuple[StrictStr, ...]
    degraded_components: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.can_answer != (self.decision == "answer"):
            raise ValueError("can_answer must agree with decision")
        if self.can_answer != (self.error_code is None):
            raise ValueError("answerable outcomes cannot carry an error")
        for values, field in (
            (self.eligible_evidence_ids, "eligible_evidence_ids"),
            (self.excluded_evidence_ids, "excluded_evidence_ids"),
            (self.conflict_ids, "conflict_ids"),
            (self.degraded_components, "degraded_components"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field} must be sorted and unique")
        if set(self.eligible_evidence_ids) & set(self.excluded_evidence_ids):
            raise ValueError("eligible and excluded Evidence must be disjoint")
        requirement_ids = tuple(item.requirement_id for item in self.satisfied_requirements)
        missing_ids = tuple(item.requirement_id for item in self.missing_requirements)
        if requirement_ids != tuple(sorted(set(requirement_ids))):
            raise ValueError("satisfied requirements must be sorted and unique")
        if missing_ids != tuple(sorted(set(missing_ids))):
            raise ValueError("missing requirements must be sorted and unique")
        if set(requirement_ids) & set(missing_ids):
            raise ValueError("a requirement cannot be both satisfied and missing")
        return self


class EvidenceValidator:
    """Apply the versioned sufficiency policy without using ranking scores."""

    def __init__(self, policy: SufficiencyPolicy = DEFAULT_SUFFICIENCY_POLICY) -> None:
        if type(policy) is not SufficiencyPolicy:
            raise TypeError("policy must be an exact SufficiencyPolicy")
        self.policy = policy

    def validate(
        self,
        intent: QueryIntent,
        evidence: Sequence[Evidence],
        conflicts: Sequence[Conflict],
        as_of: date | datetime,
    ) -> ValidationOutcome:
        if type(intent) is not QueryIntent:
            raise TypeError("intent must be an exact business QueryIntent")
        if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
            raise TypeError("evidence must be a sequence")
        if isinstance(conflicts, (str, bytes)) or not isinstance(conflicts, Sequence):
            raise TypeError("conflicts must be a sequence")
        as_of_date = _as_of_date(as_of)
        requirements = EvidenceRequirements.for_intent(intent, policy=self.policy)

        if intent.kind == "out_of_scope":
            reason = ValidationReason(
                code="out_of_scope", requirement_id=None, branch=None, evidence_ids=(),
                blocking=True, detail=intent.out_of_scope_reason or "unsupported_request",
            )
            return ValidationOutcome(
                can_answer=False,
                decision="refuse",
                error_code="out_of_scope",
                requirements=requirements,
                satisfied_requirements=(),
                missing_requirements=(),
                reasons=(reason,),
                eligible_evidence_ids=(),
                excluded_evidence_ids=(),
                conflict_ids=(),
                degraded_components=(),
            )

        ordered_evidence, contract_reasons = _validated_evidence(evidence)
        ordered_conflicts = _validated_conflicts(conflicts)
        degraded = tuple(
            sorted(
                {
                    component
                    for item in ordered_evidence
                    if item.retrieval_provenance is not None
                    for component in item.retrieval_provenance.degraded_components
                }
            )
        )
        conflict_reasons, blocked_ids, conflict_ids = _conflict_reasons(
            requirements, ordered_evidence, ordered_conflicts, as_of_date
        )

        all_reasons: list[ValidationReason] = [*contract_reasons, *conflict_reasons]
        satisfied: list[SatisfiedRequirement] = []
        missing: list[MissingRequirement] = []
        eligible_ids: set[str] = set()
        fatal_components = set(self.policy.fatal_degraded_components)

        for requirement in requirements.items:
            branch_evidence = tuple(
                item for item in ordered_evidence if item.locator.branch == requirement.branch
            )
            valid: list[Evidence] = []
            failures: list[ValidationReason] = []
            for item in branch_evidence:
                if item.evidence_id in blocked_ids:
                    continue
                item_failures = _check_evidence(
                    requirement,
                    item,
                    as_of_date,
                    fatal_components=fatal_components,
                    sources_by_fact=self.policy.sources_by_fact,
                )
                failures.extend(item_failures)
                if not any(reason.blocking for reason in item_failures):
                    valid.append(item)

            source_count = _independent_source_count(valid)
            if source_count >= requirement.minimum_independent_sources:
                ids = tuple(sorted(item.evidence_id for item in valid))
                eligible_ids.update(ids)
                satisfied.append(
                    SatisfiedRequirement(
                        requirement_id=requirement.requirement_id,
                        branch=requirement.branch,
                        evidence_ids=ids,
                        independent_source_count=source_count,
                    )
                )
                all_reasons.extend(
                    reason.model_copy(update={"blocking": False})
                    for reason in failures
                )
                continue

            all_reasons.extend(failures)
            if not branch_evidence:
                all_reasons.append(
                    ValidationReason(
                        code="missing", requirement_id=requirement.requirement_id,
                        branch=requirement.branch, evidence_ids=(), blocking=True,
                        detail="required evidence branch returned no Evidence",
                    )
                )
            if valid and source_count < requirement.minimum_independent_sources:
                all_reasons.append(
                    ValidationReason(
                        code="insufficient_diversity", requirement_id=requirement.requirement_id,
                        branch=requirement.branch,
                        evidence_ids=tuple(sorted(item.evidence_id for item in valid)),
                        blocking=True,
                        detail=(
                            f"independent source count {source_count} is below "
                            f"required {requirement.minimum_independent_sources}"
                        ),
                    )
                )
            elif branch_evidence and not failures and not valid:
                all_reasons.append(
                    ValidationReason(
                        code="missing", requirement_id=requirement.requirement_id,
                        branch=requirement.branch,
                        evidence_ids=tuple(sorted(item.evidence_id for item in branch_evidence)),
                        blocking=True, detail="no eligible Evidence remains",
                    )
                )

            relevant = [
                reason for reason in all_reasons
                if reason.blocking and reason.requirement_id == requirement.requirement_id
            ]
            codes = tuple(sorted({reason.code for reason in relevant}, key=_REASON_ORDER.__getitem__))
            missing.append(
                MissingRequirement(
                    requirement_id=requirement.requirement_id,
                    branch=requirement.branch,
                    reason_codes=codes or ("missing",),
                    considered_evidence_ids=tuple(sorted(item.evidence_id for item in branch_evidence)),
                )
            )

        reasons = _stable_reasons(all_reasons)
        has_conflict = any(reason.code == "contradictory" and reason.blocking for reason in reasons)
        blocking_codes = {reason.code for reason in reasons if reason.blocking}
        if not missing and not has_conflict and not contract_reasons:
            decision, error_code = "answer", None
        elif has_conflict:
            decision, error_code = "refuse", "evidence_conflict"
        elif "degraded_branch" in blocking_codes and blocking_codes <= {"degraded_branch", "missing"}:
            decision, error_code = "rewrite_once", "evidence_retryable"
        else:
            decision, error_code = "refuse", "evidence_insufficient"

        all_ids = {item.evidence_id for item in ordered_evidence}
        excluded_ids = tuple(sorted(all_ids - eligible_ids))
        return ValidationOutcome(
            can_answer=decision == "answer",
            decision=decision,
            error_code=error_code,
            requirements=requirements,
            satisfied_requirements=tuple(sorted(satisfied, key=lambda item: item.requirement_id)),
            missing_requirements=tuple(sorted(missing, key=lambda item: item.requirement_id)),
            reasons=reasons,
            eligible_evidence_ids=tuple(sorted(eligible_ids)),
            excluded_evidence_ids=excluded_ids,
            conflict_ids=conflict_ids,
            degraded_components=degraded,
        )


def _validated_evidence(
    values: Sequence[Evidence],
) -> tuple[tuple[Evidence, ...], tuple[ValidationReason, ...]]:
    by_id: dict[str, Evidence] = {}
    poisoned_ids: set[str] = set()
    reasons: list[ValidationReason] = []
    for value in values:
        if type(value) is not Evidence:
            reasons.append(
                ValidationReason(
                    code="invalid_contract", requirement_id=None, branch=None,
                    evidence_ids=(), blocking=True,
                    detail="input contains a value that is not an exact Evidence contract",
                )
            )
            continue
        try:
            verified = Evidence.model_validate(value.model_dump(mode="python"))
        except Exception:
            reasons.append(
                ValidationReason(
                    code="invalid_contract", requirement_id=None,
                    branch=getattr(getattr(value, "locator", None), "branch", None),
                    evidence_ids=(value.evidence_id,) if isinstance(value.evidence_id, str) else (),
                    blocking=True, detail="Evidence failed contract revalidation",
                )
            )
            continue
        previous = by_id.get(verified.evidence_id)
        if verified.evidence_id in poisoned_ids:
            continue
        if previous is None:
            by_id[verified.evidence_id] = verified
            continue
        # A repeated stable identity may differ only in ranking trace. Pick a
        # deterministic representative; sufficiency never reads those scores.
        if _nonranking_payload(previous) != _nonranking_payload(verified):
            reasons.append(
                ValidationReason(
                    code="invalid_contract", requirement_id=None,
                    branch=verified.locator.branch,
                    evidence_ids=(verified.evidence_id,), blocking=True,
                    detail="duplicate Evidence identity differs outside ranking trace",
                )
            )
            by_id.pop(verified.evidence_id, None)
            poisoned_ids.add(verified.evidence_id)
            continue
        by_id[verified.evidence_id] = min(
            previous, verified, key=lambda item: item.model_dump_json()
        )
    return tuple(sorted(by_id.values(), key=lambda item: item.evidence_id)), _stable_reasons(reasons)


def _nonranking_payload(evidence: Evidence) -> dict[str, object]:
    payload = evidence.model_dump(mode="python")
    provenance = payload.get("retrieval_provenance")
    if isinstance(provenance, dict):
        for key in ("rank", "pre_rerank_rank", "fusion_score", "rerank_score"):
            provenance.pop(key, None)
        components = provenance.get("components", ())
        for component in components:
            for key in ("rank", "raw_score", "relevance_contribution"):
                component.pop(key, None)
    return payload


def _validated_conflicts(values: Sequence[Conflict]) -> tuple[Conflict, ...]:
    by_id: dict[str, Conflict] = {}
    for value in values:
        if type(value) is not Conflict:
            raise TypeError("conflicts must contain exact Conflict contracts")
        verified = Conflict.model_validate(value.model_dump(mode="python"))
        previous = by_id.get(verified.conflict_id)
        if previous is not None and previous != verified:
            raise ValueError("duplicate conflict identity has different content")
        by_id[verified.conflict_id] = verified
    return tuple(sorted(by_id.values(), key=lambda item: item.conflict_id))


def _conflict_reasons(
    requirements: EvidenceRequirements,
    evidence: tuple[Evidence, ...],
    conflicts: tuple[Conflict, ...],
    as_of: date,
) -> tuple[list[ValidationReason], set[str], tuple[str, ...]]:
    reasons: list[ValidationReason] = []
    blocked: set[str] = set()
    relevant_ids: set[str] = set()
    evidence_ids = {item.evidence_id for item in evidence}
    selected: set[str] = set()
    mentioned: set[str] = set()
    for conflict in conflicts:
        relevant = [item for item in requirements.items if _conflict_relevant(item, conflict, as_of)]
        if not relevant:
            continue
        relevant_ids.add(conflict.conflict_id)
        mentioned.update(conflict.evidence_ids)
        if conflict.status == "resolved":
            selected.update(conflict.selected_evidence_ids)
            blocked.update(set(conflict.evidence_ids) - set(conflict.selected_evidence_ids))
            continue
        blocked.update(conflict.evidence_ids)
        for requirement in relevant:
            reasons.append(
                ValidationReason(
                    code="contradictory", requirement_id=requirement.requirement_id,
                    branch=requirement.branch,
                    evidence_ids=tuple(sorted(set(conflict.evidence_ids) & evidence_ids)),
                    blocking=True,
                    detail=f"unresolved conflict {conflict.conflict_id} covers a core fact",
                )
            )
    for item in evidence:
        if item.conflict_group_id is not None and item.evidence_id not in mentioned and item.evidence_id not in selected:
            blocked.add(item.evidence_id)
            matching = [
                requirement
                for requirement in requirements.items
                if _evidence_relevant(requirement, item, as_of)
            ]
            for requirement in matching:
                reasons.append(
                    ValidationReason(
                        code="contradictory", requirement_id=requirement.requirement_id,
                        branch=requirement.branch, evidence_ids=(item.evidence_id,), blocking=True,
                        detail="Evidence names a conflict group without a supplied resolution",
                    )
                )
    return reasons, blocked, tuple(sorted(relevant_ids))


def _conflict_relevant(requirement: EvidenceRequirement, conflict: Conflict, as_of: date) -> bool:
    if conflict.fact_type not in requirement.fact_types:
        return False
    if requirement.required_entity_ids and conflict.entity_id not in requirement.required_entity_ids:
        return False
    start = _date_value(conflict.valid_from)
    end = _date_value(conflict.valid_to)
    target_start = requirement.temporal_start or as_of
    target_end = requirement.temporal_end or as_of
    return not ((end is not None and end < target_start) or (start is not None and start > target_end))


def _evidence_relevant(
    requirement: EvidenceRequirement,
    evidence: Evidence,
    as_of: date,
) -> bool:
    if evidence.locator.branch != requirement.branch or evidence.fact_type not in requirement.fact_types:
        return False
    if requirement.required_entity_ids and evidence.entity_id not in requirement.required_entity_ids:
        return False
    start = _date_value(evidence.valid_from)
    end = _date_value(evidence.valid_to)
    target_start = requirement.temporal_start or as_of
    target_end = requirement.temporal_end or as_of
    return not (
        (end is not None and end < target_start)
        or (start is not None and start > target_end)
    )


def _check_evidence(
    requirement: EvidenceRequirement,
    evidence: Evidence,
    as_of: date,
    *,
    fatal_components: set[str],
    sources_by_fact: Mapping[str, tuple[str, ...]],
) -> list[ValidationReason]:
    failures: list[ValidationReason] = []

    def add(code: ReasonCode, detail: str, *, blocking: bool = True) -> None:
        failures.append(
            ValidationReason(
                code=code, requirement_id=requirement.requirement_id,
                branch=requirement.branch, evidence_ids=(evidence.evidence_id,),
                blocking=blocking, detail=detail,
            )
        )

    if evidence.locator.branch == "sql":
        _check_sql(requirement, evidence, as_of, add)
    else:
        _check_rag(requirement, evidence, as_of, fatal_components, sources_by_fact, add)
    return failures


def _check_sql(requirement, evidence, as_of, add) -> None:
    provenance = evidence.sql_provenance
    locator = evidence.locator
    if provenance is None:
        add("invalid_contract", "SQL Evidence lacks SQL provenance")
        return
    if provenance.row_count == 0:
        add("empty_result", "SQL aggregate returned zero rows")
    if not locator.raw_record_locators:
        add("missing_locator", "SQL aggregate has no bounded raw-record locator")
    if locator.raw_record_locators_truncated:
        add("truncated_locator", "SQL raw-record locator population was truncated")
    if tuple(sorted(provenance.metric_names)) != requirement.required_metrics:
        add("metric_mismatch", "SQL metric set does not equal the requested metric set")
    try:
        payload = json.loads(evidence.content)
    except json.JSONDecodeError:
        add("invalid_contract", "SQL content is not valid JSON")
        payload = {"rows": []}
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    if any(metric not in row for row in rows for metric in requirement.required_metrics):
        add("metric_mismatch", "SQL result rows omit a requested metric")

    expected_grain = set(requirement.required_dimensions)
    if requirement.require_currency:
        expected_grain.add("currency")
        if not evidence.currencies or "currency" not in provenance.aggregation_grain:
            add("currency_missing", "currency-valued aggregate lacks currency grain")
    if requirement.require_unit:
        expected_grain.add("unit")
        if not evidence.units or "unit" not in provenance.aggregation_grain:
            add("unit_missing", "quantity aggregate lacks unit grain")
    if (
        len(set(provenance.aggregation_grain)) != len(provenance.aggregation_grain)
        or set(provenance.aggregation_grain) != expected_grain
    ):
        add("grain_mismatch", "SQL aggregation grain differs from the requested claim grain")
    if provenance.time_grain != requirement.required_time_grain:
        add("grain_mismatch", "SQL time grain differs from the requested time grain")
    if requirement.temporal_start is not None:
        if (
            provenance.effective_start_date > requirement.temporal_start
            or provenance.effective_end_date < requirement.temporal_end
        ):
            add("temporal_gap", "SQL validity does not fully cover the requested interval")
    elif requirement.maximum_age_days is not None and provenance.effective_end_date < as_of - timedelta(days=requirement.maximum_age_days):
        add("stale", "SQL effective end is older than the lead policy permits")
    if provenance.effective_start_date > as_of or provenance.effective_end_date > as_of:
        add("out_of_scope", "SQL validity extends beyond as_of")
    if requirement.require_synthetic is not None and evidence.is_synthetic != requirement.require_synthetic:
        add("dimension_mismatch", "SQL dataset synthetic scope differs from the request")

    sql = provenance.normalized_sql.casefold()
    where_sql = sql.partition(" where ")[2]
    dimension_markers = (
        (requirement.required_country_codes, ".country_code"),
        (requirement.required_hs_codes, ".hs_code"),
        (requirement.required_company_names, ".company_name"),
        (requirement.required_entity_ids, ".id"),
    )
    for expected, marker in dimension_markers:
        if expected and marker not in where_sql:
            add("dimension_mismatch", f"validated SQL omits required {marker[1:]} scope")
    expected_filter_groups = sum(
        bool(values)
        for values in (
            requirement.required_country_codes,
            requirement.required_hs_codes,
            requirement.required_company_names,
            requirement.required_entity_ids,
        )
    ) + int(requirement.temporal_start is not None)
    user_filter_groups = {
        name.rsplit("_", maxsplit=1)[0]
        for name in provenance.bound_filter_names
        if name.startswith("filter_")
    }
    if len(user_filter_groups) < expected_filter_groups:
        add("dimension_mismatch", "validated SQL has fewer user filter groups than the request")


def _check_rag(requirement, evidence, as_of, fatal_components, sources_by_fact, add) -> None:
    provenance = evidence.retrieval_provenance
    if provenance is None:
        add("invalid_contract", "RAG Evidence lacks retrieval provenance")
        return
    components = set(provenance.degraded_components)
    fatal = components & fatal_components
    if fatal:
        add("degraded_branch", "required retrieval branch has a fatal degraded component")
    elif components:
        add("degraded_branch", "retrieval branch is partially degraded", blocking=False)
    if evidence.fact_type == "unknown":
        add("unknown_fact", "unknown fact types cannot satisfy a factual requirement")
    elif evidence.fact_type not in requirement.fact_types:
        add("fact_type_mismatch", "Evidence fact type does not cover the requested fact")
    elif not _content_supports(evidence.fact_type, evidence.content):
        add("unsupported_fact_content", "content lacks a factual signal for its declared fact type")

    if evidence.source_type not in sources_by_fact.get(evidence.fact_type, ()):
        add("low_authority", "source category is not approved for this fact type")
    if requirement.requested_source_types and evidence.source_type not in requirement.requested_source_types:
        add("dimension_mismatch", "source category is outside the requested retrieval scope")
    if requirement.require_resolved_entity and (
        provenance.entity_resolution_status != "resolved"
        or provenance.entity_resolution_id is None
        or evidence.entity_id != provenance.entity_resolution_id
    ):
        add("unresolved_entity", "company-scoped Evidence lacks a resolved consistent entity")

    mismatch = False
    if requirement.required_entity_ids and evidence.entity_id not in requirement.required_entity_ids:
        mismatch = True
    if requirement.required_company_names:
        expected_names = {item.casefold() for item in requirement.required_company_names}
        if evidence.company_name is None or evidence.company_name.casefold() not in expected_names:
            mismatch = True
        elif evidence.company_name.casefold() not in evidence.content.casefold():
            mismatch = True
    if requirement.required_country_codes and evidence.country_code not in requirement.required_country_codes:
        mismatch = True
    if requirement.required_hs_codes and evidence.hs_code not in requirement.required_hs_codes:
        mismatch = True
    if requirement.require_synthetic is not None and evidence.is_synthetic != requirement.require_synthetic:
        mismatch = True
    if mismatch:
        add("dimension_mismatch", "Evidence entity/company/country/HS/synthetic scope differs from the request")

    published = _aware_date(evidence.publish_time)
    valid_from = _date_value(evidence.valid_from)
    valid_to = _date_value(evidence.valid_to)
    if evidence.publish_time is not None and published is None:
        add("out_of_scope", "publish_time must be timezone-aware")
    if isinstance(evidence.valid_from, datetime) and (
        evidence.valid_from.tzinfo is None or evidence.valid_from.utcoffset() is None
    ):
        add("out_of_scope", "valid_from must be timezone-aware when it is a datetime")
    if isinstance(evidence.valid_to, datetime) and (
        evidence.valid_to.tzinfo is None or evidence.valid_to.utcoffset() is None
    ):
        add("out_of_scope", "valid_to must be timezone-aware when it is a datetime")
    observed = published if published is not None else valid_from
    if observed is not None and observed > as_of:
        add("out_of_scope", "Evidence is dated after as_of")
    if valid_to is not None and valid_to < as_of and requirement.temporal_start is None:
        add("stale", "Evidence validity ended before as_of")
    if requirement.temporal_start is not None:
        if published is None or not (requirement.temporal_start <= published <= requirement.temporal_end):
            add("temporal_gap", "publication time is outside the requested interval")
    if requirement.maximum_age_days is not None:
        if observed is None or observed < as_of - timedelta(days=requirement.maximum_age_days):
            add("stale", "Evidence observation is older than the fact policy permits")


_CONTENT_SIGNALS = {
    "company_status": (
        "active", "capacity", "closed", "closure", "expanded", "expansion", "factory",
        "launched", "opened", "operating", "operational", "production",
        "产能", "扩产", "开业", "停业", "生产", "经营", "运营",
    ),
    "contact": ("address", "contact", "email", "phone", "地址", "联系", "电话", "邮箱"),
    "market_signal": ("demand", "growth", "market", "price", "需求", "增长", "市场", "价格"),
    "product_offering": ("catalog", "offer", "product", "sell", "supply", "产品", "供应", "销售"),
    "regulation": ("compliance", "deadline", "effective", "must", "regulation", "require", "合规", "法规", "生效", "要求"),
    "risk": ("closed", "recall", "risk", "sanction", "停业", "制裁", "召回", "风险"),
    "trade_activity": ("export", "import", "procurement", "purchase", "trade", "采购", "出口", "进口", "贸易"),
}


def _content_supports(fact_type: str, content: str) -> bool:
    signals = _CONTENT_SIGNALS.get(fact_type)
    if signals is None:
        return False
    normalized = re.sub(r"\s+", " ", content.casefold())
    return any(signal in normalized for signal in signals)


def _independent_source_count(evidence: Sequence[Evidence]) -> int:
    sql = [item for item in evidence if item.locator.branch == "sql"]
    rag = [item for item in evidence if item.locator.branch == "rag"]
    count = len({item.source_id for item in sql})
    edges: dict[str, set[str]] = {}
    for item in rag:
        provenance = item.retrieval_provenance
        if provenance is not None:
            edges.setdefault(item.source_id, set()).add(provenance.dedupe_cluster_id)
    # Maximum bipartite matching: both source identity and dedupe cluster must
    # be independent, so neither repeated pages nor syndicated copies count.
    matched_cluster: dict[str, str] = {}

    def assign(source: str, visited: set[str]) -> bool:
        for cluster in sorted(edges[source]):
            if cluster in visited:
                continue
            visited.add(cluster)
            previous = matched_cluster.get(cluster)
            if previous is None or assign(previous, visited):
                matched_cluster[cluster] = source
                return True
        return False

    for source in sorted(edges):
        assign(source, set())
    return count + len(matched_cluster)


def _stable_reasons(values: Sequence[ValidationReason]) -> tuple[ValidationReason, ...]:
    unique: dict[tuple[object, ...], ValidationReason] = {}
    for item in values:
        key = (
            _REASON_ORDER[item.code], item.requirement_id or "", item.branch or "",
            item.evidence_ids, item.blocking, item.detail,
        )
        unique[key] = item
    return tuple(unique[key] for key in sorted(unique))


def _as_of_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime as_of must be timezone-aware")
        return value.astimezone(timezone.utc).date()
    if type(value) is date:
        return value
    raise TypeError("as_of must be a date or timezone-aware datetime")


def _date_value(value: date | datetime | None) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    return value


def _aware_date(value: datetime | None) -> date | None:
    if value is None or value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc).date()
