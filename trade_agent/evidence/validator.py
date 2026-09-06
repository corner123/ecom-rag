"""Deterministic, score-independent validation of evidence coverage."""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import math
import re
from typing import Literal, Mapping, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from trade_agent.agents.intent import QueryIntent
from trade_agent.evidence.models import Conflict, Evidence
from trade_agent.evidence.requirements import (
    BranchErrorCode,
    DEFAULT_SUFFICIENCY_POLICY,
    EvidenceRequirement,
    EvidenceRequirements,
    RETRYABLE_BRANCH_ERROR_CODES,
    InvalidIntentContract,
    PartialDegradationCode,
    SourceAuthorityRule,
    SufficiencyPolicy,
)

ReasonCode = Literal[
    "branch_hard_error", "branch_report_missing", "branch_zero_hits", "conflict_mismatch", "contradictory",
    "currency_missing", "degraded_branch", "dimension_mismatch", "empty_result",
    "fact_type_mismatch", "grain_mismatch", "insufficient_diversity", "invalid_contract",
    "invalid_intent", "invalid_metric", "low_authority", "metric_mismatch", "missing", "missing_locator",
    "out_of_scope", "partial_degradation", "stale", "temporal_gap", "truncated_locator", "unit_missing",
    "unknown_fact", "unresolved_entity", "unsupported_fact_content",
]
_ORDERED_CODES = (
    "out_of_scope", "invalid_intent", "conflict_mismatch", "contradictory", "invalid_contract",
    "branch_report_missing", "branch_hard_error", "degraded_branch", "partial_degradation", "branch_zero_hits", "missing", "empty_result",
    "missing_locator", "truncated_locator", "unresolved_entity", "unknown_fact",
    "fact_type_mismatch", "unsupported_fact_content", "low_authority", "dimension_mismatch",
    "metric_mismatch", "invalid_metric", "grain_mismatch", "currency_missing", "unit_missing", "temporal_gap",
    "stale", "insufficient_diversity",
)
_REASON_ORDER = {code: index for index, code in enumerate(_ORDERED_CODES)}


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


class BranchExecutionReport(_Contract):
    branch: Literal["sql", "rag"]
    attempted: StrictBool
    completed: StrictBool
    zero_hits: StrictBool
    degraded_components: tuple[PartialDegradationCode, ...] = ()
    error_codes: tuple[BranchErrorCode, ...] = ()

    @model_validator(mode="after")
    def coherent(self) -> Self:
        for values in (self.degraded_components, self.error_codes):
            if values != tuple(sorted(set(values))) or any(not item.strip() for item in values):
                raise ValueError("branch report details must be sorted unique nonblank strings")
        if self.completed and not self.attempted:
            raise ValueError("completed branch must have been attempted")
        if self.zero_hits and (not self.attempted or not self.completed):
            raise ValueError("zero_hits requires a completed attempt")
        if not self.attempted and (self.completed or self.zero_hits or self.degraded_components or not self.error_codes):
            raise ValueError("an unattempted branch requires an error and no execution state")
        if self.attempted and not self.completed and not self.error_codes:
            raise ValueError("an incomplete attempted branch requires an error")
        if self.degraded_components and not self.attempted:
            raise ValueError("partial degradation requires an attempted branch")
        return self


class ValidationContext(_Contract):
    branch_reports: tuple[BranchExecutionReport, ...]

    @model_validator(mode="after")
    def stable(self) -> Self:
        branches = tuple(item.branch for item in self.branch_reports)
        if branches != tuple(sorted(set(branches))):
            raise ValueError("branch reports must be sorted by unique branch")
        return self


class ValidationReason(_Contract):
    code: ReasonCode
    requirement_id: StrictStr | None
    branch: Literal["sql", "rag"] | None
    evidence_ids: tuple[StrictStr, ...] = ()
    blocking: StrictBool
    detail: StrictStr

    @model_validator(mode="after")
    def stable(self) -> Self:
        if self.evidence_ids != tuple(sorted(set(self.evidence_ids))) or not self.detail.strip():
            raise ValueError("reason IDs must be stable and detail nonblank")
        return self


class IndependentSourceProof(_Contract):
    evidence_id: StrictStr = Field(pattern=r"^(?:sql|rag)_[0-9a-f]{64}$")
    group_key: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class SatisfiedRequirement(_Contract):
    requirement_id: StrictStr
    branch: Literal["sql", "rag"]
    evidence_ids: tuple[StrictStr, ...]
    independent_source_count: StrictInt = Field(ge=1)
    independence_proofs: tuple[IndependentSourceProof, ...]

    @model_validator(mode="after")
    def stable(self) -> Self:
        if not self.evidence_ids or self.evidence_ids != tuple(sorted(set(self.evidence_ids))):
            raise ValueError("satisfied evidence IDs must be nonempty, sorted and unique")
        proof_keys = tuple((item.group_key, item.evidence_id) for item in self.independence_proofs)
        if proof_keys != tuple(sorted(set(proof_keys))):
            raise ValueError("independence proofs must be sorted and unique")
        if (
            self.independent_source_count != len(self.independence_proofs)
            or self.independent_source_count > len(self.evidence_ids)
            or len({item.group_key for item in self.independence_proofs}) != len(self.independence_proofs)
            or len({item.evidence_id for item in self.independence_proofs}) != len(self.independence_proofs)
            or not {item.evidence_id for item in self.independence_proofs}.issubset(self.evidence_ids)
        ):
            raise ValueError("independent source count must close over unique support proofs")
        return self


class MissingRequirement(_Contract):
    requirement_id: StrictStr
    branch: Literal["sql", "rag"]
    reason_codes: tuple[ReasonCode, ...]
    considered_evidence_ids: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def stable(self) -> Self:
        if not self.reason_codes or self.reason_codes != tuple(sorted(set(self.reason_codes), key=_REASON_ORDER.__getitem__)):
            raise ValueError("missing reason codes must be nonempty, sorted and unique")
        if self.considered_evidence_ids != tuple(sorted(set(self.considered_evidence_ids))):
            raise ValueError("considered evidence IDs must be sorted and unique")
        return self


class ValidationOutcome(_Contract):
    can_answer: StrictBool
    decision: Literal["answer", "rewrite_once", "refuse"]
    error_code: Literal["evidence_conflict", "evidence_insufficient", "evidence_retryable", "invalid_intent", "out_of_scope"] | None
    intent_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    policy_fingerprint: StrictStr
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
        if self.policy_fingerprint != self.requirements.policy_fingerprint:
            raise ValueError("outcome policy fingerprint differs from requirements")
        if self.can_answer != (self.decision == "answer") or self.can_answer != (self.error_code is None):
            raise ValueError("answer, decision and error code disagree")
        for values in (self.eligible_evidence_ids, self.excluded_evidence_ids, self.conflict_ids, self.degraded_components):
            if values != tuple(sorted(set(values))):
                raise ValueError("outcome tuple must be sorted and unique")
        if set(self.eligible_evidence_ids) & set(self.excluded_evidence_ids):
            raise ValueError("eligible and excluded Evidence must be disjoint")
        satisfied_ids = tuple(item.requirement_id for item in self.satisfied_requirements)
        missing_ids = tuple(item.requirement_id for item in self.missing_requirements)
        required_ids = {item.requirement_id for item in self.requirements.items}
        if satisfied_ids != tuple(sorted(set(satisfied_ids))) or missing_ids != tuple(sorted(set(missing_ids))):
            raise ValueError("requirement results must be sorted and unique")
        if set(satisfied_ids) & set(missing_ids) or set(satisfied_ids) | set(missing_ids) != required_ids:
            raise ValueError("satisfied and missing must exactly partition requirements")
        requirement_by_id = {item.requirement_id: item for item in self.requirements.items}
        for item in (*self.satisfied_requirements, *self.missing_requirements):
            if requirement_by_id[item.requirement_id].branch != item.branch:
                raise ValueError("requirement result branch differs from its requirement")
        for reason in self.reasons:
            if reason.requirement_id is not None and (
                reason.requirement_id not in requirement_by_id
                or reason.branch != requirement_by_id[reason.requirement_id].branch
            ):
                raise ValueError("reason requirement/branch does not exist in requirements")
        eligible = {evidence_id for item in self.satisfied_requirements for evidence_id in item.evidence_ids}
        if eligible != set(self.eligible_evidence_ids):
            raise ValueError("eligible Evidence must equal satisfied support")
        reason_keys = [(_REASON_ORDER[item.code], item.requirement_id or "", item.branch or "", item.evidence_ids, item.blocking, item.detail) for item in self.reasons]
        if reason_keys != sorted(set(reason_keys)):
            raise ValueError("reasons must be deterministic and unique")
        blocking = {item.code for item in self.reasons if item.blocking}
        for item in self.missing_requirements:
            present = {reason.code for reason in self.reasons if reason.blocking and reason.requirement_id == item.requirement_id}
            if set(item.reason_codes) != present:
                raise ValueError("missing reason codes must equal all matching blocking reasons")
        expected_error = (
            None if self.decision == "answer"
            else "evidence_retryable" if self.decision == "rewrite_once"
            else "invalid_intent" if "invalid_intent" in blocking
            else "out_of_scope" if blocking == {"out_of_scope"} and not required_ids
            else "evidence_conflict" if {"contradictory", "conflict_mismatch"} & blocking
            else "evidence_insufficient"
        )
        if self.error_code != expected_error:
            raise ValueError("decision, blockers, and error code disagree")
        if self.decision == "answer" and (self.missing_requirements or blocking or not required_ids):
            raise ValueError("answer requires nonempty fully satisfied unblocked requirements")
        if self.decision == "rewrite_once" and (not self.missing_requirements or "degraded_branch" not in blocking or blocking - {"degraded_branch", "branch_zero_hits", "missing"}):
            raise ValueError("rewrite_once is reserved for fatal branch degradation")
        if self.error_code == "evidence_conflict" and not ({"contradictory", "conflict_mismatch"} & blocking):
            raise ValueError("conflict error requires a conflict blocker")
        if (self.error_code == "invalid_intent") != ("invalid_intent" in blocking):
            raise ValueError("invalid intent error requires its blocker")
        if self.decision != "answer" and not blocking:
            raise ValueError("a non-answer requires a blocking reason")
        if self.error_code == "out_of_scope" and (blocking != {"out_of_scope"} or required_ids):
            raise ValueError("out-of-scope outcome requires only its terminal blocker")
        if self.error_code != "out_of_scope" and not required_ids and "invalid_intent" not in blocking:
            raise ValueError("zero-requirement outcomes must be out-of-scope or invalid intent")
        if self.error_code == "evidence_insufficient" and ({"contradictory", "conflict_mismatch"} & blocking):
            raise ValueError("conflict blockers require evidence_conflict")
        if self.error_code == "evidence_insufficient" and "degraded_branch" in blocking and not (blocking - {"degraded_branch", "branch_zero_hits", "missing"}):
            raise ValueError("fatal-only blockers require rewrite_once")
        return self


class EvidenceValidator:
    def __init__(self, policy: SufficiencyPolicy = DEFAULT_SUFFICIENCY_POLICY) -> None:
        if type(policy) is not SufficiencyPolicy:
            raise TypeError("policy must be an exact SufficiencyPolicy")
        self.policy = SufficiencyPolicy.model_validate(policy.model_dump(mode="python"))

    def validate(self, intent: QueryIntent, evidence: Sequence[Evidence], conflicts: Sequence[Conflict], as_of: date | datetime, *, context: ValidationContext | None = None) -> ValidationOutcome:
        if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
            raise TypeError("evidence must be a sequence")
        if isinstance(conflicts, (str, bytes)) or not isinstance(conflicts, Sequence):
            raise TypeError("conflicts must be a sequence")
        as_of_instant = _as_of_instant(as_of)
        as_of_is_date = type(as_of) is date
        try:
            current_intent_fingerprint = intent_fingerprint(intent)
        except (TypeError, ValueError):
            current_intent_fingerprint = sha256(b"invalid-intent-contract").hexdigest()
        try:
            requirements = EvidenceRequirements.for_intent(intent, policy=self.policy)
        except InvalidIntentContract as error:
            requirements = EvidenceRequirements.invalid(getattr(intent, "kind", "invalid"), self.policy)
            return self._terminal(requirements, current_intent_fingerprint, "invalid_intent", "invalid_intent", str(error))
        if requirements.intent_kind == "out_of_scope":
            return self._terminal(requirements, current_intent_fingerprint, "out_of_scope", "out_of_scope", getattr(intent, "out_of_scope_reason", None) or "unsupported_request")

        ordered_evidence, contract_reasons = _validated_evidence(evidence)
        try:
            ordered_conflicts = _validated_conflicts(conflicts)
        except (TypeError, ValueError):
            ordered_conflicts = ()
            contract_reasons += (ValidationReason(code="invalid_contract", requirement_id=None, branch=None, blocking=True, detail="conflicts failed contract revalidation"),)
        context_reasons, fatal_branches, degraded = _context_reasons(requirements, context, ordered_evidence, self.policy)
        conflict_reasons, blocked_ids, conflict_ids = _conflict_reasons(requirements, ordered_evidence, ordered_conflicts, as_of_instant)
        all_reasons = [*contract_reasons, *context_reasons, *conflict_reasons]
        satisfied: list[SatisfiedRequirement] = []
        missing: list[MissingRequirement] = []
        eligible_ids: set[str] = set()

        for requirement in requirements.items:
            branch_evidence = tuple(item for item in ordered_evidence if item.locator.branch == requirement.branch)
            valid: list[Evidence] = []
            failures: list[ValidationReason] = []
            branch_blocked = requirement.branch in fatal_branches or any(reason.blocking and reason.requirement_id == requirement.requirement_id for reason in context_reasons)
            for item in branch_evidence:
                if item.evidence_id in blocked_ids:
                    continue
                item_failures = _check_evidence(
                    requirement, item, as_of_instant, as_of_is_date, self.policy
                )
                failures.extend(item_failures)
                if not branch_blocked and not any(reason.blocking for reason in item_failures):
                    valid.append(item)
            independence_proofs = _independent_source_proofs(valid, self.policy)
            source_count = len(independence_proofs)
            if not branch_blocked and source_count >= requirement.minimum_independent_sources:
                ids = tuple(sorted(item.evidence_id for item in valid))
                eligible_ids.update(ids)
                satisfied.append(SatisfiedRequirement(
                    requirement_id=requirement.requirement_id,
                    branch=requirement.branch,
                    evidence_ids=ids,
                    independent_source_count=source_count,
                    independence_proofs=independence_proofs,
                ))
                all_reasons.extend(reason.model_copy(update={"blocking": False}) for reason in failures)
                continue
            all_reasons.extend(failures)
            existing = [reason for reason in all_reasons if reason.blocking and reason.requirement_id == requirement.requirement_id]
            if not branch_evidence and not any(reason.code == "missing" for reason in existing):
                all_reasons.append(ValidationReason(code="missing", requirement_id=requirement.requirement_id, branch=requirement.branch, blocking=True, detail="required evidence branch returned no Evidence"))
            if valid and source_count < requirement.minimum_independent_sources:
                all_reasons.append(ValidationReason(code="insufficient_diversity", requirement_id=requirement.requirement_id, branch=requirement.branch, evidence_ids=tuple(sorted(item.evidence_id for item in valid)), blocking=True, detail=f"independent source count {source_count} is below required {requirement.minimum_independent_sources}"))
            relevant = [reason for reason in all_reasons if reason.blocking and reason.requirement_id == requirement.requirement_id]
            if not relevant:
                reason = ValidationReason(code="missing", requirement_id=requirement.requirement_id, branch=requirement.branch, evidence_ids=tuple(sorted(item.evidence_id for item in branch_evidence)), blocking=True, detail="no eligible Evidence remains")
                all_reasons.append(reason)
                relevant.append(reason)
            codes = tuple(sorted({reason.code for reason in relevant}, key=_REASON_ORDER.__getitem__))
            missing.append(MissingRequirement(requirement_id=requirement.requirement_id, branch=requirement.branch, reason_codes=codes, considered_evidence_ids=tuple(sorted(item.evidence_id for item in branch_evidence))))

        reasons = _stable_reasons(all_reasons)
        blocking_codes = {reason.code for reason in reasons if reason.blocking}
        if not missing and not blocking_codes:
            decision, error_code = "answer", None
        elif {"contradictory", "conflict_mismatch"} & blocking_codes:
            decision, error_code = "refuse", "evidence_conflict"
        elif "degraded_branch" in blocking_codes and blocking_codes <= {"degraded_branch", "branch_zero_hits", "missing"}:
            decision, error_code = "rewrite_once", "evidence_retryable"
        else:
            decision, error_code = "refuse", "evidence_insufficient"
        all_ids = {item.evidence_id for item in ordered_evidence}
        return ValidationOutcome(can_answer=decision == "answer", decision=decision, error_code=error_code, intent_fingerprint=current_intent_fingerprint,
            policy_fingerprint=self.policy.policy_fingerprint, requirements=requirements,
            satisfied_requirements=tuple(sorted(satisfied, key=lambda item: item.requirement_id)), missing_requirements=tuple(sorted(missing, key=lambda item: item.requirement_id)), reasons=reasons,
            eligible_evidence_ids=tuple(sorted(eligible_ids)), excluded_evidence_ids=tuple(sorted(all_ids - eligible_ids)), conflict_ids=conflict_ids, degraded_components=degraded)

    def _terminal(self, requirements: EvidenceRequirements, current_intent_fingerprint: str, reason_code: Literal["invalid_intent", "out_of_scope"], error_code: Literal["invalid_intent", "out_of_scope"], detail: str) -> ValidationOutcome:
        reason = ValidationReason(code=reason_code, requirement_id=None, branch=None, blocking=True, detail=detail)
        return ValidationOutcome(can_answer=False, decision="refuse", error_code=error_code, intent_fingerprint=current_intent_fingerprint, policy_fingerprint=self.policy.policy_fingerprint,
            requirements=requirements, satisfied_requirements=(), missing_requirements=(), reasons=(reason,), eligible_evidence_ids=(), excluded_evidence_ids=(), conflict_ids=(), degraded_components=())


def intent_fingerprint(intent: QueryIntent) -> str:
    """Stable identity of the exact validated request, independent of policy."""
    if type(intent) is not QueryIntent:
        raise TypeError("intent must be an exact QueryIntent")
    checked = QueryIntent.model_validate(intent.model_dump(mode="python", round_trip=True))
    canonical = json.dumps(
        checked.model_dump(mode="json", round_trip=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _context_reasons(requirements: EvidenceRequirements, context: ValidationContext | None, evidence: tuple[Evidence, ...], policy: SufficiencyPolicy) -> tuple[list[ValidationReason], set[str], tuple[str, ...]]:
    if type(context) is not ValidationContext:
        reports = {}
    else:
        try:
            verified_context = ValidationContext.model_validate(context.model_dump(mode="python"))
        except Exception:
            verified_context = ValidationContext(branch_reports=())
        reports = {item.branch: item for item in verified_context.branch_reports}
    fatal_errors = set(policy.fatal_error_codes)
    hard_errors = set(policy.hard_error_codes)
    fatal_evidence = set(policy.fatal_evidence_degraded_components)
    partial_evidence = set(policy.partial_degraded_components)
    degraded = {component for item in evidence if item.retrieval_provenance for component in item.retrieval_provenance.degraded_components}
    reasons: list[ValidationReason] = []
    fatal_branches: set[str] = set()
    for branch in requirements.required_branches:
        report = reports.get(branch)
        branch_items = [item for item in requirements.items if item.branch == branch]
        if report is None:
            for requirement in branch_items:
                reasons.append(ValidationReason(code="branch_report_missing", requirement_id=requirement.requirement_id, branch=branch, blocking=True, detail="required branch has no execution report"))
            continue
        degraded.update(report.degraded_components)
        evidence_components = {
            component
            for item in evidence
            if item.locator.branch == branch and item.retrieval_provenance
            for component in item.retrieval_provenance.degraded_components
        }
        evidence_fatal = evidence_components & fatal_evidence
        report_unknown = set(report.degraded_components) - partial_evidence
        evidence_unknown = evidence_components - fatal_evidence - partial_evidence
        report_fatal = set(report.error_codes) & fatal_errors
        report_hard = set(report.error_codes) & (
            hard_errors | (set(RETRYABLE_BRANCH_ERROR_CODES) - fatal_errors)
        )
        is_fatal = bool(evidence_fatal or report_fatal)
        if is_fatal:
            fatal_branches.add(branch)
            for requirement in branch_items:
                reasons.append(ValidationReason(code="degraded_branch", requirement_id=requirement.requirement_id, branch=branch, blocking=True, detail="required branch execution is fatally degraded"))
        if report_hard or report_unknown or evidence_unknown:
            for requirement in branch_items:
                reasons.append(ValidationReason(code="branch_hard_error", requirement_id=requirement.requirement_id, branch=branch, blocking=True, detail="required branch reported a non-retryable hard error"))
        partial = (set(report.degraded_components) & partial_evidence) | (evidence_components & partial_evidence)
        if partial:
            blocks = policy.partial_degradation_mode == "block"
            for requirement in branch_items:
                reasons.append(ValidationReason(code="partial_degradation", requirement_id=requirement.requirement_id, branch=branch, blocking=blocks, detail="required branch execution is partially degraded"))
        if report.zero_hits:
            for requirement in branch_items:
                reasons.append(ValidationReason(code="branch_zero_hits", requirement_id=requirement.requirement_id, branch=branch, blocking=True, detail="required branch completed with zero hits"))
    return reasons, fatal_branches, tuple(sorted(degraded))


def _validated_evidence(values: Sequence[Evidence]) -> tuple[tuple[Evidence, ...], tuple[ValidationReason, ...]]:
    by_id: dict[str, Evidence] = {}
    poisoned: set[str] = set()
    reasons: list[ValidationReason] = []
    for value in values:
        if type(value) is not Evidence:
            reasons.append(ValidationReason(code="invalid_contract", requirement_id=None, branch=None, blocking=True, detail="input is not an exact Evidence contract"))
            continue
        try:
            verified = Evidence.model_validate(value.model_dump(mode="python"))
        except Exception:
            reasons.append(ValidationReason(code="invalid_contract", requirement_id=None, branch=getattr(getattr(value, "locator", None), "branch", None), evidence_ids=(value.evidence_id,) if isinstance(value.evidence_id, str) else (), blocking=True, detail="Evidence failed contract revalidation"))
            continue
        previous = by_id.get(verified.evidence_id)
        if verified.evidence_id in poisoned:
            continue
        if previous is not None and _nonranking_payload(previous) != _nonranking_payload(verified):
            by_id.pop(verified.evidence_id, None); poisoned.add(verified.evidence_id)
            reasons.append(ValidationReason(code="invalid_contract", requirement_id=None, branch=verified.locator.branch, evidence_ids=(verified.evidence_id,), blocking=True, detail="duplicate Evidence identity differs outside ranking trace"))
        elif previous is None or verified.model_dump_json() < previous.model_dump_json():
            by_id[verified.evidence_id] = verified
    return tuple(sorted(by_id.values(), key=lambda item: item.evidence_id)), _stable_reasons(reasons)


def _nonranking_payload(evidence: Evidence) -> dict[str, object]:
    payload = evidence.model_dump(mode="python")
    provenance = payload.get("retrieval_provenance")
    if isinstance(provenance, dict):
        for key in ("rank", "pre_rerank_rank", "fusion_score", "rerank_score"):
            provenance.pop(key, None)
        for component in provenance.get("components", ()):
            for key in ("rank", "raw_score", "relevance_contribution"):
                component.pop(key, None)
    return payload


def _validated_conflicts(values: Sequence[Conflict]) -> tuple[Conflict, ...]:
    by_id: dict[str, Conflict] = {}
    for value in values:
        if type(value) is not Conflict:
            raise TypeError("conflicts must contain exact Conflict contracts")
        verified = Conflict.model_validate(value.model_dump(mode="python"))
        if verified.conflict_id in by_id and by_id[verified.conflict_id] != verified:
            raise ValueError("duplicate conflict identity differs")
        by_id[verified.conflict_id] = verified
    return tuple(sorted(by_id.values(), key=lambda item: item.conflict_id))


def _conflict_reasons(requirements: EvidenceRequirements, evidence: tuple[Evidence, ...], conflicts: tuple[Conflict, ...], as_of: datetime) -> tuple[list[ValidationReason], set[str], tuple[str, ...]]:
    reasons: list[ValidationReason] = []
    blocked: set[str] = set()
    by_id = {item.evidence_id: item for item in evidence}
    conflicts_by_id = {item.conflict_id: item for item in conflicts}
    relevant_ids: set[str] = set()
    for conflict in conflicts:
        referenced = [by_id.get(item) for item in conflict.evidence_ids]
        mismatch = any(
            item is None
            or item.conflict_group_id != conflict.conflict_id
            or item.fact_type != conflict.fact_type
            or item.entity_id != conflict.entity_id
            or not _conflict_scope_matches(item, conflict)
            for item in referenced
        )
        mismatch = mismatch or any(by_id[item].conflict_group_id != conflict.conflict_id for item in conflict.selected_evidence_ids if item in by_id)
        relevant = [requirement for requirement in requirements.items if _conflict_relevant(requirement, conflict, as_of)]
        affected = {
            requirement.requirement_id: requirement
            for requirement in requirements.items
            if requirement in relevant
            or any(item is not None and _evidence_relevant(requirement, item, as_of) for item in referenced)
        }
        if relevant:
            relevant_ids.add(conflict.conflict_id)
        if mismatch:
            blocked.update(item for item in conflict.evidence_ids if item in by_id)
            for requirement in affected.values():
                reasons.append(ValidationReason(code="conflict_mismatch", requirement_id=requirement.requirement_id, branch=requirement.branch, evidence_ids=tuple(sorted(set(conflict.evidence_ids) & set(by_id))), blocking=True, detail=f"conflict {conflict.conflict_id} does not match Evidence conflict groups"))
            continue
        if conflict.status == "resolved":
            blocked.update(set(conflict.evidence_ids) - set(conflict.selected_evidence_ids))
        else:
            blocked.update(conflict.evidence_ids)
            for requirement in relevant:
                reasons.append(ValidationReason(code="contradictory", requirement_id=requirement.requirement_id, branch=requirement.branch, evidence_ids=tuple(sorted(set(conflict.evidence_ids) & set(by_id))), blocking=True, detail=f"unresolved conflict {conflict.conflict_id} covers a core fact"))
    for item in evidence:
        group = conflicts_by_id.get(item.conflict_group_id) if item.conflict_group_id else None
        if item.conflict_group_id and (
            group is None or item.evidence_id not in group.evidence_ids
        ):
            blocked.add(item.evidence_id)
            for requirement in requirements.items:
                if _evidence_relevant(requirement, item, as_of):
                    reasons.append(ValidationReason(code="conflict_mismatch", requirement_id=requirement.requirement_id, branch=requirement.branch, evidence_ids=(item.evidence_id,), blocking=True, detail="Evidence is not reported as a member of its named conflict group"))
    return reasons, blocked, tuple(sorted(relevant_ids))


def _conflict_scope_matches(evidence: Evidence, conflict: Conflict) -> bool:
    expected_currencies = () if conflict.currency is None else (conflict.currency,)
    expected_units = () if conflict.unit is None else (conflict.unit,)
    return (
        evidence.currencies == expected_currencies
        and evidence.units == expected_units
        and evidence.aggregation_grain == conflict.aggregation_grain
        and _conflict_interval_within_evidence(evidence, conflict)
    )


def _conflict_interval_within_evidence(
    evidence: Evidence, conflict: Conflict
) -> bool:
    try:
        evidence_start = (
            _aware_boundary(evidence.valid_from, False)
            if evidence.valid_from is not None else None
        )
        evidence_end = (
            _aware_boundary(evidence.valid_to, True)
            if evidence.valid_to is not None else None
        )
        conflict_start = (
            _aware_boundary(conflict.valid_from, False)
            if conflict.valid_from is not None else None
        )
        conflict_end = (
            _aware_boundary(conflict.valid_to, True)
            if conflict.valid_to is not None else None
        )
    except ValueError:
        return False
    if (
        evidence_start is not None
        and evidence_end is not None
        and evidence_start > evidence_end
    ) or (
        conflict_start is not None
        and conflict_end is not None
        and conflict_start > conflict_end
    ):
        return False
    if conflict_start is None and evidence_start is not None:
        return False
    if conflict_end is None and evidence_end is not None:
        return False
    if (
        evidence_start is not None
        and conflict_start is not None
        and evidence_start > conflict_start
    ):
        return False
    if (
        evidence_end is not None
        and conflict_end is not None
        and evidence_end < conflict_end
    ):
        return False
    return True


def _conflict_relevant(requirement: EvidenceRequirement, conflict: Conflict, as_of: datetime) -> bool:
    if conflict.fact_type not in requirement.fact_types or requirement.required_entity_ids and conflict.entity_id not in requirement.required_entity_ids:
        return False
    start, end = _utc_date(conflict.valid_from), _utc_date(conflict.valid_to)
    target_start = _utc_date(requirement.temporal_start) or as_of.date()
    target_end = _utc_date(requirement.temporal_end) or as_of.date()
    return not (end is not None and end < target_start or start is not None and start > target_end)


def _evidence_relevant(requirement: EvidenceRequirement, evidence: Evidence, as_of: datetime) -> bool:
    return evidence.locator.branch == requirement.branch and evidence.fact_type in requirement.fact_types and (not requirement.required_entity_ids or evidence.entity_id in requirement.required_entity_ids)


def _check_evidence(requirement: EvidenceRequirement, evidence: Evidence, as_of: datetime, as_of_is_date: bool, policy: SufficiencyPolicy) -> list[ValidationReason]:
    failures: list[ValidationReason] = []
    def add(code: ReasonCode, detail: str, *, blocking: bool = True) -> None:
        failures.append(ValidationReason(code=code, requirement_id=requirement.requirement_id, branch=requirement.branch, evidence_ids=(evidence.evidence_id,), blocking=blocking, detail=detail))
    if evidence.locator.branch == "sql":
        _check_sql(requirement, evidence, as_of, add)
    else:
        _check_rag(requirement, evidence, as_of, as_of_is_date, policy, add)
    return failures


def _check_sql(requirement: EvidenceRequirement, evidence: Evidence, as_of: datetime, add) -> None:
    provenance, locator = evidence.sql_provenance, evidence.locator
    if provenance is None:
        add("invalid_contract", "SQL Evidence lacks SQL provenance"); return
    if provenance.row_count == 0: add("empty_result", "SQL aggregate returned zero rows")
    if not locator.raw_record_locators: add("missing_locator", "SQL aggregate has no bounded raw-record locator")
    if locator.raw_record_locators_truncated: add("truncated_locator", "SQL raw-record locator population was truncated")
    if tuple(sorted(provenance.metric_names)) != requirement.required_metrics: add("metric_mismatch", "SQL metric set differs from requested metric")
    try:
        payload = json.loads(evidence.content)
    except json.JSONDecodeError:
        payload = {"rows": []}; add("invalid_contract", "SQL content is not valid JSON")
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    for metric in requirement.required_metrics:
        if any(metric not in row or not _finite_number(row.get(metric)) for row in rows if isinstance(row, dict)) or any(not isinstance(row, dict) for row in rows):
            add("invalid_metric", "SQL result rows need finite numeric requested metrics")
    expected_grain = set(requirement.required_dimensions)
    if requirement.require_currency:
        expected_grain.add("currency")
        row_currencies = {
            row.get("currency") for row in rows
            if isinstance(row, dict) and isinstance(row.get("currency"), str) and row.get("currency")
        }
        if (
            not evidence.currencies
            or "currency" not in provenance.aggregation_grain
            or any(not isinstance(row, dict) or not isinstance(row.get("currency"), str) or not row.get("currency") for row in rows)
            or tuple(sorted(row_currencies)) != evidence.currencies
        ): add("currency_missing", "every currency-valued row needs a nonblank currency matching Evidence scope")
    if requirement.require_unit:
        expected_grain.add("unit")
        row_units = {
            row.get("unit") for row in rows
            if isinstance(row, dict) and isinstance(row.get("unit"), str) and row.get("unit")
        }
        if (
            not evidence.units
            or "unit" not in provenance.aggregation_grain
            or any(not isinstance(row, dict) or not isinstance(row.get("unit"), str) or not row.get("unit") for row in rows)
            or tuple(sorted(row_units)) != evidence.units
        ): add("unit_missing", "every quantity row needs a nonblank unit matching Evidence scope")
    if len(set(provenance.aggregation_grain)) != len(provenance.aggregation_grain) or set(provenance.aggregation_grain) != expected_grain: add("grain_mismatch", "SQL aggregation grain differs from requested claim grain")
    if provenance.time_grain != requirement.required_time_grain: add("grain_mismatch", "SQL time grain differs from requested time grain")
    as_of_date = as_of.date()
    if requirement.temporal_start is not None:
        start, end = _utc_date(requirement.temporal_start), _utc_date(requirement.temporal_end)
        if provenance.effective_start_date > start or provenance.effective_end_date < end: add("temporal_gap", "SQL validity does not fully cover requested interval")
    elif requirement.maximum_age_days is not None and provenance.effective_end_date < as_of_date - timedelta(days=requirement.maximum_age_days): add("stale", "SQL effective end is older than lead policy permits")
    if provenance.effective_start_date > as_of_date or provenance.effective_end_date > as_of_date: add("out_of_scope", "SQL validity extends beyond as_of")
    if requirement.require_synthetic is not None and evidence.is_synthetic != requirement.require_synthetic: add("dimension_mismatch", "SQL dataset synthetic scope differs from request")
    try:
        bindings, ast_filter_names = _sql_scope_bindings(provenance.normalized_sql)
    except _SqlScopeRejected as error:
        add("invalid_contract", str(error))
        return
    if tuple(provenance.bound_filter_names) != tuple(sorted(ast_filter_names)):
        add("invalid_contract", "SQL AST placeholders and bound filter names do not close exactly")
    expected_scopes = (
        (requirement.required_entity_ids, requirement.required_entity_column),
        (requirement.required_company_names, requirement.required_company_column),
        (requirement.required_country_codes, requirement.required_country_column),
        (requirement.required_hs_codes, requirement.required_hs_column),
    )
    covered: set[str] = set()
    for values, column in expected_scopes:
        if not values:
            continue
        names = bindings.get(column or "", ())
        if len(names) != len(values) or not _sequential_value_placeholders(names):
            add("dimension_mismatch", f"SQL does not bind every requested value on {column}")
        covered.update(names)
    if requirement.temporal_start is not None:
        date_names = bindings.get("tr.trade_date", ())
        if len(date_names) != 2 or not _paired_range_placeholders(date_names):
            add("dimension_mismatch", "SQL does not bind the requested trade-date interval")
        covered.update(date_names)
    policy_names = {"policy_end_date", "policy_is_synthetic", "policy_start_date"}
    user_names = set(ast_filter_names) - policy_names
    if user_names != covered:
        add("invalid_contract", "SQL contains an unused or unclassified user filter placeholder")


class _SqlScopeRejected(ValueError):
    """The SQL provenance cannot prove an executable typed predicate scope."""


def _sql_scope_bindings(sql: str) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    try:
        statements = sqlglot.parse(sql, read="mysql", error_level="RAISE")
    except ParseError as error:
        raise _SqlScopeRejected("SQL provenance is not valid MySQL") from error
    if len(statements) != 1 or type(statements[0]) is not exp.Select:
        raise _SqlScopeRejected("SQL provenance must contain exactly one SELECT")
    tree = statements[0]
    if any(
        node.comments
        or node.args.get("hint") is not None
        or bool(node.args.get("hints"))
        or bool(node.args.get("operation_modifiers"))
        or (
            isinstance(node, exp.Join)
            and str(node.args.get("kind", "")).casefold() == "straight_join"
        )
        for node in tree.walk()
    ):
        raise _SqlScopeRejected(
            "SQL provenance comments, hints, and operation modifiers are forbidden"
        )
    where = tree.args.get("where")
    if type(where) is not exp.Where:
        raise _SqlScopeRejected("SQL provenance requires a WHERE predicate")

    bindings: dict[str, list[str]] = {}
    predicate_names: list[str] = []
    policy_shapes: set[tuple[str, tuple[str, ...]]] = set()
    for predicate in _and_predicates(where.this):
        column, names = _executable_predicate(predicate)
        predicate_names.extend(names)
        shape = (column, names)
        if shape in {
            ("data_scope.is_synthetic", ("policy_is_synthetic",)),
            ("tr.trade_date", ("policy_start_date", "policy_end_date")),
        }:
            policy_shapes.add(shape)
            continue
        if any(not name.startswith("filter_") for name in names):
            raise _SqlScopeRejected("SQL contains an unclassified non-policy placeholder")
        if column in bindings:
            raise _SqlScopeRejected("SQL scope contains multiple predicates for one business column")
        bindings[column] = list(names)

    expected_policy = {
        ("data_scope.is_synthetic", ("policy_is_synthetic",)),
        ("tr.trade_date", ("policy_start_date", "policy_end_date")),
    }
    if policy_shapes != expected_policy:
        raise _SqlScopeRejected("SQL omits or changes a required policy predicate")
    all_placeholders = tuple(item.name.casefold() for item in tree.find_all(exp.Placeholder))
    if (
        len(predicate_names) != len(set(predicate_names))
        or sorted(predicate_names) != sorted(all_placeholders)
    ):
        raise _SqlScopeRejected("SQL placeholders must occur once in executable WHERE predicates")
    return (
        {column: tuple(sorted(names)) for column, names in bindings.items()},
        tuple(sorted(predicate_names)),
    )


def _and_predicates(node: exp.Expression) -> tuple[exp.Expression, ...]:
    if type(node) is exp.Paren:
        return _and_predicates(node.this)
    if type(node) is exp.And:
        return (*_and_predicates(node.this), *_and_predicates(node.expression))
    if isinstance(node, exp.Or) or isinstance(node, exp.Not):
        raise _SqlScopeRejected("SQL predicate scope cannot contain OR or NOT")
    return (node,)


def _executable_predicate(predicate: exp.Expression) -> tuple[str, tuple[str, ...]]:
    if type(predicate) is exp.EQ:
        column = _qualified_column(predicate.this)
        names = (_placeholder_name(predicate.expression),)
    elif type(predicate) is exp.In:
        column = _qualified_column(predicate.this)
        expressions = tuple(predicate.expressions)
        if not expressions or predicate.args.get("query") is not None:
            raise _SqlScopeRejected("SQL IN scope requires a nonempty placeholder list")
        names = tuple(_placeholder_name(item) for item in expressions)
    elif type(predicate) is exp.Between:
        column = _qualified_column(predicate.this)
        names = (
            _placeholder_name(predicate.args.get("low")),
            _placeholder_name(predicate.args.get("high")),
        )
    else:
        raise _SqlScopeRejected("SQL WHERE contains an unsupported predicate")
    return column, names


def _qualified_column(value: exp.Expression | None) -> str:
    if (
        type(value) is not exp.Column
        or not value.table
        or value.args.get("db") is not None
        or value.args.get("catalog") is not None
    ):
        raise _SqlScopeRejected("SQL scope predicates require qualified alias columns")
    return f"{value.table.casefold()}.{value.name.casefold()}"


def _placeholder_name(value: exp.Expression | None) -> str:
    if type(value) is not exp.Placeholder or not value.name:
        raise _SqlScopeRejected("SQL scope predicates accept only bound placeholders")
    return value.name.casefold()


def _sequential_value_placeholders(names: tuple[str, ...]) -> bool:
    parsed = [re.fullmatch(r"filter_(\d+)_(\d+)", name) for name in names]
    if not parsed or any(item is None for item in parsed):
        return False
    groups = {item.group(1) for item in parsed if item is not None}
    indexes = sorted(int(item.group(2)) for item in parsed if item is not None)
    return len(groups) == 1 and indexes == list(range(len(names)))


def _paired_range_placeholders(names: tuple[str, ...]) -> bool:
    parsed = [re.fullmatch(r"filter_(\d+)_(start|end)", name) for name in names]
    return (
        all(item is not None for item in parsed)
        and len({item.group(1) for item in parsed if item is not None}) == 1
        and {item.group(2) for item in parsed if item is not None} == {"start", "end"}
    )


def _check_rag(requirement: EvidenceRequirement, evidence: Evidence, as_of: datetime, as_of_is_date: bool, policy: SufficiencyPolicy, add) -> None:
    provenance = evidence.retrieval_provenance
    if provenance is None: add("invalid_contract", "RAG Evidence lacks retrieval provenance"); return
    # Branch degradation is evaluated once from the explicit execution report
    # plus Task 4 provenance in _context_reasons; ranking metadata is irrelevant.
    if evidence.fact_type == "unknown": add("unknown_fact", "unknown fact types cannot satisfy a factual requirement")
    elif evidence.fact_type not in requirement.fact_types: add("fact_type_mismatch", "Evidence fact type differs from requested fact")
    elif not _content_supports(evidence.fact_type, evidence.content): add("unsupported_fact_content", "content lacks a factual signal for declared fact type")
    if evidence.source_type not in policy.sources_by_fact.get(evidence.fact_type, ()) or _authority_publisher(evidence, policy.authority_rules, set(policy.accepted_authority_directness)) is None:
        add("low_authority", "source category and reviewed identity do not establish authority")
    if requirement.requested_source_types and evidence.source_type not in requirement.requested_source_types: add("dimension_mismatch", "source category is outside requested retrieval scope")
    if requirement.require_resolved_entity and (provenance.entity_resolution_status != "resolved" or provenance.entity_resolution_id is None or evidence.entity_id != provenance.entity_resolution_id): add("unresolved_entity", "company-scoped Evidence lacks a resolved consistent entity")
    mismatch = bool(requirement.required_entity_ids and evidence.entity_id not in requirement.required_entity_ids)
    if requirement.required_company_names:
        names = {item.casefold() for item in requirement.required_company_names}
        mismatch |= evidence.company_name is None or evidence.company_name.casefold() not in names or evidence.company_name.casefold() not in evidence.content.casefold()
    mismatch |= bool(requirement.required_country_codes and evidence.country_code not in requirement.required_country_codes)
    mismatch |= bool(requirement.required_hs_codes and evidence.hs_code not in requirement.required_hs_codes)
    mismatch |= requirement.require_synthetic is not None and evidence.is_synthetic != requirement.require_synthetic
    if mismatch: add("dimension_mismatch", "Evidence entity/company/country/HS/synthetic scope differs from request")
    published = _aware_instant(evidence.publish_time)
    valid_from, valid_to = _utc_date(evidence.valid_from), _utc_date(evidence.valid_to)
    valid_from_instant = _aware_instant(evidence.valid_from) if isinstance(evidence.valid_from, datetime) else None
    valid_to_instant = _aware_instant(evidence.valid_to) if isinstance(evidence.valid_to, datetime) else None
    if evidence.publish_time is not None and published is None: add("out_of_scope", "publish_time must be timezone-aware")
    if isinstance(evidence.valid_from, datetime) and _aware_instant(evidence.valid_from) is None: add("out_of_scope", "valid_from datetime must be timezone-aware")
    if isinstance(evidence.valid_to, datetime) and _aware_instant(evidence.valid_to) is None: add("out_of_scope", "valid_to datetime must be timezone-aware")
    if published is not None and published > as_of: add("out_of_scope", "publication is after as_of")
    if (
        valid_from_instant is not None and valid_from_instant > as_of
        or valid_from_instant is None and valid_from is not None and valid_from > as_of.date()
    ): add("out_of_scope", "validity starts after as_of")
    if requirement.temporal_start is None and (
        valid_to_instant is not None and valid_to_instant < as_of
        or valid_to_instant is None and valid_to is not None and valid_to < as_of.date()
    ): add("stale", "Evidence validity ended before as_of")
    if requirement.temporal_start is not None:
        start, end = _aware_boundary(requirement.temporal_start, False), _aware_boundary(requirement.temporal_end, True)
        if published is None or not start <= published <= end: add("temporal_gap", "publication is outside requested interval")
    elif requirement.maximum_age_days is not None:
        observed = published or valid_from_instant or (_aware_boundary(valid_from, False) if valid_from is not None else None)
        stale = observed is None
        if observed is not None:
            stale = (
                observed.date() < as_of.date() - timedelta(days=requirement.maximum_age_days)
                if as_of_is_date
                else observed < as_of - timedelta(days=requirement.maximum_age_days)
            )
        if stale: add("stale", "Evidence observation is older than fact policy permits")


def _finite_number(value: object) -> bool:
    if value is None or isinstance(value, bool): return False
    if isinstance(value, float): return math.isfinite(value)
    if isinstance(value, int): return True
    if isinstance(value, Decimal): return value.is_finite()
    if isinstance(value, str):
        try: return Decimal(value).is_finite()
        except InvalidOperation: return False
    return False


def _authority_publisher(evidence: Evidence, rules: tuple[SourceAuthorityRule, ...], accepted: set[str]) -> str | None:
    identity = evidence.locator.source_identity
    parsed = urlsplit(identity)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or identity not in {evidence.source_url, evidence.canonical_url}
    ):
        return None
    host = parsed.hostname.casefold().rstrip(".")
    for rule in rules:
        if rule.source_type != evidence.source_type or rule.directness not in accepted:
            continue
        if identity in rule.exact_source_identities or any(host == suffix.casefold() or host.endswith("." + suffix.casefold()) for suffix in rule.domain_suffixes):
            return rule.publisher_id
    return None


_CONTENT_SIGNALS = {
    "company_status": ("active", "capacity", "closed", "closure", "expanded", "expansion", "factory", "launched", "opened", "operating", "operational", "production", "产能", "扩产", "开业", "停业", "生产", "经营", "运营"),
    "contact": ("address", "contact", "email", "phone", "地址", "联系", "电话", "邮箱"),
    "market_signal": ("demand", "growth", "market", "price", "需求", "增长", "市场", "价格"),
    "product_offering": ("catalog", "offer", "product", "sell", "supply", "产品", "供应", "销售"),
    "regulation": ("compliance", "deadline", "effective", "must", "regulation", "require", "合规", "法规", "生效", "要求"),
    "risk": ("closed", "recall", "risk", "sanction", "停业", "制裁", "召回", "风险"),
    "trade_activity": ("export", "import", "procurement", "purchase", "trade", "采购", "出口", "进口", "贸易"),
}


def _content_supports(fact_type: str, content: str) -> bool:
    return any(signal in re.sub(r"\s+", " ", content.casefold()) for signal in _CONTENT_SIGNALS.get(fact_type, ()))


def _independent_source_proofs(
    evidence: Sequence[Evidence], policy: SufficiencyPolicy
) -> tuple[IndependentSourceProof, ...]:
    sql_candidates = {
        (("sql_source_id", item.source_id),): item.evidence_id
        for item in sorted(evidence, key=lambda item: item.evidence_id, reverse=True)
        if item.locator.branch == "sql"
    }
    dimensions = policy.independence_dimensions
    rag_candidates: dict[tuple[tuple[str, str], ...], str] = {}
    for item in evidence:
        if item.locator.branch != "rag" or item.retrieval_provenance is None:
            continue
        publisher = _authority_publisher(
            item, policy.authority_rules, set(policy.accepted_authority_directness)
        ) or _source_origin(item.locator.source_identity)
        values = {
            "publisher_identity": publisher,
            "content_sha256": item.locator.content_hash,
            "dedupe_cluster_id": item.retrieval_provenance.dedupe_cluster_id,
        }
        key = tuple((dimension, values[dimension]) for dimension in dimensions)
        current = rag_candidates.get(key)
        if current is None or item.evidence_id < current:
            rag_candidates[key] = item.evidence_id
    ordered = tuple(sorted(rag_candidates.items()))

    def maximum(
        index: int, used: dict[str, set[str]]
    ) -> tuple[tuple[tuple[tuple[str, str], ...], str], ...]:
        if index == len(ordered):
            return ()
        best = maximum(index + 1, used)
        key, evidence_id = ordered[index]
        candidate = dict(key)
        if all(candidate[dimension] not in used[dimension] for dimension in dimensions):
            for dimension in dimensions:
                used[dimension].add(candidate[dimension])
            selected = ((key, evidence_id), *maximum(index + 1, used))
            for dimension in dimensions:
                used[dimension].remove(candidate[dimension])
            if len(selected) > len(best) or len(selected) == len(best) and selected < best:
                best = selected
        return best

    used = {dimension: set() for dimension in dimensions}
    selected = (*sorted(sql_candidates.items()), *maximum(0, used))
    proofs = (
        IndependentSourceProof(
            evidence_id=evidence_id,
            group_key=sha256(json.dumps(key, separators=(",", ":")).encode()).hexdigest(),
        )
        for key, evidence_id in selected
    )
    return tuple(sorted(proofs, key=lambda item: (item.group_key, item.evidence_id)))


def _source_origin(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return value
    try:
        port = parsed.port
    except ValueError:
        return value
    default_port = port is None or parsed.scheme == "http" and port == 80 or parsed.scheme == "https" and port == 443
    authority = parsed.hostname.casefold() if default_port else f"{parsed.hostname.casefold()}:{port}"
    return f"{parsed.scheme.casefold()}://{authority}"


def _stable_reasons(values: Sequence[ValidationReason]) -> tuple[ValidationReason, ...]:
    unique = {(_REASON_ORDER[item.code], item.requirement_id or "", item.branch or "", item.evidence_ids, item.blocking, item.detail): item for item in values}
    return tuple(unique[key] for key in sorted(unique))


def _as_of_instant(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        aware = _aware_instant(value)
        if aware is None: raise ValueError("datetime as_of must be timezone-aware")
        return aware
    if type(value) is date: return datetime.combine(value, time.max, timezone.utc)
    raise TypeError("as_of must be date or timezone-aware datetime")


def _aware_instant(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is None or value.utcoffset() is None: return None
    return value.astimezone(timezone.utc)


def _utc_date(value: date | datetime | None) -> date | None:
    if isinstance(value, datetime):
        aware = _aware_instant(value)
        return aware.date() if aware else None
    return value


def _aware_boundary(value: date | datetime | None, end: bool) -> datetime:
    if isinstance(value, datetime):
        aware = _aware_instant(value)
        if aware is None: raise ValueError("temporal requirement must be timezone-aware")
        return aware
    if isinstance(value, date): return datetime.combine(value, time.max if end else time.min, timezone.utc)
    raise ValueError("temporal boundary is missing")
