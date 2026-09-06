"""Deterministic, score-independent grounding checks for generated claims."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Self

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, model_validator

from trade_agent.evidence.models import Claim, Evidence
from trade_agent.evidence.validator import ValidationOutcome
from trade_agent.agents.intent import QueryIntent
from trade_agent.generation.base import ClaimScope, DraftAnswer, generation_evidence, sql_claim_rows


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


class GuardOutcome(_Contract):
    accepted: StrictBool
    answer: StrictStr | None
    claims: tuple[Claim, ...]
    refusal_reason: StrictStr | None
    error_codes: tuple[StrictStr, ...] = ()

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if (self.answer is None) == (self.refusal_reason is None):
            raise ValueError("exactly one of answer or refusal_reason is required")
        if self.accepted != (self.answer is not None):
            raise ValueError("accepted must agree with answer presence")
        if self.answer is not None and (not self.answer.strip() or not self.claims):
            raise ValueError("accepted result needs retained claims and nonblank answer")
        if self.refusal_reason is not None and (not self.refusal_reason.strip() or self.claims):
            raise ValueError("refusal result needs a reason and no claims")
        if self.error_codes != tuple(sorted(set(self.error_codes))) or any(not value.strip() for value in self.error_codes):
            raise ValueError("error codes must be sorted, unique and nonblank")
        return self


class ClaimHallucinationGuard:
    """Accept only claims exactly supported by the supplied generation Evidence."""

    def guard(
        self,
        draft: DraftAnswer,
        evidence: Sequence[Evidence],
        intent: QueryIntent,
        validation: ValidationOutcome,
    ) -> GuardOutcome:
        checked_draft = _checked_draft(draft)
        checked_validation = ValidationOutcome.model_validate(validation.model_dump(mode="python"))
        retained_evidence = generation_evidence(intent, evidence, validation)
        evidence_by_id = _checked_evidence(retained_evidence)
        if checked_draft.refusal_reason is not None:
            return GuardOutcome(
                accepted=False,
                answer=None,
                claims=(),
                refusal_reason=checked_draft.refusal_reason,
                error_codes=(),
            )
        retained: list[Claim] = []
        unsupported_core = False
        unsupported = False
        scope_by_claim = {item.claim_id: item for item in checked_draft.claim_scopes}
        for claim in checked_draft.claims:
            if _claim_supported(claim, scope_by_claim[claim.claim_id], evidence_by_id):
                retained.append(claim)
            else:
                unsupported = True
                unsupported_core = unsupported_core or _is_required_core_claim(claim, checked_validation)
        error_codes = ("claim_unsupported",) if unsupported else ()
        if unsupported_core:
            return GuardOutcome(
                accepted=False,
                answer=None,
                claims=(),
                refusal_reason="core_claim_unsupported",
                error_codes=error_codes,
            )
        if not retained:
            return GuardOutcome(
                accepted=False,
                answer=None,
                claims=(),
                refusal_reason="no_supported_claims",
                error_codes=tuple(sorted(set((*error_codes, "no_supported_claims")))),
            )
        if not _core_requirements_covered(retained, checked_validation):
            return GuardOutcome(
                accepted=False,
                answer=None,
                claims=(),
                refusal_reason="core_requirement_uncovered",
                error_codes=tuple(sorted(set((*error_codes, "core_requirement_uncovered")))),
            )
        ordered = tuple(retained)
        return GuardOutcome(
            accepted=True,
            answer="\n".join(claim.text for claim in ordered),
            claims=ordered,
            refusal_reason=None,
            error_codes=error_codes,
        )


def _checked_draft(value: DraftAnswer) -> DraftAnswer:
    if type(value) is not DraftAnswer:
        raise TypeError("draft must be an exact DraftAnswer")
    return DraftAnswer.model_validate(value.model_dump(mode="python"))


def _checked_evidence(value: Sequence[Evidence]) -> dict[str, Evidence]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("evidence must be a sequence")
    result: dict[str, Evidence] = {}
    for item in value:
        if type(item) is not Evidence:
            raise TypeError("guard accepts exact Evidence contracts")
        verified = Evidence.model_validate(item.model_dump(mode="python"))
        if verified.evidence_id in result:
            raise ValueError("guard Evidence IDs must be unique")
        result[verified.evidence_id] = verified
    return result


def _claim_supported(claim: Claim, scope: ClaimScope, evidence_by_id: Mapping[str, Evidence]) -> bool:
    if claim.status != "supported" or not claim.evidence_ids:
        return False
    cited = tuple(evidence_by_id.get(evidence_id) for evidence_id in claim.evidence_ids)
    if any(item is None for item in cited):
        return False
    return all(_claim_matches_evidence(claim, scope, item) for item in cited if item is not None)


def _claim_matches_evidence(claim: Claim, scope: ClaimScope, evidence: Evidence) -> bool:
    if claim.entity_id is not None and claim.entity_id != evidence.entity_id:
        return False
    if claim.fact_type != evidence.fact_type:
        return False
    if claim.period_start != evidence.valid_from:
        return False
    if claim.period_end != evidence.valid_to:
        return False
    if scope.country_code != evidence.country_code or scope.hs_code != evidence.hs_code:
        return False
    if scope.aggregation_grain != evidence.aggregation_grain:
        return False
    if evidence.locator.branch == "sql":
        return bool(sql_claim_rows(claim, evidence))
    if claim.text != evidence.content:
        return False
    if claim.value is not None:
        return False
    if claim.unit is not None and (claim.unit not in evidence.units or claim.unit not in evidence.content):
        return False
    if claim.currency is not None and (claim.currency not in evidence.currencies or claim.currency not in evidence.content):
        return False
    return True


def _is_required_core_claim(claim: Claim, validation: ValidationOutcome) -> bool:
    """The validator, never a provider field, classifies required facts as core."""
    return claim.fact_type is not None and any(
        requirement.core and claim.fact_type in requirement.fact_types
        for requirement in validation.requirements.items
    )


def _core_requirements_covered(claims: Sequence[Claim], validation: ValidationOutcome) -> bool:
    """Every validator-satisfied core requirement needs a retained atomic claim."""
    satisfied_by_id = {item.requirement_id: set(item.evidence_ids) for item in validation.satisfied_requirements}
    for requirement in validation.requirements.items:
        if not requirement.core:
            continue
        support_ids = satisfied_by_id.get(requirement.requirement_id, set())
        if not any(
            claim.fact_type in requirement.fact_types and bool(set(claim.evidence_ids) & support_ids)
            for claim in claims
        ):
            return False
    return True
