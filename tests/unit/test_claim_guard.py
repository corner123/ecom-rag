from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from trade_agent.evidence.claim_guard import ClaimHallucinationGuard
from trade_agent.agents.intent import QueryIntent
from trade_agent.generation.base import DraftAnswer
from trade_agent.generation.deterministic import DeterministicAnswerGenerator
from tests.unit.test_evidence_validator import _rag, _validated
from tests.unit.test_generation import validated_context


@pytest.fixture
def guard() -> ClaimHallucinationGuard:
    return ClaimHallucinationGuard()


@pytest.fixture
def supported_draft(validated_context):
    intent, evidence, validation = validated_context
    draft = DeterministicAnswerGenerator().generate(intent, (evidence,), validation)
    return draft, evidence, intent, validation


def _mutate_claim(draft: DraftAnswer, mutation: str) -> DraftAnswer:
    claim = draft.claims[0]
    updates = {
        "amount": {"value": "999.99", "text": claim.text.replace("12.30", "999.99")},
        "currency": {"currency": "EUR", "text": claim.text.replace("USD", "EUR")},
        "date": {
            "period_start": date(2026, 1, 1),
            "period_end": date(2026, 1, 31),
            "text": claim.text.replace("2026-03-04 to 2026-09-04", "2026-01-01 to 2026-01-31"),
        },
        "entity": {"entity_id": "company:forged", "text": claim.text.replace("Acme", "Forged")},
        "evidence_id": {"evidence_ids": ("sql_" + "f" * 64,)},
    }
    return draft.model_copy(update={"claims": (claim.model_copy(update=updates[mutation]),)})


@pytest.mark.parametrize("mutation", ["amount", "currency", "date", "entity", "evidence_id"])
def test_guard_rejects_mutated_claim(mutation: str, guard: ClaimHallucinationGuard, supported_draft) -> None:
    draft, evidence, intent, validation = supported_draft
    outcome = guard.guard(_mutate_claim(draft, mutation), (evidence,), intent, validation)

    assert outcome.accepted is False
    assert "claim_unsupported" in outcome.error_codes
    assert outcome.answer is None


def test_guard_rebuilds_answer_from_retained_noncore_claims(guard: ClaimHallucinationGuard, supported_draft) -> None:
    draft, evidence, intent, validation = supported_draft
    claim = draft.claims[0]
    noncore = claim.model_copy(update={"claim_id": "claim_" + "e" * 64, "fact_type": "analysis_note", "value": "999.99", "text": "unrelated claim"})
    scopes = (draft.claim_scopes[0], draft.claim_scopes[0].model_copy(update={"claim_id": noncore.claim_id}))
    draft = draft.model_copy(update={"claims": (claim, noncore), "claim_scopes": scopes, "core_claim_ids": (claim.claim_id,)})

    outcome = guard.guard(draft, (evidence,), intent, validation)

    assert outcome.accepted is True
    assert outcome.claims == (claim,)
    assert outcome.answer == claim.text
    assert "999.99" not in outcome.answer


def test_guard_derives_core_claims_from_validation_not_provider_ids(guard: ClaimHallucinationGuard, supported_draft) -> None:
    draft, evidence, intent, validation = supported_draft
    claim = draft.claims[0]
    required = claim.model_copy(update={
        "claim_id": "claim_" + "e" * 64,
        "value": "999.99",
        "text": claim.text.replace("12.30", "999.99"),
    })
    scopes = (
        draft.claim_scopes[0],
        draft.claim_scopes[0].model_copy(update={"claim_id": required.claim_id}),
    )
    forged = draft.model_copy(update={"claims": (claim, required), "claim_scopes": scopes, "core_claim_ids": ()})

    outcome = guard.guard(forged, (evidence,), intent, validation)

    assert outcome.accepted is False
    assert outcome.refusal_reason == "core_claim_unsupported"


@pytest.mark.parametrize("mutation", ["unit", "fact", "grain", "jurisdiction", "hs_code", "datetime"])
def test_guard_rejects_structured_scope_mutations(mutation: str, guard: ClaimHallucinationGuard, supported_draft) -> None:
    draft, evidence, intent, validation = supported_draft
    claim = draft.claims[0]
    scope = draft.claim_scopes[0]
    claim_update: dict[str, object] = {}
    scope_update: dict[str, object] = {}
    if mutation == "unit":
        claim_update = {"unit": "kg", "text": claim.text + " kg"}
    elif mutation == "fact":
        claim_update = {"fact_type": "quantity", "text": claim.text.replace("trade_amount", "quantity")}
    elif mutation == "grain":
        scope_update = {"aggregation_grain": ("forged_grain",)}
    elif mutation == "jurisdiction":
        scope_update = {"country_code": "CN"}
    elif mutation == "hs_code":
        scope_update = {"hs_code": "999999"}
    else:
        claim_update = {
            "period_start": datetime(2026, 3, 4, 1, tzinfo=timezone.utc),
            "period_end": datetime(2026, 9, 4, 1, tzinfo=timezone.utc),
        }
    forged = draft.model_copy(update={
        "claims": (claim.model_copy(update=claim_update),),
        "claim_scopes": (scope.model_copy(update=scope_update),),
    })

    outcome = guard.guard(forged, (evidence,), intent, validation)

    assert outcome.accepted is False
    assert "claim_unsupported" in outcome.error_codes


def test_guard_preserves_rag_datetime_instants_and_structured_scope(guard: ClaimHallucinationGuard) -> None:
    instant = datetime(2026, 8, 20, 10, tzinfo=timezone.utc)
    evidence = _rag(suffix="guard-precision", country_code="US", hs_code="850440", valid_from=instant)
    intent = QueryIntent(question="website status", kind="external_intelligence", need_external_intel=True)
    validation = _validated(intent, (evidence,))
    assert validation.can_answer
    draft = DeterministicAnswerGenerator().generate(intent, (evidence,), validation)
    scope = draft.claim_scopes[0].model_copy(update={"country_code": "CN"})
    claim = draft.claims[0].model_copy(update={"period_start": datetime(2026, 8, 20, 11, tzinfo=timezone.utc)})
    forged = draft.model_copy(update={"claims": (claim,), "claim_scopes": (scope,)})

    outcome = guard.guard(forged, (evidence,), intent, validation)

    assert outcome.accepted is False
    assert "claim_unsupported" in outcome.error_codes
