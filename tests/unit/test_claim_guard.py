from __future__ import annotations

from datetime import date

import pytest

from trade_agent.evidence.claim_guard import ClaimHallucinationGuard
from trade_agent.generation.base import DraftAnswer
from trade_agent.generation.deterministic import DeterministicAnswerGenerator
from tests.unit.test_generation import validated_context


@pytest.fixture
def guard() -> ClaimHallucinationGuard:
    return ClaimHallucinationGuard()


@pytest.fixture
def supported_draft(validated_context):
    intent, evidence, validation = validated_context
    draft = DeterministicAnswerGenerator().generate(intent, (evidence,), validation)
    return draft, evidence


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
    draft, evidence = supported_draft
    outcome = guard.guard(_mutate_claim(draft, mutation), (evidence,))

    assert outcome.accepted is False
    assert "claim_unsupported" in outcome.error_codes
    assert outcome.answer is None


def test_guard_rebuilds_answer_from_retained_noncore_claims(guard: ClaimHallucinationGuard, supported_draft) -> None:
    draft, evidence = supported_draft
    claim = draft.claims[0]
    noncore = claim.model_copy(update={"claim_id": "claim_" + "e" * 64, "value": "999.99", "text": "unrelated claim"})
    draft = draft.model_copy(update={"claims": (claim, noncore), "core_claim_ids": (claim.claim_id,)})

    outcome = guard.guard(draft, (evidence,))

    assert outcome.accepted is True
    assert outcome.claims == (claim,)
    assert outcome.answer == claim.text
    assert "999.99" not in outcome.answer
