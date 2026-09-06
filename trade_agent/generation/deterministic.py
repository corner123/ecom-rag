"""Reproducible evidence-only structured answer generation."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from trade_agent.agents.intent import QueryIntent
from trade_agent.evidence.models import Claim, Evidence
from trade_agent.evidence.validator import ValidationOutcome
from trade_agent.generation.base import AnswerGenerator, ClaimScope, DraftAnswer, claim_id_for, generation_evidence, render_sql_claim


class DeterministicAnswerGenerator(AnswerGenerator):
    """Produce only canonical claims that can be replayed from retained Evidence."""

    def generate(
        self,
        intent: QueryIntent,
        evidence: Sequence[Evidence],
        validation: ValidationOutcome,
    ) -> DraftAnswer:
        retained = generation_evidence(intent, evidence, validation)
        if not validation.can_answer:
            return DraftAnswer(answer=None, claims=(), refusal_reason=validation.error_code or "evidence_insufficient")
        supported = tuple(
            sorted(
                ((claim, item) for item in retained for claim in _claims_for(item)),
                key=lambda item: item[0].claim_id,
            )
        )
        claims = tuple(claim for claim, _ in supported)
        if not claims:
            return DraftAnswer(answer=None, claims=(), refusal_reason="evidence_insufficient")
        return DraftAnswer(
            answer="\n".join(item.text for item in claims),
            claims=claims,
            refusal_reason=None,
            core_claim_ids=tuple(sorted(item.claim_id for item in claims)),
            claim_scopes=tuple(_claim_scope(claim, evidence) for claim, evidence in supported),
        )


def _claims_for(evidence: Evidence) -> tuple[Claim, ...]:
    if evidence.locator.branch == "sql":
        payload = json.loads(evidence.content)
        metrics = payload["metrics"]
        rows = payload["rows"]
        return tuple(
            _sql_claim(evidence, row, metric)
            for row in rows
            if isinstance(row, dict)
            for metric in metrics
            if metric in row
        )
    return (_retrieval_claim(evidence),)


def _sql_claim(evidence: Evidence, row: Mapping[str, object], metric: str) -> Claim:
    value = str(row[metric])
    currency = row.get("currency") if isinstance(row.get("currency"), str) else None
    unit = row.get("unit") if isinstance(row.get("unit"), str) else None
    text = render_sql_claim(evidence, row, metric)
    return Claim(
        claim_id=claim_id_for(evidence_id=evidence.evidence_id, payload={"metric": metric, "row": dict(row)}),
        text=text,
        status="supported",
        evidence_ids=(evidence.evidence_id,),
        entity_id=evidence.entity_id,
        fact_type=metric,
        value=value,
        unit=unit,
        currency=currency,
        period_start=evidence.valid_from,
        period_end=evidence.valid_to,
        confidence=1.0,
    )


def _retrieval_claim(evidence: Evidence) -> Claim:
    return Claim(
        claim_id=claim_id_for(evidence_id=evidence.evidence_id, payload={"content": evidence.content}),
        text=evidence.content,
        status="supported",
        evidence_ids=(evidence.evidence_id,),
        entity_id=evidence.entity_id,
        fact_type=evidence.fact_type,
        value=None,
        unit=None,
        currency=None,
        period_start=evidence.valid_from,
        period_end=evidence.valid_to,
        confidence=1.0,
    )


def _claim_scope(claim: Claim, evidence: Evidence) -> ClaimScope:
    return ClaimScope(
        claim_id=claim.claim_id,
        country_code=evidence.country_code,
        hs_code=evidence.hs_code,
        aggregation_grain=evidence.aggregation_grain,
    )
