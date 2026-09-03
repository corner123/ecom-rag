from __future__ import annotations

from datetime import datetime, timezone

from trade_agent.entities.dedup import EvidenceDeduplicator
from trade_agent.entities.models import EvidenceCandidate
from trade_agent.entities.conflicts import ConflictArbitrator


AS_OF = datetime(2026, 6, 1, tzinfo=timezone.utc)


def candidate(**overrides) -> EvidenceCandidate:
    values = {
        "evidence_id": "history",
        "entity_id": "entity_1",
        "fact_type": "purchase_activity",
        "fact_value": "active",
        "source_type": "customs_profile",
        "source_weight": 0.95,
        "content": "March procurement increased",
        "content_hash": "a" * 64,
        "parent_document_hash": "b" * 64,
        "source_url": "https://customs.synthetic.example/march",
        "canonical_url": "https://customs.synthetic.example/march",
        "publish_time": "2026-04-01T00:00:00+00:00",
        "valid_from": "2026-03-01T00:00:00+00:00",
        "valid_to": "2026-03-31T23:59:59+00:00",
        "unit": "USD",
        "aggregation_grain": "company_hs_month",
        "syndication_group_id": None,
    }
    values.update(overrides)
    return EvidenceCandidate.model_validate(values)


def test_historical_activity_and_current_closure_can_both_be_true() -> None:
    clusters = EvidenceDeduplicator().cluster(
        [
            candidate(),
            candidate(
                evidence_id="closure",
                fact_type="company_status",
                fact_value="closed",
                source_type="official_website",
                source_weight=0.9,
                canonical_url="https://company.example/closure",
                source_url="https://company.example/closure",
                content_hash="c" * 64,
                content="The company closed on 2026-05-01",
                publish_time="2026-05-01T00:00:00+00:00",
                valid_from="2026-05-01T00:00:00+00:00",
                valid_to=None,
                unit=None,
            ),
        ]
    )
    outcome = ConflictArbitrator().arbitrate(clusters, as_of=AS_OF)
    assert {fact.status for fact in outcome.facts} == {"supported"}
    assert outcome.lead_status == "low"


def test_current_evidence_supersedes_expired_same_fact() -> None:
    clusters = EvidenceDeduplicator().cluster(
        [
            candidate(),
            candidate(
                evidence_id="current",
                fact_value="inactive",
                canonical_url="https://customs.synthetic.example/may",
                source_url="https://customs.synthetic.example/may",
                content_hash="c" * 64,
                content="May procurement stopped",
                publish_time="2026-05-01T00:00:00+00:00",
                valid_from="2026-05-01T00:00:00+00:00",
                valid_to=None,
            ),
        ]
    )
    outcome = ConflictArbitrator().arbitrate(clusters, as_of=AS_OF)
    by_id = {fact.evidence_id: fact for fact in outcome.facts}
    assert by_id["history"].status == "superseded"
    assert by_id["current"].status == "supported"


def test_same_grain_high_authority_conflict_is_unresolved() -> None:
    clusters = EvidenceDeduplicator().cluster(
        [
            candidate(valid_to=None),
            candidate(
                evidence_id="closure",
                fact_value="closed",
                source_type="regulator",
                source_weight=0.98,
                canonical_url="https://regulator.example/closed",
                source_url="https://regulator.example/closed",
                content_hash="c" * 64,
                content="The entity is closed",
                valid_to=None,
            ),
        ]
    )
    outcome = ConflictArbitrator().arbitrate(clusters, as_of=AS_OF)
    assert all(fact.status == "conflicted" for fact in outcome.facts)
    assert outcome.lead_status == "block"
    assert any("high_authority_conflict" in fact.reasons for fact in outcome.facts)
