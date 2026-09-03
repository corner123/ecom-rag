from __future__ import annotations

from trade_agent.entities.models import EvidenceCandidate


def candidate(**overrides) -> EvidenceCandidate:
    values = {
        "evidence_id": "evidence_1",
        "entity_id": "entity_1",
        "fact_type": "market_signal",
        "fact_value": "procurement grew",
        "source_type": "industry_news",
        "source_weight": 0.5,
        "content": "Harbor increased HS 850440 charger procurement in March",
        "content_hash": "a" * 64,
        "parent_document_hash": "b" * 64,
        "source_url": "https://newsroom.example/a",
        "canonical_url": "https://newsroom.example/original",
        "publish_time": "2026-03-01T00:00:00+00:00",
        "valid_from": "2026-03-01T00:00:00+00:00",
        "valid_to": None,
        "unit": None,
        "aggregation_grain": "monthly",
        "syndication_group_id": None,
    }
    values.update(overrides)
    return EvidenceCandidate.model_validate(values)


def test_canonical_url_and_syndication_do_not_multiply_sources() -> None:
    from trade_agent.entities.dedup import EvidenceDeduplicator

    clusters = EvidenceDeduplicator().cluster(
        [
            candidate(),
            candidate(evidence_id="evidence_2", source_url="https://news.example/syndicated"),
        ]
    )
    assert len(clusters) == 1
    assert "canonical_url" in clusters[0].reasons


def test_exact_content_hash_clusters_even_without_url() -> None:
    from trade_agent.entities.dedup import EvidenceDeduplicator

    clusters = EvidenceDeduplicator().cluster(
        [candidate(canonical_url=None, source_url=None), candidate(evidence_id="evidence_2", canonical_url=None, source_url=None)]
    )
    assert len(clusters) == 1
    assert "content_hash" in clusters[0].reasons


def test_near_duplicate_threshold_clusters_and_distinct_content_stays_separate() -> None:
    from trade_agent.entities.dedup import EvidenceDeduplicator

    original = candidate(canonical_url=None, source_url=None)
    syndicated = candidate(
        evidence_id="evidence_2",
        canonical_url=None,
        source_url=None,
        content_hash="c" * 64,
        content="Harbor increased HS 850440 charger procurement during March",
    )
    distinct = candidate(
        evidence_id="evidence_3",
        canonical_url=None,
        source_url=None,
        content_hash="d" * 64,
        content="Regulator published a new customs declaration rule",
    )
    clusters = EvidenceDeduplicator().cluster([original, syndicated, distinct])
    assert len(clusters) == 2
    assert any("near_duplicate" in cluster.reasons for cluster in clusters)
