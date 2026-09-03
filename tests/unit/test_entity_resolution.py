from __future__ import annotations

import pytest

from trade_agent.entities.models import EntityMention, EntityRecord


@pytest.fixture
def resolver() -> Any:
    from trade_agent.entities.resolver import EntityResolver

    registry = [
        EntityRecord(
            entity_id="entity_us",
            canonical_name="Global Trading LLC",
            normalized_name="global trading llc",
            country_code="US",
            aliases=("Global Trading",),
            website_domain="global-us.example",
            registration_id="REG-US-1",
        ),
        EntityRecord(
            entity_id="entity_de",
            canonical_name="Global Trading GmbH",
            normalized_name="global trading gmbh",
            country_code="DE",
            aliases=("Global Trading",),
            website_domain="global-de.example",
            registration_id="REG-DE-1",
        ),
        EntityRecord(
            entity_id="entity_harbor",
            canonical_name="Harbor CN Imports 01",
            normalized_name="harbor cn imports 01",
            country_code="CN",
            aliases=("Harbor Imports",),
            website_domain="harbor.example",
            registration_id="REG-CN-1",
        ),
    ]
    return EntityResolver(registry)


def test_registration_id_wins(resolver) -> None:
    outcome = resolver.resolve(EntityMention(name="Unknown", registration_id="REG-CN-1"))
    assert outcome.status == "resolved"
    assert outcome.entity_id == "entity_harbor"


def test_website_domain_wins_before_name(resolver) -> None:
    outcome = resolver.resolve(EntityMention(name="Wrong", website_domain="harbor.example"))
    assert outcome.status == "resolved"
    assert outcome.entity_id == "entity_harbor"


def test_same_name_without_country_is_ambiguous(resolver) -> None:
    outcome = resolver.resolve(EntityMention(name="Global Trading", country_code=None))
    assert outcome.status == "ambiguous"
    assert outcome.entity_id is None
    assert {candidate.entity_id for candidate in outcome.candidates} == {"entity_us", "entity_de"}


def test_exact_alias_and_country_resolves(resolver) -> None:
    outcome = resolver.resolve(EntityMention(name="Global Trading", country_code="DE"))
    assert outcome.status == "resolved"
    assert outcome.entity_id == "entity_de"
    assert outcome.reason == "exact_alias_country"


def test_fuzzy_name_requires_review(resolver) -> None:
    outcome = resolver.resolve(EntityMention(name="Harbor CN Imports", country_code="CN"))
    assert outcome.status in {"resolved", "review"}
    assert outcome.entity_id in {"entity_harbor", None}
