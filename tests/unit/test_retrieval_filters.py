from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from trade_agent.retrieval.filters import compile_filter_binding
from trade_agent.retrieval.planner import QueryIntent, RetrievalPlanner


def test_unknown_filter_field_is_rejected() -> None:
    from trade_agent.retrieval.filters import RetrievalFilter

    with pytest.raises(ValidationError):
        RetrievalFilter.model_validate({"sql": "drop table companies"})


def test_planner_preserves_identifiers_and_normalizes_controlled_values() -> None:
    from trade_agent.retrieval.filters import RetrievalFilter
    from trade_agent.schemas.source import SourceType

    plan = RetrievalPlanner().plan(
        QueryIntent(
            query="查询 Harbor 采购 850440",
            region="north america",
            country_codes=["us"],
            hs_codes=["850440"],
            entity_ids=["entity_harbor_01"],
            source_types=["official_website"],
            fact_types=["trade_activity"],
            published_after=datetime(2026, 1, 1, tzinfo=timezone.utc),
            published_before=datetime(2026, 12, 31, tzinfo=timezone.utc),
            extraction_confidence=0.92,
        )
    )
    assert plan.filter == RetrievalFilter(
        region="North America",
        country_codes=["US"],
        hs_codes=["850440"],
        entity_ids=["entity_harbor_01"],
        source_types=[SourceType.OFFICIAL_WEBSITE],
        fact_types=["trade_activity"],
        published_after=datetime(2026, 1, 1, tzinfo=timezone.utc),
        published_before=datetime(2026, 12, 31, tzinfo=timezone.utc),
    )
    assert plan.applied_constraints == (
        "region",
        "country_codes",
        "hs_codes",
        "entity_ids",
        "source_types",
        "fact_types",
        "published_range",
    )
    assert plan.unapplied_constraints == ()
    assert compile_filter_binding(plan.filter).expression


def test_low_confidence_constraints_remain_unapplied_and_visible() -> None:
    plan = RetrievalPlanner().plan(
        QueryIntent(
            query="recent charger demand",
            region="Europe",
            hs_codes=["850440"],
            extraction_confidence=0.41,
        )
    )
    assert plan.filter.model_dump(exclude_none=True, exclude_defaults=True) == {}
    assert plan.applied_constraints == ()
    assert [item.field for item in plan.unapplied_constraints] == ["region", "hs_codes"]
    assert all(item.reason == "low_extraction_confidence" for item in plan.unapplied_constraints)


def test_planner_rejects_contradictory_publication_range() -> None:
    with pytest.raises(ValidationError):
        RetrievalPlanner().plan(
            QueryIntent(
                query="charger demand",
                published_after=datetime(2026, 2, 1, tzinfo=timezone.utc),
                published_before=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        )
