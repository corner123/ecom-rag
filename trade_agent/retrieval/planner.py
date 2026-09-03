"""Typed, auditable planning between query intent and executable filters."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictStr, field_validator, model_validator

from trade_agent.retrieval.filters import RetrievalFilter
from trade_agent.schemas.source import FactType, SourceType


PLANNER_VERSION = "trade-retrieval-planner-v1"
_CONSTRAINT_CONFIDENCE_THRESHOLD = 0.70
_REGION_ALIASES = {
    "africa": "Africa",
    "asia": "Asia",
    "europe": "Europe",
    "middle east": "Middle East",
    "north america": "North America",
    "oceania": "Oceania",
    "south america": "South America",
}


class UnappliedConstraint(BaseModel):
    """A structured constraint deliberately kept out of retrieval filtering."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field: StrictStr
    value: str
    reason: Literal["low_extraction_confidence"]


class QueryIntent(BaseModel):
    """Validated structured query constraints extracted upstream."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: StrictStr
    region: StrictStr | None = None
    country_codes: tuple[StrictStr, ...] = ()
    hs_codes: tuple[StrictStr, ...] = ()
    entity_ids: tuple[StrictStr, ...] = ()
    source_types: tuple[SourceType | StrictStr, ...] = ()
    fact_types: tuple[FactType | StrictStr, ...] = ()
    published_after: datetime | None = None
    published_before: datetime | None = None
    is_synthetic: StrictBool | None = None
    extraction_confidence: StrictFloat = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @field_validator("published_after", "published_before")
    @classmethod
    def require_aware_times(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("publication filters must be timezone-aware")
        return value

    @model_validator(mode="after")
    def ordered_publication_range(self) -> "QueryIntent":
        if (
            self.published_after is not None
            and self.published_before is not None
            and self.published_after > self.published_before
        ):
            raise ValueError("published_after must not exceed published_before")
        return self


class RetrievalPlan(BaseModel):
    """The only filter form retrieval components are allowed to execute."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: StrictStr
    filter: RetrievalFilter
    extraction_confidence: StrictFloat
    applied_constraints: tuple[StrictStr, ...]
    unapplied_constraints: tuple[UnappliedConstraint, ...]
    planner_version: str = PLANNER_VERSION


class RetrievalPlanner:
    """Compile typed intent into a filter with an explicit audit trail."""

    version = PLANNER_VERSION

    def plan(self, intent: QueryIntent) -> RetrievalPlan:
        if type(intent) is not QueryIntent:
            raise TypeError("intent must be an exact QueryIntent instance")
        supplied = {
            "region": intent.region,
            "country_codes": intent.country_codes,
            "hs_codes": intent.hs_codes,
            "entity_ids": intent.entity_ids,
            "source_types": intent.source_types,
            "fact_types": intent.fact_types,
            "published_range": (
                None
                if intent.published_after is None and intent.published_before is None
                else (intent.published_after, intent.published_before)
            ),
            "is_synthetic": intent.is_synthetic,
        }
        confidence = intent.extraction_confidence
        if confidence >= _CONSTRAINT_CONFIDENCE_THRESHOLD:
            filter_ = RetrievalFilter(
                region=_normalize_region(intent.region),
                country_codes=intent.country_codes,
                hs_codes=intent.hs_codes,
                entity_ids=intent.entity_ids,
                source_types=tuple(
                    SourceType(value.value if isinstance(value, SourceType) else value)
                    for value in intent.source_types
                ),
                fact_types=tuple(
                    FactType(value.value if isinstance(value, FactType) else value)
                    for value in intent.fact_types
                ),
                published_after=intent.published_after,
                published_before=intent.published_before,
                is_synthetic=intent.is_synthetic,
            )
            applied = tuple(field for field, value in supplied.items() if value is not None and value != ())
            unapplied: tuple[UnappliedConstraint, ...] = ()
        else:
            filter_ = RetrievalFilter()
            applied = ()
            unapplied = tuple(
                UnappliedConstraint(field=field, value=_trace_value(value), reason="low_extraction_confidence")
                for field, value in supplied.items()
                if value is not None and value != ()
            )
        return RetrievalPlan(
            query=intent.query,
            filter=filter_,
            extraction_confidence=confidence,
            applied_constraints=applied,
            unapplied_constraints=unapplied,
        )


def candidate_chunk_ids(
    chunks: Mapping[str, Mapping[str, Any]],
    filter_: RetrievalFilter,
) -> set[str]:
    """Apply the same scalar semantics to any in-memory candidate universe."""

    if type(filter_) is not RetrievalFilter:
        raise TypeError("filter_ must be an exact RetrievalFilter instance")
    def scalar_value(value: Any) -> Any:
        return value.value if hasattr(value, "value") else value

    selected: set[str] = set()
    for chunk_id, metadata in chunks.items():
        if filter_.region is not None and metadata.get("region") != filter_.region:
            continue
        if filter_.country_codes and metadata.get("country_code") not in filter_.country_codes:
            continue
        if filter_.hs_codes and metadata.get("hs_code") not in filter_.hs_codes:
            continue
        if filter_.entity_ids and metadata.get("entity_id") not in filter_.entity_ids:
            continue
        if filter_.source_types and scalar_value(metadata.get("source_type")) not in {
            scalar_value(value) for value in filter_.source_types
        }:
            continue
        if filter_.fact_types and scalar_value(metadata.get("fact_type")) not in {
            scalar_value(value) for value in filter_.fact_types
        }:
            continue
        publish_epoch = metadata.get("publish_time_epoch")
        if filter_.published_after is not None and (
            not isinstance(publish_epoch, int) or publish_epoch < int(filter_.published_after.timestamp())
        ):
            continue
        if filter_.published_before is not None and (
            not isinstance(publish_epoch, int) or publish_epoch > int(filter_.published_before.timestamp())
        ):
            continue
        if filter_.is_synthetic is not None and metadata.get("is_synthetic") is not filter_.is_synthetic:
            continue
        selected.add(chunk_id)
    return selected


def _normalize_region(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.strip().split())
    return _REGION_ALIASES.get(normalized.casefold(), normalized)


def _trace_value(value: Any) -> str:
    if isinstance(value, tuple):
        return ",".join(_trace_value(item) for item in value)
    if hasattr(value, "value"):
        return str(value.value)
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    return str(value)
