"""Allowlisted metadata filters and deterministic Milvus compilation.

Only :class:`RetrievalFilter` can cross the public dense-search boundary.  The
compiler never accepts an already-built expression, so neither an LLM nor an
API caller can smuggle raw Milvus syntax into a query.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)

from trade_agent.schemas.source import FactType, SourceType

FILTER_EXPRESSION_VERSION = "trade-filter-v1"
_COUNTRY = re.compile(r"[A-Za-z]{2}")
_HS_CODE = re.compile(r"[0-9]{4,10}")
_ENTITY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9:_.-]{0,127}")


def _ordered_unique(values: Any, *, field: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError(f"{field} must be a list or tuple")
    if any(not isinstance(value, (str, SourceType, FactType)) for value in values):
        raise ValueError(f"{field} items must be strings or supported enums")
    unique: dict[tuple[type[Any], str], Any] = {}
    for value in values:
        unique[(type(value), str(value))] = value
    return tuple(sorted(unique.values(), key=lambda value: str(value)))


def _aware_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("publication filters must be timezone-aware")
    return value.astimezone(timezone.utc)


class RetrievalFilter(BaseModel):
    """The complete scalar-filter allowlist for the trade chunk collection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    region: StrictStr | None = None
    country_codes: tuple[StrictStr, ...] = ()
    hs_codes: tuple[StrictStr, ...] = ()
    entity_ids: tuple[StrictStr, ...] = ()
    source_types: tuple[SourceType, ...] = ()
    fact_types: tuple[FactType, ...] = ()
    published_after: datetime | None = None
    published_before: datetime | None = None
    is_synthetic: StrictBool | None = None

    @field_validator(
        "country_codes",
        "hs_codes",
        "entity_ids",
        "source_types",
        "fact_types",
        mode="before",
    )
    @classmethod
    def stable_sequences(cls, value: Any, info: Any) -> tuple[Any, ...]:
        return _ordered_unique(value, field=info.field_name)

    @field_validator("region")
    @classmethod
    def safe_region(cls, value: str | None) -> str | None:
        if value is not None:
            if not value.strip() or len(value.encode("utf-8")) > 128:
                raise ValueError("region must be nonblank and at most 128 UTF-8 bytes")
            if any(unicodedata.category(character).startswith("C") for character in value):
                raise ValueError("region must not contain control characters")
        return value

    @field_validator("country_codes")
    @classmethod
    def normalized_countries(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not _COUNTRY.fullmatch(value) for value in values):
            raise ValueError("country codes must use two ASCII letters")
        return tuple(sorted({value.upper() for value in values}))

    @field_validator("hs_codes")
    @classmethod
    def valid_hs_codes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not _HS_CODE.fullmatch(value) for value in values):
            raise ValueError("HS codes must contain 4 to 10 digits")
        return values

    @field_validator("entity_ids")
    @classmethod
    def safe_entity_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not _ENTITY_ID.fullmatch(value) for value in values):
            raise ValueError("entity IDs contain unsupported characters")
        return values

    @field_validator("published_after", "published_before")
    @classmethod
    def aware_times(cls, value: datetime | None) -> datetime | None:
        return _aware_utc(value)

    @model_validator(mode="after")
    def ordered_range(self) -> "RetrievalFilter":
        if (
            self.published_after is not None
            and self.published_before is not None
            and self.published_after > self.published_before
        ):
            raise ValueError("published_after must not exceed published_before")
        return self


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _string_list(values: tuple[str, ...]) -> str:
    return "[" + ",".join(_quote(value) for value in values) + "]"


def compile_filter(filter_: RetrievalFilter) -> str:
    """Compile a validated filter in a stable order; raw strings are rejected."""

    if type(filter_) is not RetrievalFilter:
        raise TypeError("filter_ must be an exact RetrievalFilter instance")
    clauses: list[str] = []
    if filter_.region is not None:
        clauses.append(f"region == {_quote(filter_.region)}")
    for field_name, values in (
        ("country_code", filter_.country_codes),
        ("hs_code", filter_.hs_codes),
        ("entity_id", filter_.entity_ids),
        ("source_type", tuple(value.value for value in filter_.source_types)),
        ("fact_type", tuple(value.value for value in filter_.fact_types)),
    ):
        if values:
            clauses.append(f"{field_name} in {_string_list(values)}")
    if filter_.published_after is not None:
        clauses.append(f"publish_time_epoch >= {int(filter_.published_after.timestamp())}")
    if filter_.published_before is not None:
        clauses.append(f"publish_time_epoch <= {int(filter_.published_before.timestamp())}")
    if filter_.is_synthetic is not None:
        clauses.append(f"is_synthetic == {'true' if filter_.is_synthetic else 'false'}")
    return " and ".join(clauses)


@dataclass(frozen=True, slots=True)
class CompiledFilter:
    """Milvus template plus typed bindings and a stable redacted trace form."""

    expression: str
    parameters: dict[str, object]
    audit_expression: str
    version: str = FILTER_EXPRESSION_VERSION


def compile_filter_binding(filter_: RetrievalFilter) -> CompiledFilter:
    """Compile an executable expression using Milvus template parameters.

    ``audit_expression`` is never executed.  It exists only for deterministic
    retrieval traces and uses JSON escaping after the Pydantic allowlist.
    """

    if type(filter_) is not RetrievalFilter:
        raise TypeError("filter_ must be an exact RetrievalFilter instance")
    clauses: list[str] = []
    parameters: dict[str, object] = {}

    def scalar(field: str, value: object) -> None:
        name = f"{field}_0"
        clauses.append(f"{field} == {{{name}}}")
        parameters[name] = value

    def sequence(field: str, values: tuple[str, ...]) -> None:
        name = f"{field}_0"
        clauses.append(f"{field} in {{{name}}}")
        parameters[name] = list(values)

    if filter_.region is not None:
        scalar("region", filter_.region)
    for field_name, values in (
        ("country_code", filter_.country_codes),
        ("hs_code", filter_.hs_codes),
        ("entity_id", filter_.entity_ids),
        ("source_type", tuple(value.value for value in filter_.source_types)),
        ("fact_type", tuple(value.value for value in filter_.fact_types)),
    ):
        if values:
            sequence(field_name, values)
    if filter_.published_after is not None:
        name = "published_after_0"
        clauses.append(f"publish_time_epoch >= {{{name}}}")
        parameters[name] = int(filter_.published_after.timestamp())
    if filter_.published_before is not None:
        name = "published_before_0"
        clauses.append(f"publish_time_epoch <= {{{name}}}")
        parameters[name] = int(filter_.published_before.timestamp())
    if filter_.is_synthetic is not None:
        scalar("is_synthetic", filter_.is_synthetic)
    return CompiledFilter(
        expression=" and ".join(clauses),
        parameters=parameters,
        audit_expression=compile_filter(filter_),
    )
