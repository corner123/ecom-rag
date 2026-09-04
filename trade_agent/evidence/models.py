"""Strict shared Evidence contracts, initially populated by SQL execution."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RawRecordLocator(BaseModel):
    """The reviewed composite unique key for one source trade record."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source_id: int = Field(gt=0)
    raw_record_id: str = Field(min_length=1, max_length=100)


class EvidenceLocator(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str
    table: str = "trade_records"
    scope: Literal["bounded_predicate_population"] = "bounded_predicate_population"
    raw_record_locators: tuple[RawRecordLocator, ...]
    raw_record_locators_truncated: bool = False


class SqlProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str
    normalized_sql: str
    bound_filter_names: tuple[str, ...]
    schema_fingerprint: str
    dataset_id: str
    is_synthetic: bool
    effective_start_date: date
    effective_end_date: date
    aggregation_grain: tuple[str, ...]
    time_grain: str
    execution_ms: float = Field(ge=0)
    row_count: int = Field(ge=0)
    result_hash: str
    estimated_scan_rows: int = Field(ge=0)
    max_execution_time_ms: int = Field(gt=0)
    client_timeout_ms: int = Field(gt=0)


class Evidence(BaseModel):
    """One bounded fact bundle with stable identity and replay provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    entity_id: str | None = None
    fact_type: str
    source_type: str
    source_weight: float = Field(ge=0, le=1)
    content: str
    source_url: str | None = None
    locator: EvidenceLocator
    raw_record_id: str | None = None
    publish_time: datetime | None = None
    valid_from: date | datetime | None = None
    valid_to: date | datetime | None = None
    confidence: float = Field(ge=0, le=1)
    is_synthetic: bool
    retrieval_provenance: None = None
    sql_provenance: SqlProvenance
    conflict_group_id: str | None = None

    @model_validator(mode="after")
    def ordered_validity(self) -> "Evidence":
        if self.valid_from is not None and self.valid_to is not None:
            start = self.valid_from.date() if isinstance(self.valid_from, datetime) else self.valid_from
            end = self.valid_to.date() if isinstance(self.valid_to, datetime) else self.valid_to
            if start > end:
                raise ValueError("valid_from must not exceed valid_to")
        return self
