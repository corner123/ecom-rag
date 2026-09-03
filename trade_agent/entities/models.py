from __future__ import annotations

from datetime import datetime
from typing import Literal
import re

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictStr, field_validator


_SHA256 = re.compile(r"[0-9a-f]{64}")
_COUNTRY = re.compile(r"[A-Za-z]{2}")


class EntityRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_id: StrictStr
    canonical_name: StrictStr
    normalized_name: StrictStr
    country_code: StrictStr | None = None
    aliases: tuple[StrictStr, ...] = ()
    website_domain: StrictStr | None = None
    registration_id: StrictStr | None = None

    @field_validator("entity_id", "canonical_name", "normalized_name")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("country_code")
    @classmethod
    def country(cls, value: str | None) -> str | None:
        if value is not None and not _COUNTRY.fullmatch(value):
            raise ValueError("country_code must contain two letters")
        return value.upper() if value else None


class EntityMention(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: StrictStr
    country_code: StrictStr | None = None
    aliases: tuple[StrictStr, ...] = ()
    website_domain: StrictStr | None = None
    registration_id: StrictStr | None = None

    @field_validator("name")
    @classmethod
    def nonblank_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must not be blank")
        return value

    @field_validator("country_code")
    @classmethod
    def country(cls, value: str | None) -> str | None:
        if value is not None and not _COUNTRY.fullmatch(value):
            raise ValueError("country_code must contain two letters")
        return value.upper() if value else None


class EvidenceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: StrictStr
    entity_id: StrictStr
    fact_type: StrictStr
    fact_value: StrictStr
    source_type: StrictStr
    source_weight: StrictFloat = Field(ge=0, le=1)
    content: StrictStr
    content_hash: StrictStr
    parent_document_hash: StrictStr
    source_url: StrictStr | None = None
    canonical_url: StrictStr | None = None
    publish_time: datetime
    valid_from: datetime
    valid_to: datetime | None
    unit: StrictStr | None = None
    aggregation_grain: StrictStr | None = None
    syndication_group_id: StrictStr | None = None

    @field_validator("evidence_id", "entity_id", "fact_type", "fact_value", "source_type")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("content_hash", "parent_document_hash")
    @classmethod
    def sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("hash must be a 64-character hexadecimal digest")
        return value.lower()


class EvidenceCluster(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cluster_id: StrictStr
    candidates: tuple[EvidenceCandidate, ...] = Field(min_length=1)
    reasons: frozenset[StrictStr]


ArbitrationStatus = Literal["supported", "superseded", "conflicted"]


class ArbitrationFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: StrictStr
    entity_id: StrictStr
    fact_type: StrictStr
    fact_value: StrictStr
    status: ArbitrationStatus
    reasons: frozenset[StrictStr]


class ArbitrationOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    facts: tuple[ArbitrationFact, ...]
    groups: tuple[StrictStr, ...] = ()
    lead_status: Literal["low", "medium", "high", "block"] = "low"
