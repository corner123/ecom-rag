"""Canonical, strict contracts for trade-source ingestion and chunks."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import Enum
from typing import Any, TypeAlias
from urllib.parse import urlparse

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator, model_validator

# Pydantic's JSON serializer enforces serializability at the boundary; keeping
# this alias open avoids recursive-schema limitations while allowing structured
# parser units and aggregation attributes.
JsonValue: TypeAlias = Any
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_HS = re.compile(r"^[0-9]{4,10}$")


class FileType(str, Enum):
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    JSONL = "jsonl"
    PDF = "pdf"
    GENERATED_PROFILE = "generated_profile"


class SourceType(str, Enum):
    OFFICIAL_WEBSITE = "official_website"
    B2B = "b2b"
    INDUSTRY_NEWS = "industry_news"
    SOCIAL = "social"
    REGULATOR = "regulator"
    CUSTOMS_PROFILE = "customs_profile"
    TRADE_LEDGER = "trade_ledger"


class FactType(str, Enum):
    TRADE_ACTIVITY = "trade_activity"
    PRODUCT_OFFERING = "product_offering"
    COMPANY_STATUS = "company_status"
    REGULATION = "regulation"
    RISK = "risk"
    CONTACT = "contact"
    MARKET_SIGNAL = "market_signal"


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value


def _sha(value: str, field: str) -> str:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase or uppercase 64-character SHA-256 hex digest")
    return value.lower()


def _reserved_synthetic_url(value: AnyUrl) -> bool:
    host = (urlparse(str(value)).hostname or "").lower().rstrip(".")
    return host == "example" or host.endswith(".example") or host == "synthetic.example" or host.endswith(".synthetic.example")


def content_sha256(value: str | bytes) -> str:
    """Return the SHA-256 digest of UTF-8 text or bytes; reject implicit coercion."""
    if isinstance(value, str):
        payload = value.encode("utf-8")
    elif isinstance(value, bytes):
        payload = value
    else:
        raise TypeError("content_sha256 accepts str or bytes only")
    return hashlib.sha256(payload).hexdigest()


def stable_id(prefix: str, *parts: str) -> str:
    """Build a deterministic ID using canonical length-prefixed UTF-8 components."""
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,31}", prefix):
        raise ValueError("prefix must start with a letter and contain only letters, numbers, '_' or '-'")
    if any(not isinstance(part, str) for part in parts):
        raise TypeError("stable_id parts must be strings")
    canonical = json.dumps(
        [[len(part.encode("utf-8")), part] for part in parts],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{digest}"


class SourceLocator(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: int | None = Field(default=None, ge=1)
    block: int | None = Field(default=None, ge=0)
    table: StrictStr | None = None
    section: StrictStr | None = None
    post: StrictStr | None = None
    row: int | None = Field(default=None, ge=0)
    profile: StrictStr | None = None
    sql: StrictStr | None = None
    raw: StrictStr | None = None

    @field_validator("table", "section", "post", "profile", "sql", "raw")
    @classmethod
    def strings_nonblank(cls, value: str | None) -> str | None:
        return None if value is None else _nonblank(value)


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class SourceRecord(_StrictBase):
    source_id: StrictStr
    source_type: SourceType
    file_type: FileType
    url: AnyUrl
    title: StrictStr
    language: StrictStr
    fetched_at: datetime
    is_synthetic: StrictBool
    canonical_url: AnyUrl | None = None
    content_hash: StrictStr | None = None
    license_scope: StrictStr | None = None

    @field_validator("source_id", "title", "language", "license_scope")
    @classmethod
    def nonblank_fields(cls, value: str | None) -> str | None:
        return _nonblank(value) if value is not None else None
    _time = field_validator("fetched_at")(_aware)
    _hash = field_validator("content_hash")(classmethod(lambda cls, v: _sha(v, "content_hash") if v is not None else v))

    @model_validator(mode="after")
    def synthetic_urls_are_reserved(self) -> "SourceRecord":
        if self.is_synthetic:
            if not _reserved_synthetic_url(self.url):
                raise ValueError("synthetic sources must use reserved example.com or synthetic.example hosts")
        return self


class DocumentRecord(_StrictBase):
    document_id: StrictStr
    source_id: StrictStr
    file_type: FileType
    source_type: SourceType
    title: StrictStr
    language: StrictStr
    content: StrictStr
    content_hash: StrictStr
    fetched_at: datetime
    is_synthetic: StrictBool
    units: list[dict[str, JsonValue]] = Field(default_factory=list)
    attributes: dict[str, JsonValue] = Field(default_factory=dict)
    source_url: AnyUrl | None = None
    canonical_url: AnyUrl | None = None

    @field_validator("document_id", "source_id", "title", "language", "content")
    @classmethod
    def nonblank_fields(cls, value: str) -> str:
        return _nonblank(value)
    _time = field_validator("fetched_at")(_aware)
    _hash = field_validator("content_hash")(classmethod(lambda cls, v: _sha(v, "content_hash")))

    @model_validator(mode="after")
    def content_hash_matches(self) -> "DocumentRecord":
        if content_sha256(self.content) != self.content_hash:
            raise ValueError("content_hash does not match content")
        if self.is_synthetic and any(url is not None and not _reserved_synthetic_url(url) for url in (self.source_url, self.canonical_url)):
            raise ValueError("synthetic documents must use reserved example hosts")
        return self


class ChunkMetadata(_StrictBase):
    chunk_id: StrictStr
    document_id: StrictStr
    entity_id: StrictStr | None = None
    company_name: StrictStr | None = None
    normalized_name: StrictStr | None = None
    country_code: StrictStr | None = None
    region: StrictStr | None = None
    hs_code: StrictStr | None = None
    product_name: StrictStr | None = None
    sku: StrictStr | None = None
    file_type: FileType
    source_type: SourceType
    source_weight: float = Field(ge=0, le=1)
    fact_type: FactType | None = None
    publish_time: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    ingested_at: datetime
    source_url: AnyUrl | None = None
    canonical_url: AnyUrl | None = None
    source_locator: SourceLocator
    raw_record_id: StrictStr | None = None
    aggregation_info: dict[str, JsonValue] | None = None
    content_hash: StrictStr
    parent_document_hash: StrictStr
    language: StrictStr
    ocr_confidence: float | None = Field(default=None, ge=0, le=1)
    is_synthetic: StrictBool
    license_scope: StrictStr | None = None
    dedupe_cluster_id: StrictStr | None = None

    @field_validator("chunk_id", "document_id", "entity_id", "company_name", "normalized_name", "region", "product_name", "sku", "raw_record_id", "language", "license_scope", "dedupe_cluster_id")
    @classmethod
    def nonblank_optional_fields(cls, value: str | None) -> str | None:
        return _nonblank(value) if value is not None else None

    @field_validator("country_code")
    @classmethod
    def normalize_country(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not re.fullmatch(r"[A-Za-z]{2}", value):
            raise ValueError("country_code must be ISO 3166-1 alpha-2")
        return value.upper()

    @field_validator("hs_code")
    @classmethod
    def validate_hs(cls, value: str | None) -> str | None:
        if value is not None and not _HS.fullmatch(value):
            raise ValueError("hs_code must be a digit string")
        return value

    @field_validator("content_hash", "parent_document_hash")
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _sha(value, info.field_name)
    _times = field_validator("publish_time", "valid_from", "valid_to", "ingested_at")(_aware)

    @model_validator(mode="after")
    def valid_range(self) -> "ChunkMetadata":
        if self.valid_from is not None and self.valid_to is not None and self.valid_from > self.valid_to:
            raise ValueError("valid_from must be before or equal to valid_to")
        if self.is_synthetic and any(url is not None and not _reserved_synthetic_url(url) for url in (self.source_url, self.canonical_url)):
            raise ValueError("synthetic chunks must use reserved example hosts")
        return self


class ChunkRecord(_StrictBase):
    content: StrictStr
    metadata: ChunkMetadata

    @field_validator("content")
    @classmethod
    def content_nonblank(cls, value: str) -> str:
        return _nonblank(value)

    @model_validator(mode="after")
    def content_hash_matches_metadata(self) -> "ChunkRecord":
        if content_sha256(self.content) != self.metadata.content_hash:
            raise ValueError("metadata.content_hash does not match chunk content")
        return self
