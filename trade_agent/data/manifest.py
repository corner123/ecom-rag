"""Canonical, immutable ingestion build manifests."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from collections import Counter
import re
from pathlib import PurePosixPath
from copy import deepcopy

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, field_validator, model_validator

from trade_agent.data.quarantine import QuarantineRecord
from trade_agent.schemas.source import ChunkRecord, DocumentRecord, FileType, SourceLocator, SourceType, stable_id

METADATA_SCHEMA_VERSION = "task6-source-metadata-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False):
        if update is None:
            return super().model_copy(deep=deep)
        value = self.model_dump(mode="python")
        if deep:
            value = deepcopy(value)
        value.update(update)
        return type(self).model_validate(value)


class SourceBuildRecord(_FrozenModel):
    source_id: StrictStr
    path: StrictStr
    source_type: StrictStr | None = None
    file_type: StrictStr
    actual_content_hash: StrictStr | None = None
    manifest_content_hash: StrictStr | None = None
    status: StrictStr
    parser_backend: StrictStr | None = None
    degraded: StrictBool = False

    @field_validator("source_id", "path", "file_type", "status")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("parser_backend")
    @classmethod
    def backend_nonblank_when_present(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("parser backend must not be blank")
        return value

    @field_validator("actual_content_hash", "manifest_content_hash")
    @classmethod
    def hashes(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("source hashes must be lowercase SHA-256 digests")
        return value

    @model_validator(mode="after")
    def source_consistency(self) -> "SourceBuildRecord":
        if self.status not in {"parsed", "quarantined"}:
            raise ValueError("source status is invalid")
        if not re.fullmatch(r"source_[0-9a-f]{32}", self.source_id):
            raise ValueError("source ID must be stable")
        if self.source_type not in {item.value for item in SourceType}:
            raise ValueError("source type is unsupported")
        if self.file_type not in {item.value for item in FileType} | {"unsupported"}:
            raise ValueError("file type is unsupported")
        if self.path.startswith("/") or ".." in PurePosixPath(self.path).parts:
            raise ValueError("source path must be safe and relative")
        if self.status == "parsed" and (self.actual_content_hash != self.manifest_content_hash or not self.parser_backend):
            raise ValueError("parsed source requires matching hashes and parser backend")
        return self


class CountEntry(_FrozenModel):
    name: StrictStr
    count: StrictInt = Field(ge=0)

    @field_validator("name")
    @classmethod
    def count_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("count name must not be blank")
        return value


class ParserBackends(_FrozenModel):
    document_router_version: StrictStr
    chunk_router_version: StrictStr
    max_tokens: StrictInt = Field(gt=0)
    overlap_tokens: StrictInt = Field(ge=0)
    mineru_statuses: tuple[StrictStr, ...]
    degraded_components: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def ordered_values(self) -> "ParserBackends":
        if not self.document_router_version.strip() or not self.chunk_router_version.strip() or not self.mineru_statuses:
            raise ValueError("backend versions and MinerU statuses must be nonblank")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap must be below max")
        for values in (self.mineru_statuses, self.degraded_components):
            if tuple(sorted(values)) != values or len(set(values)) != len(values) or any(not value.strip() for value in values):
                raise ValueError("backend tuples must be nonblank, sorted, and unique")
        return self


class DocumentSnapshot(_FrozenModel):
    """An immutable canonical payload that can restore the full router record."""
    document_id: StrictStr
    payload: StrictStr

    @classmethod
    def freeze(cls, record: DocumentRecord) -> "DocumentSnapshot":
        return cls(document_id=record.document_id, payload=canonical_json(record.model_dump(mode="json")))

    def restore(self) -> DocumentRecord:
        return DocumentRecord.model_validate_json(self.payload)

    @model_validator(mode="after")
    def matches_payload(self) -> "DocumentSnapshot":
        restored = self.restore()
        if self.payload != canonical_json(restored.model_dump(mode="json")):
            raise ValueError("document snapshot payload must be canonical JSON")
        if restored.document_id != self.document_id:
            raise ValueError("document snapshot ID does not match payload")
        return self


class ChunkSnapshot(_FrozenModel):
    """An immutable canonical payload that can restore the full chunk/content."""
    chunk_id: StrictStr
    content: StrictStr
    payload: StrictStr

    @classmethod
    def freeze(cls, record: ChunkRecord) -> "ChunkSnapshot":
        return cls(chunk_id=record.metadata.chunk_id, content=record.content, payload=canonical_json(record.model_dump(mode="json")))

    def restore(self) -> ChunkRecord:
        return ChunkRecord.model_validate_json(self.payload)

    @model_validator(mode="after")
    def matches_payload(self) -> "ChunkSnapshot":
        record = self.restore()
        if self.payload != canonical_json(record.model_dump(mode="json")):
            raise ValueError("chunk snapshot payload must be canonical JSON")
        if record.metadata.chunk_id != self.chunk_id or record.content != self.content:
            raise ValueError("chunk snapshot fields do not match payload")
        return self


class BuildManifest(_FrozenModel):
    build_id: StrictStr
    config_hash: StrictStr
    sources: tuple[SourceBuildRecord, ...]
    documents: tuple[DocumentSnapshot, ...]
    chunks: tuple[ChunkSnapshot, ...]
    quarantined: tuple[QuarantineRecord, ...]
    counts_by_source_type: tuple[CountEntry, ...]
    counts_by_file_type: tuple[CountEntry, ...]
    parser_backends: ParserBackends
    metadata_complete: StrictBool
    fingerprint: StrictStr
    metadata_schema_version: StrictStr = METADATA_SCHEMA_VERSION

    @field_validator("build_id", "config_hash", "metadata_schema_version")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(document.document_id for document in self.documents)

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(chunk.chunk_id for chunk in self.chunks)

    @property
    def source_type_counts(self) -> dict[str, int]:
        return {entry.name: entry.count for entry in self.counts_by_source_type}

    @property
    def file_type_counts(self) -> dict[str, int]:
        return {entry.name: entry.count for entry in self.counts_by_file_type}

    def _fingerprint_value(self) -> dict[str, Any]:
        documents = [json.loads(item.payload) for item in self.documents]
        chunks = [json.loads(item.payload) for item in self.chunks]
        for item in documents:
            item.pop("fetched_at", None)
        for item in chunks:
            item["metadata"].pop("ingested_at", None)
        return {
            "config_hash": self.config_hash,
            "parser_backends": self.parser_backends.model_dump(mode="json"),
            "metadata_schema_version": self.metadata_schema_version,
            "sources": [item.model_dump(mode="json") for item in self.sources],
            "documents": documents,
            "chunks": chunks,
            "quarantined": [item.model_dump(mode="json") for item in self.quarantined],
        }

    @model_validator(mode="after")
    def verify_integrity(self) -> "BuildManifest":
        if self.metadata_schema_version != METADATA_SCHEMA_VERSION:
            raise ValueError("metadata schema version is unsupported")
        if not re.fullmatch(r"[0-9a-f]{64}", self.config_hash) or not re.fullmatch(r"[0-9a-f]{64}", self.fingerprint):
            raise ValueError("hashes must be lowercase SHA-256 digests")
        if len({item.source_id for item in self.sources}) != len(self.sources):
            raise ValueError("source IDs must be unique")
        for source in self.sources:
            if source.source_id != stable_id("source", source.path):
                raise ValueError("source ID does not match path")
        source_by_id = {item.source_id: item for item in self.sources}
        documents = {item.document_id: item.restore() for item in self.documents}
        for document in documents.values():
            source = source_by_id.get(document.source_id)
            if source is None or source.source_type != document.source_type.value or source.file_type != document.file_type.value:
                raise ValueError("document does not match source")
        for snapshot in self.chunks:
            chunk = snapshot.restore()
            document = documents.get(chunk.metadata.document_id)
            if document is None:
                raise ValueError("chunk does not match document")
            if (
                chunk.metadata.source_type != document.source_type
                or chunk.metadata.file_type != document.file_type
                or chunk.metadata.parent_document_hash != document.content_hash
                or chunk.metadata.language != document.language
                or chunk.metadata.is_synthetic != document.is_synthetic
                or str(chunk.metadata.source_url) != str(document.source_url)
                or str(chunk.metadata.canonical_url) != str(document.canonical_url)
                or chunk.metadata.ingested_at != document.fetched_at
            ):
                raise ValueError("chunk does not match document")
            locators = []
            for unit in document.units:
                raw_locator = unit.get("locator")
                if isinstance(raw_locator, dict):
                    try:
                        locators.append(
                            canonical_json(
                                SourceLocator.model_validate(raw_locator).model_dump(
                                    mode="json", exclude_none=True
                                )
                            )
                        )
                    except Exception:
                        continue
            chunk_locator = canonical_json(
                chunk.metadata.source_locator.model_dump(mode="json", exclude_none=True)
            )
            if chunk_locator not in locators:
                raise ValueError("chunk locator does not match document units")
            attrs = document.attributes
            expected = {
                "entity_id": attrs.get("entity_id"),
                "company_name": attrs.get("company") or attrs.get("supplier") or attrs.get("entity"),
                "normalized_name": attrs.get("normalized_name"),
                "country_code": attrs.get("country_code"),
                "hs_code": attrs.get("hs_code"),
                "product_name": attrs.get("product_name"),
                "sku": attrs.get("sku"),
                "source_weight": attrs.get("source_weight", 0.5),
                "fact_type": attrs.get("fact_type"),
                "publish_time": attrs.get("publish_time"),
                "valid_from": attrs.get("valid_from"),
                "valid_to": attrs.get("valid_to"),
                "raw_record_id": attrs.get("raw_record_id"),
                "aggregation_info": attrs.get("aggregation_info"),
                "license_scope": attrs.get("license_scope"),
                "dedupe_cluster_id": attrs.get("dedupe_cluster_id"),
            }
            for field, expected_value in expected.items():
                actual_value = getattr(chunk.metadata, field)
                if hasattr(actual_value, "value"):
                    actual_value = actual_value.value
                elif hasattr(actual_value, "isoformat"):
                    actual_value = actual_value.isoformat()
                if expected_value != actual_value:
                    raise ValueError(f"chunk metadata {field} does not match document")
        if len(set(self.document_ids)) != len(self.documents) or len(set(self.chunk_ids)) != len(self.chunks):
            raise ValueError("document and chunk IDs must be unique")
        if len({item.name for item in self.counts_by_source_type}) != len(self.counts_by_source_type) or len({item.name for item in self.counts_by_file_type}) != len(self.counts_by_file_type):
            raise ValueError("count names must be unique")
        if tuple(sorted(item.name for item in self.counts_by_source_type)) != tuple(item.name for item in self.counts_by_source_type) or tuple(sorted(item.name for item in self.counts_by_file_type)) != tuple(item.name for item in self.counts_by_file_type):
            raise ValueError("count names must be sorted")
        if self.source_type_counts != Counter(item.source_type or "unknown" for item in self.sources):
            raise ValueError("source type counts do not match sources")
        if self.file_type_counts != Counter(item.file_type for item in self.sources):
            raise ValueError("file type counts do not match sources")
        complete = bool(self.chunks) and all(item.restore().metadata.source_locator.raw is not None for item in self.chunks)
        if self.metadata_complete != complete:
            raise ValueError("metadata_complete does not match chunks")
        expected = canonical_hash(self._fingerprint_value())
        if self.fingerprint != expected:
            raise ValueError("fingerprint does not match manifest payload")
        expected_id = "build_" + expected[:32]
        if self.build_id != expected_id:
            raise ValueError("build_id does not match manifest fingerprint")
        return self
