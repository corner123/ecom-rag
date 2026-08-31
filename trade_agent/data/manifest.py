"""Canonical, immutable ingestion build manifests."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from collections import Counter
import re
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, field_validator, model_validator

from trade_agent.data.quarantine import QuarantineRecord
from trade_agent.schemas.source import ChunkRecord, DocumentRecord

METADATA_SCHEMA_VERSION = "task6-source-metadata-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


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

    @field_validator("source_id", "path", "file_type", "status", "parser_backend")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
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
        if self.path.startswith("/") or ".." in PurePosixPath(self.path).parts:
            raise ValueError("source path must be safe and relative")
        if self.status == "parsed" and (self.actual_content_hash != self.manifest_content_hash or not self.parser_backend):
            raise ValueError("parsed source requires matching hashes and parser backend")
        return self


class CountEntry(_FrozenModel):
    name: StrictStr
    count: StrictInt = Field(ge=0)


class ParserBackends(_FrozenModel):
    document_router_version: StrictStr
    chunk_router_version: StrictStr
    max_tokens: StrictInt = Field(gt=0)
    overlap_tokens: StrictInt = Field(ge=0)
    mineru_statuses: tuple[StrictStr, ...]
    degraded_components: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def ordered_values(self) -> "ParserBackends":
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
        if self.restore().document_id != self.document_id:
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
        if len(set(self.document_ids)) != len(self.documents) or len(set(self.chunk_ids)) != len(self.chunks):
            raise ValueError("document and chunk IDs must be unique")
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
