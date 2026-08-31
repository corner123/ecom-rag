"""Canonical, immutable ingestion build manifests."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator

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

    @field_validator("source_id", "path", "file_type", "status")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class CountEntry(_FrozenModel):
    name: StrictStr
    count: int = Field(ge=0)


class ParserBackends(_FrozenModel):
    document_router_version: StrictStr
    chunk_router_version: StrictStr
    max_tokens: int = Field(gt=0)
    overlap_tokens: int = Field(ge=0)
    mineru_statuses: tuple[StrictStr, ...]
    degraded_components: tuple[StrictStr, ...]


class DocumentSnapshot(_FrozenModel):
    """An immutable canonical payload that can restore the full router record."""
    document_id: StrictStr
    payload: StrictStr

    @classmethod
    def freeze(cls, record: DocumentRecord) -> "DocumentSnapshot":
        return cls(document_id=record.document_id, payload=canonical_json(record.model_dump(mode="json")))

    def restore(self) -> DocumentRecord:
        return DocumentRecord.model_validate_json(self.payload)


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
