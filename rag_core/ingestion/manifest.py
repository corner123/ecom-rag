"""Build manifest and incremental-diff records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from rag_core.sources.schema import ChunkRecord, DocumentRecord, SCHEMA_VERSION, SourceRecord


@dataclass(slots=True)
class RecordChanges:
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.added = sorted(set(self.added))
        self.modified = sorted(set(self.modified))
        self.deleted = sorted(set(self.deleted))

    @property
    def changed(self) -> bool:
        return bool(self.added or self.modified or self.deleted)

    def to_dict(self) -> dict[str, list[str]]:
        return asdict(self)


@dataclass(slots=True)
class ManifestDiff:
    documents: RecordChanges = field(default_factory=RecordChanges)
    chunks: RecordChanges = field(default_factory=RecordChanges)

    @property
    def changed(self) -> bool:
        return self.documents.changed or self.chunks.changed

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents.to_dict(),
            "chunks": self.chunks.to_dict(),
        }


@dataclass(slots=True)
class BuildManifest:
    build_id: str
    created_at: str
    sources: list[SourceRecord]
    documents: list[DocumentRecord]
    chunks: list[ChunkRecord]
    schema_version: str = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_unique("source_id", [record.source_id for record in self.sources])
        _ensure_unique("doc_id", [record.doc_id for record in self.documents])
        _ensure_unique("chunk_id", [record.chunk_id for record in self.chunks])
        if any(not record.doc_id for record in self.documents):
            raise ValueError("all manifest documents require doc_id")
        if any(not record.chunk_id for record in self.chunks):
            raise ValueError("all manifest chunks require chunk_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "build_id": self.build_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "sources": [record.to_dict() for record in self.sources],
            "documents": [record.to_dict() for record in self.documents],
            "chunks": [record.to_dict() for record in self.chunks],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BuildManifest":
        raw_schema_version = value.get("schema_version")
        if raw_schema_version is None:
            raise ValueError(
                "build manifest has no schema_version; rebuild it with the current ingestion pipeline"
            )
        schema_version = str(raw_schema_version)
        _validate_read_schema_version(schema_version)
        metadata = dict(value.get("metadata") or {})
        pipeline_version = metadata.get("pipeline_version")
        if pipeline_version is not None and str(pipeline_version) != SCHEMA_VERSION:
            raise ValueError(
                "build manifest pipeline_version is incompatible with its "
                f"{SCHEMA_VERSION} schema; rebuild it"
            )
        return cls(
            schema_version=schema_version,
            build_id=str(value["build_id"]),
            created_at=str(value["created_at"]),
            metadata=metadata,
            sources=[SourceRecord.from_dict(item) for item in value.get("sources", [])],
            documents=[DocumentRecord.from_dict(item) for item in value.get("documents", [])],
            chunks=[ChunkRecord.from_dict(item) for item in value.get("chunks", [])],
        )

    @classmethod
    def read(cls, path: str | Path) -> "BuildManifest":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("build manifest root must be an object")
        return cls.from_dict(value)

    def write(self, path: str | Path) -> Path:
        """Atomically write a complete build manifest."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
        return output


def _ensure_unique(label: str, values: list[str]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"manifest contains duplicate {label} values")


def _validate_read_schema_version(value: str) -> None:
    def parse(version: str) -> tuple[int, ...]:
        parts = version.split(".")
        if not parts or any(not part.isdigit() for part in parts):
            raise ValueError(f"invalid build manifest schema_version: {version!r}")
        return tuple(int(part) for part in parts)

    incoming = parse(value)
    current = parse(SCHEMA_VERSION)
    if incoming > current:
        raise ValueError(
            f"build manifest schema {value} is newer than supported {SCHEMA_VERSION}; upgrade this application"
        )
    if incoming < current:
        raise ValueError(
            f"build manifest schema {value} is older than supported {SCHEMA_VERSION}; rebuild it instead of implicitly upgrading metadata"
        )
