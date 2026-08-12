"""Shared, JSON-serializable records for source ingestion.

The source adapters create :class:`DocumentRecord` objects without IDs.  The
ingestion pipeline assigns stable IDs and produces :class:`ChunkRecord`
objects.  Keeping these records independent from LangChain lets ingestion run
without initializing the existing RAG engine or an embedding model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "1.1"
DOCUMENT_METADATA_SCHEMA_VERSION = "1.1"
DOCUMENT_METADATA_FIELDS = frozenset(
    {
        "identity",
        "file_type",
        "content_format",
        "parser_backend",
        "parser_version",
        "page_number",
        "sheet_name",
        "table_id",
        "block_id",
        "parse_warnings",
        "parse_degraded",
        # Compatibility aliases used by older consumers.
        "page",
        "sheet",
        "table",
        "warnings",
        "degraded",
    }
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp suitable for manifests."""

    return datetime.now(timezone.utc).isoformat()


def content_hash(content: str | bytes) -> str:
    """Return a stable SHA-256 hash for text or bytes."""

    data = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash a JSON-compatible value using deterministic serialization."""

    payload = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return content_hash(payload)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


@dataclass(slots=True)
class SourceRecord:
    """Provenance for one logical source at a particular revision."""

    source_id: str
    source_type: str
    uri: str
    version: str = "unversioned"
    license: str = "unknown"
    fetched_at: str = field(default_factory=utc_now)
    content_hash: str = ""
    commit_sha: str | None = None
    dirty: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if not self.source_type.strip():
            raise ValueError("source_type must not be empty")
        if not self.uri.strip():
            raise ValueError("uri must not be empty")
        self.metadata = _json_safe(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceRecord":
        return cls(**dict(value))


@dataclass(slots=True)
class DocumentRecord:
    """A normalized source document before or after stable ID assignment.

    File adapters put parser provenance in ``metadata``.  The stable contract
    uses ``file_type``, ``content_format``, ``parser_backend``,
    ``parser_version``, ``parse_warnings`` and ``parse_degraded`` for every
    routed file; ``page_number``, ``sheet_name``, ``table_id`` and ``block_id``
    are populated when the source format provides those coordinates.
    """

    source_id: str
    relative_path: str
    content: str
    doc_id: str = ""
    content_hash: str = ""
    media_type: str = "text/plain"
    title: str = ""
    language: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("document source_id must not be empty")
        if not self.relative_path.strip():
            raise ValueError("document relative_path must not be empty")
        if not self.content_hash:
            self.content_hash = content_hash(self.content)
        self.relative_path = self.relative_path.replace("\\", "/")
        metadata = dict(_json_safe(self.metadata))
        page_number = metadata.get("page_number", metadata.get("page"))
        sheet_name = metadata.get("sheet_name", metadata.get("sheet"))
        table_id = metadata.get("table_id", metadata.get("table"))
        warnings = metadata.get("parse_warnings", metadata.get("warnings", []))
        if isinstance(warnings, str):
            warnings = [warnings] if warnings else []
        elif not isinstance(warnings, list):
            warnings = [str(warnings)] if warnings is not None else []
        degraded = bool(
            metadata.get("parse_degraded", metadata.get("degraded", False))
        )
        metadata.update(
            {
                "document_metadata_schema_version": DOCUMENT_METADATA_SCHEMA_VERSION,
                "file_type": metadata.get("file_type")
                or _infer_file_type(self.relative_path, self.media_type, self.language),
                "content_format": metadata.get("content_format")
                or _infer_content_format(self.media_type),
                "parser_backend": metadata.get("parser_backend") or "source-adapter",
                "parser_version": metadata.get("parser_version") or "unknown",
                "page_number": page_number,
                "sheet_name": sheet_name,
                "table_id": table_id,
                "block_id": metadata.get("block_id"),
                "parse_warnings": warnings,
                "parse_degraded": degraded,
                # Compatibility aliases retained for current consumers.
                "page": page_number,
                "sheet": sheet_name,
                "table": table_id,
                "warnings": warnings,
                "degraded": degraded,
            }
        )
        self.metadata = _json_safe(metadata)

    @property
    def identity(self) -> str:
        return str(self.metadata.get("identity") or self.relative_path)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DocumentRecord":
        return cls(**dict(value))


@dataclass(slots=True)
class ChunkRecord:
    """A deterministic chunk derived from a document."""

    source_id: str
    doc_id: str
    ordinal: int
    content: str
    chunk_id: str = ""
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("chunk source_id must not be empty")
        if not self.doc_id.strip():
            raise ValueError("chunk doc_id must not be empty")
        if self.ordinal < 0:
            raise ValueError("chunk ordinal must be non-negative")
        if not self.content_hash:
            self.content_hash = content_hash(self.content)
        self.metadata = _json_safe(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChunkRecord":
        return cls(**dict(value))


def _infer_file_type(relative_path: str, media_type: str, language: str) -> str:
    lowered_media = media_type.lower()
    suffix = Path(relative_path.split("#", 1)[0]).suffix.lower()
    if lowered_media == "application/pdf" or suffix == ".pdf":
        return "pdf"
    if lowered_media == "text/csv" or suffix == ".csv":
        return "csv"
    if "spreadsheetml" in lowered_media or suffix == ".xlsx":
        return "xlsx"
    if "markdown" in lowered_media or suffix in {".md", ".markdown"}:
        return "markdown"
    if lowered_media == "application/json" or suffix == ".json":
        return "json"
    if language.lower() in {
        "python", "javascript", "typescript", "java", "go", "rust", "c",
        "cpp", "shell", "powershell", "sql", "dockerfile", "makefile",
    }:
        return "code"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix == ".toml":
        return "toml"
    return "text"


def _infer_content_format(media_type: str) -> str:
    lowered = media_type.lower()
    if "markdown" in lowered:
        return "markdown"
    if lowered == "application/json":
        return "json"
    if lowered == "application/pdf":
        return "pdf-text"
    return "text"
