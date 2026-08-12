"""Deterministic source-to-manifest ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

from rag_core.sources.base import CollectedSource, SourceAdapter
from rag_core.sources.schema import (
    ChunkRecord,
    DocumentRecord,
    SourceRecord,
    canonical_hash,
    content_hash,
    utc_now,
)

from .manifest import BuildManifest, ManifestDiff, RecordChanges


PIPELINE_VERSION = "1.1"
CHUNKER_VERSION = "character-boundary-v1"
_VOLATILE_METADATA_KEYS = {
    "fetched_at",
    "etag",
    "last_modified",
    "response_headers",
    "commit_sha",
    "dirty",
}
T = TypeVar("T")


@dataclass(slots=True)
class IngestionResult:
    manifest: BuildManifest
    diff: ManifestDiff
    output_path: Path | None = None


class IngestionPipeline:
    """Collect sources, assign stable IDs, chunk, manifest, and diff.

    IDs intentionally exclude content hashes: a changed file or chunk at the
    same logical position retains its identity and appears in ``modified``.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 1_200,
        chunk_overlap: int = 120,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.clock = clock

    def build(
        self,
        sources: Iterable[SourceAdapter | CollectedSource],
        *,
        previous: BuildManifest | str | Path | None = None,
        output_path: str | Path | None = None,
    ) -> IngestionResult:
        collected = [
            item if isinstance(item, CollectedSource) else item.collect()
            for item in sources
        ]
        source_records, documents = self._normalize_collections(collected)
        chunks = [chunk for document in documents for chunk in self._chunk_document(document)]

        build_fingerprint = canonical_hash(
            {
                "pipeline_version": PIPELINE_VERSION,
                "chunker_version": CHUNKER_VERSION,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "sources": [_source_revision(record) for record in source_records],
                "documents": [
                    (record.doc_id, _document_revision(record)) for record in documents
                ],
            }
        )
        manifest = BuildManifest(
            build_id=f"build_{build_fingerprint[:20]}",
            created_at=self.clock(),
            sources=source_records,
            documents=documents,
            chunks=chunks,
            metadata={
                "pipeline_version": PIPELINE_VERSION,
                "chunker_version": CHUNKER_VERSION,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
        )
        previous_manifest = _load_previous(previous)
        diff = self.diff(manifest, previous_manifest)
        written_path = manifest.write(output_path) if output_path is not None else None
        return IngestionResult(manifest=manifest, diff=diff, output_path=written_path)

    @staticmethod
    def diff(current: BuildManifest, previous: BuildManifest | None) -> ManifestDiff:
        if previous is None:
            return ManifestDiff(
                documents=RecordChanges(added=[record.doc_id for record in current.documents]),
                chunks=RecordChanges(added=[record.chunk_id for record in current.chunks]),
            )
        return ManifestDiff(
            documents=_diff_records(
                current.documents,
                previous.documents,
                key=lambda record: record.doc_id,
                revision=_document_revision,
            ),
            chunks=_diff_records(
                current.chunks,
                previous.chunks,
                key=lambda record: record.chunk_id,
                revision=_chunk_revision,
            ),
        )

    def _normalize_collections(
        self, collected: Sequence[CollectedSource]
    ) -> tuple[list[SourceRecord], list[DocumentRecord]]:
        source_records: list[SourceRecord] = []
        documents: list[DocumentRecord] = []
        source_ids: set[str] = set()
        document_ids: set[str] = set()

        for snapshot in collected:
            record = snapshot.record
            if record.source_id in source_ids:
                raise ValueError(f"duplicate collected source_id: {record.source_id}")
            source_ids.add(record.source_id)

            normalized_for_source: list[DocumentRecord] = []
            for document in snapshot.documents:
                if document.source_id != record.source_id:
                    raise ValueError(
                        f"document source_id '{document.source_id}' does not match '{record.source_id}'"
                    )
                stable_id = _stable_id("doc", document.source_id, _normalize_identity(document.identity))
                if stable_id in document_ids:
                    raise ValueError(
                        f"duplicate document identity for source '{document.source_id}': {document.identity}"
                    )
                document_ids.add(stable_id)
                normalized = replace(
                    document,
                    doc_id=stable_id,
                    content_hash=content_hash(document.content),
                    metadata={
                        **record.metadata,
                        "source_type": record.source_type,
                        "source_version": record.version,
                        "source_license": record.license,
                        "source_commit_sha": record.commit_sha,
                        **document.metadata,
                    },
                )
                normalized_for_source.append(normalized)
                documents.append(normalized)

            normalized_for_source.sort(key=lambda item: (item.relative_path, item.identity))
            aggregate = canonical_hash(
                [(item.identity, item.content_hash) for item in normalized_for_source]
            )
            source_records.append(
                replace(
                    record,
                    content_hash=record.content_hash or aggregate,
                    metadata={
                        **record.metadata,
                        "document_count": len(normalized_for_source),
                    },
                )
            )

        source_records.sort(key=lambda item: item.source_id)
        documents.sort(key=lambda item: (item.source_id, item.relative_path, item.identity))
        return source_records, documents

    def _chunk_document(self, document: DocumentRecord) -> list[ChunkRecord]:
        if not document.content.strip():
            return []
        chunks = []
        for ordinal, (start, end) in enumerate(self._chunk_spans(document.content)):
            value = document.content[start:end]
            chunk_id = _stable_id("chunk", document.doc_id, str(ordinal), CHUNKER_VERSION)
            line_start = document.content.count("\n", 0, start) + 1
            line_end = document.content.count("\n", 0, end) + 1
            chunk_metadata = {
                **document.metadata,
                "document_path": document.relative_path,
                "document_title": document.title,
                "media_type": document.media_type,
                "language": document.language,
                "char_start": start,
                "char_end": end,
                "chunk_line_start": line_start,
                "chunk_line_end": line_end,
            }
            # Raw file documents use chunk-relative line spans. Symbol cards
            # already carry authoritative source-file spans, which must remain
            # intact for citations.
            chunk_metadata.setdefault("line_start", line_start)
            chunk_metadata.setdefault("line_end", line_end)
            chunks.append(
                ChunkRecord(
                    source_id=document.source_id,
                    doc_id=document.doc_id,
                    ordinal=ordinal,
                    content=value,
                    chunk_id=chunk_id,
                    metadata=chunk_metadata,
                )
            )
        return chunks

    def _chunk_spans(self, value: str) -> list[tuple[int, int]]:
        if len(value) <= self.chunk_size:
            return [(0, len(value))]
        spans: list[tuple[int, int]] = []
        start = 0
        while start < len(value):
            hard_end = min(len(value), start + self.chunk_size)
            end = hard_end
            if hard_end < len(value):
                minimum = start + max(1, self.chunk_size // 2)
                for separator in ("\n\n", "\n", " "):
                    candidate = value.rfind(separator, minimum, hard_end)
                    if candidate >= minimum:
                        end = candidate + len(separator)
                        break
            if end <= start:
                end = hard_end
            spans.append((start, end))
            if end >= len(value):
                break
            next_start = max(0, end - self.chunk_overlap)
            start = next_start if next_start > start else end
        return spans


def _stable_id(prefix: str, *parts: str) -> str:
    digest = content_hash("\0".join(parts))[:24]
    return f"{prefix}_{digest}"


def _normalize_identity(value: str) -> str:
    return value.strip().replace("\\", "/")


def _load_previous(value: BuildManifest | str | Path | None) -> BuildManifest | None:
    if value is None or isinstance(value, BuildManifest):
        return value
    return BuildManifest.read(value)


def _source_revision(record: SourceRecord) -> str:
    value = record.to_dict()
    value.pop("fetched_at", None)
    value["metadata"] = _stable_metadata(value.get("metadata", {}))
    return canonical_hash(value)


def _document_revision(record: DocumentRecord) -> str:
    return canonical_hash(
        {
            "content_hash": record.content_hash,
            "relative_path": record.relative_path,
            "media_type": record.media_type,
            "title": record.title,
            "language": record.language,
            "metadata": _stable_metadata(record.metadata),
        }
    )


def _chunk_revision(record: ChunkRecord) -> str:
    return canonical_hash(
        {
            "content_hash": record.content_hash,
            "ordinal": record.ordinal,
            "metadata": _stable_metadata(record.metadata),
        }
    )


def _stable_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _stable_metadata(item)
            for key, item in value.items()
            if str(key) not in _VOLATILE_METADATA_KEYS
        }
    if isinstance(value, list):
        return [_stable_metadata(item) for item in value]
    return value


def _diff_records(
    current: Sequence[T],
    previous: Sequence[T],
    *,
    key: Callable[[T], str],
    revision: Callable[[T], str],
) -> RecordChanges:
    current_by_id = {key(record): record for record in current}
    previous_by_id = {key(record): record for record in previous}
    current_ids = set(current_by_id)
    previous_ids = set(previous_by_id)
    common = current_ids & previous_ids
    return RecordChanges(
        added=list(current_ids - previous_ids),
        deleted=list(previous_ids - current_ids),
        modified=[
            record_id
            for record_id in common
            if revision(current_by_id[record_id]) != revision(previous_by_id[record_id])
        ],
    )
