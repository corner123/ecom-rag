"""Read-only source adapter for a directory of heterogeneous documents."""

from __future__ import annotations

import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .base import CollectedSource
from .document_loader import (
    DocumentLoaderRouter,
    UnsupportedDocumentType,
    is_safe_source_path,
)
from .schema import SourceRecord, canonical_hash, utc_now


class LocalDirectoryError(RuntimeError):
    pass


class LocalDirectorySource:
    """Collect supported documents without copying or modifying the directory."""

    def __init__(
        self,
        source_id: str,
        directory_path: str | Path,
        *,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        file_types: Mapping[str, str] | None = None,
        mime_types: Mapping[str, str] | None = None,
        max_file_size_bytes: int = 20_000_000,
        version: str | None = None,
        license: str = "unknown",
        metadata: dict[str, Any] | None = None,
        loader: DocumentLoaderRouter | None = None,
    ) -> None:
        self.source_id = source_id
        self.directory_path = Path(directory_path).expanduser().resolve()
        self.include = tuple(include or ("*", "**/*"))
        self.exclude = tuple(exclude or ())
        self.file_types = dict(file_types or {})
        self.mime_types = dict(mime_types or {})
        if max_file_size_bytes <= 0:
            raise ValueError("max_file_size_bytes must be positive")
        self.max_file_size_bytes = int(max_file_size_bytes)
        self.version = version
        self.license = license
        self.metadata = dict(metadata or {})
        self.loader = loader or DocumentLoaderRouter()

    def collect(self) -> CollectedSource:
        if not self.directory_path.is_dir():
            raise LocalDirectoryError(
                f"directory does not exist: {self.directory_path}"
            )
        documents = []
        skipped_large: list[str] = []
        for relative_path in self._candidate_files():
            if not self._selected(relative_path) or not is_safe_source_path(relative_path):
                continue
            absolute_path = self.directory_path / Path(relative_path)
            try:
                resolved_path = absolute_path.resolve()
                resolved_path.relative_to(self.directory_path)
                if absolute_path.is_symlink() or not resolved_path.is_file():
                    continue
                size = resolved_path.stat().st_size
            except (OSError, ValueError):
                continue
            explicit_type = _matching_override(relative_path, self.file_types)
            explicit_mime = _matching_override(relative_path, self.mime_types)
            try:
                self.loader.resolve_file_type(
                    resolved_path,
                    file_type=explicit_type,
                    mime_type=explicit_mime,
                )
            except UnsupportedDocumentType:
                continue
            if size > self.max_file_size_bytes:
                skipped_large.append(relative_path)
                continue
            records = self.loader.load(
                resolved_path,
                source_id=self.source_id,
                relative_path=relative_path,
                file_type=explicit_type,
                mime_type=explicit_mime,
                base_metadata={
                    "record_kind": "file",
                    "directory_name": self.directory_path.name,
                    "file_size": size,
                },
            )
            documents.extend(records)

        documents.sort(key=lambda item: (item.relative_path, item.identity))
        aggregate = canonical_hash(
            [(document.identity, document.content_hash) for document in documents]
        )
        source_version = self.version or f"snapshot-{aggregate[:16]}"
        record = SourceRecord(
            source_id=self.source_id,
            source_type="local_directory",
            uri=f"local-directory:{self.directory_path.name}",
            version=source_version,
            license=self.license,
            fetched_at=utc_now(),
            content_hash=aggregate,
            metadata={
                **self.metadata,
                "directory_name": self.directory_path.name,
                "include": list(self.include),
                "exclude": list(self.exclude),
                "file_types": self.file_types,
                "mime_types": self.mime_types,
                "max_file_size_bytes": self.max_file_size_bytes,
                "skipped_large_files": skipped_large,
                "document_count": len(documents),
            },
        )
        return CollectedSource(record=record, documents=documents)

    def _candidate_files(self) -> list[str]:
        values: list[str] = []
        for path in self.directory_path.rglob("*"):
            if not path.is_file():
                continue
            try:
                values.append(path.relative_to(self.directory_path).as_posix())
            except ValueError:
                continue
        return sorted(set(values))

    def _selected(self, relative_path: str) -> bool:
        if self.include and not _matches_any(relative_path, self.include):
            return False
        return not _matches_any(relative_path, self.exclude)


def _matches_any(relative_path: str, patterns: Iterable[str]) -> bool:
    path = relative_path.replace("\\", "/")
    pure_path = PurePosixPath(path)
    return any(
        fnmatch.fnmatchcase(path, pattern.replace("\\", "/"))
        or pure_path.match(pattern.replace("\\", "/"))
        for pattern in patterns
    )


def _matching_override(relative_path: str, values: Mapping[str, str]) -> str | None:
    for pattern, value in values.items():
        if _matches_any(relative_path, (pattern,)):
            return value
    return None
