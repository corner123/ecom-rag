"""Versioned source adapters for the standalone ingestion pipeline."""

from .base import CollectedSource, SourceAdapter
from .catalog import CatalogEntry, SourceCatalog
from .document_loader import (
    BackendUnavailable,
    DocumentLoaderError,
    DocumentLoaderRouter,
    EmptyDocumentError,
    MinerUCliBackend,
    MinerUOutputAdapter,
    ParsedDocumentUnit,
    RecoverableParseError,
    UnsupportedDocumentType,
    is_safe_source_path,
)
from .git_repository import GitRepositoryError, GitRepositorySource
from .local_directory import LocalDirectoryError, LocalDirectorySource
from .official_web import FetchResponse, OfficialWebSource, clean_html_document
from .schema import (
    DOCUMENT_METADATA_FIELDS,
    DOCUMENT_METADATA_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ChunkRecord,
    DocumentRecord,
    SourceRecord,
    canonical_hash,
    content_hash,
)

__all__ = [
    "SCHEMA_VERSION",
    "CatalogEntry",
    "ChunkRecord",
    "CollectedSource",
    "DocumentRecord",
    "DocumentLoaderError",
    "DocumentLoaderRouter",
    "DOCUMENT_METADATA_FIELDS",
    "DOCUMENT_METADATA_SCHEMA_VERSION",
    "BackendUnavailable",
    "EmptyDocumentError",
    "FetchResponse",
    "GitRepositoryError",
    "GitRepositorySource",
    "LocalDirectoryError",
    "LocalDirectorySource",
    "MinerUCliBackend",
    "MinerUOutputAdapter",
    "OfficialWebSource",
    "ParsedDocumentUnit",
    "RecoverableParseError",
    "SourceAdapter",
    "SourceCatalog",
    "SourceRecord",
    "UnsupportedDocumentType",
    "canonical_hash",
    "clean_html_document",
    "content_hash",
    "is_safe_source_path",
]
