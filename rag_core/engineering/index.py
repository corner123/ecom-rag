"""Partitioned dense + BM25 indexes for engineering knowledge."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from typing import Any, Iterable, Mapping
import uuid

from langchain_core.documents import Document

from config import EmbeddingConfig
from rag_core.index.dense_store import DenseVectorStore
from rag_core.index.embeddings import EmbeddingManager
from rag_core.index.engineering_milvus_store import (
    AvailabilityFailoverDenseStore,
    EngineeringMilvusVectorStore,
    MilvusAvailabilityError,
    MilvusBackendError,
    MilvusClientFactory,
    MilvusConnectionSettings,
    build_collection_name,
    resolve_milvus_owner_namespace,
)
from rag_core.index.local_vector_store import LocalVectorStore
from rag_core.ingestion import BuildManifest
from rag_core.retrieval.engineering import (
    BM25Retriever,
    EngineeringSearchResult,
    FederatedRetriever,
    rrf_fusion,
)
from rag_core.sources.schema import (
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    ChunkRecord,
    DocumentRecord,
    SourceRecord,
)


INDEX_SCHEMA_VERSION = 4
LEGACY_INDEX_SCHEMA_VERSIONS = frozenset({3})
BUILD_BACKENDS = frozenset({"faiss", "milvus", "both"})
RUNTIME_BACKENDS = frozenset({"faiss", "milvus", "auto"})
_HTTP_RE = re.compile(r"^https?://", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class PartitionSpec:
    corpus: str
    authority: str
    directory: str
    document_count: int
    weight: float = 1.0
    file_sha256: dict[str, str] = field(default_factory=dict)
    milvus_collection_name: str | None = None
    milvus_dimension: int | None = None
    milvus_metric_type: str | None = None
    milvus_owner_namespace: str | None = None


@dataclass(frozen=True, slots=True)
class _ValidatedCatalog:
    """Pure, filesystem-independent interpretation of an index catalog."""

    schema_version: int
    specs: tuple[PartitionSpec, ...]
    build_id: str
    embedding_model: str
    requested_build_backend: str
    available_backends: tuple[str, ...]
    embedding_dimension: int | None


def infer_partition(
    chunk: ChunkRecord,
    document: DocumentRecord,
    source: SourceRecord,
) -> tuple[str, str]:
    """Infer ``(corpus, authority)`` while honoring explicit source metadata."""

    metadata = {**source.metadata, **document.metadata, **chunk.metadata}
    source_type = str(metadata.get("source_type") or source.source_type).casefold()
    corpus = str(metadata.get("corpus") or "").strip().casefold()
    if not corpus:
        corpus = "official" if source_type == "official_web" else "internal"

    explicit_authority = str(metadata.get("authority") or "").strip().casefold()
    if corpus == "official" or source_type == "official_web":
        return "official", "official"
    if explicit_authority in {"code", "test", "design", "history"}:
        return corpus, explicit_authority

    path = document.relative_path.split("#", 1)[0].replace("\\", "/")
    if str(metadata.get("record_kind") or "").casefold() == "git_commit":
        return corpus, "history"
    normalized = f"/{path.casefold().lstrip('/')}"
    name = PurePosixPath(path).name.casefold()
    suffix = PurePosixPath(path).suffix.casefold()
    if "/tests/" in normalized or name.startswith("test_") or name.endswith("_test.py"):
        authority = "test"
    elif suffix in {".py", ".pyi", ".js", ".ts", ".tsx", ".go", ".rs", ".java"} or "#symbol:" in document.relative_path:
        authority = "code"
    else:
        authority = "design"
    return corpus, authority


def _partition_weight(corpus: str, authority: str) -> float:
    return {
        ("internal", "code"): 1.35,
        ("internal", "test"): 1.20,
        ("internal", "design"): 1.00,
        ("internal", "history"): 0.80,
        ("official", "official"): 0.90,
    }.get((corpus, authority), 1.0)


def _safe_partition_name(corpus: str, authority: str) -> str:
    value = f"{corpus}__{authority}"
    if not re.fullmatch(r"[a-z0-9_-]+", value):
        raise ValueError(f"unsafe partition name: {value}")
    return value


def _chunk_document(
    chunk: ChunkRecord,
    document: DocumentRecord,
    source: SourceRecord,
) -> Document:
    corpus, authority = infer_partition(chunk, document, source)
    metadata: dict[str, Any] = {
        **source.metadata,
        **document.metadata,
        **chunk.metadata,
        "chunk_id": chunk.chunk_id,
        "doc_id": document.doc_id,
        "source_id": source.source_id,
        "source_type": source.source_type,
        "source_uri": source.uri,
        "source_version": source.version,
        "source_license": source.license,
        "source_content_hash": source.content_hash,
        "source_fetched_at": source.fetched_at,
        "source_dirty": source.dirty,
        "document_content_hash": document.content_hash,
        "source_commit_sha": source.commit_sha or document.metadata.get("source_commit_sha"),
        "document_path": document.relative_path,
        "document_title": document.title,
        "corpus": corpus,
        "authority": authority,
    }
    citation_source = document.relative_path
    if corpus == "official":
        citation_source = document.relative_path if _HTTP_RE.match(document.relative_path) else source.uri
    metadata["citation_source"] = citation_source
    return Document(page_content=chunk.content, metadata=metadata)


def _as_search_result(document: Document) -> EngineeringSearchResult:
    metadata = dict(document.metadata)
    source = str(
        metadata.get("citation_source")
        or metadata.get("document_path")
        or metadata.get("source_uri")
        or "unknown-source"
    )
    return EngineeringSearchResult(
        content=document.page_content,
        source=source,
        corpus=str(metadata.get("corpus") or "internal"),
        authority=str(metadata.get("authority") or "unknown"),
        line_start=_optional_int(metadata.get("line_start") or metadata.get("chunk_line_start")),
        line_end=_optional_int(metadata.get("line_end") or metadata.get("chunk_line_end")),
        symbol=_optional_text(
            metadata.get("symbol_qualified_name")
            or metadata.get("symbol")
            or metadata.get("symbol_name")
        ),
        metadata=metadata,
    )


class DenseVectorRetriever:
    """Adapter from a local or distributed dense store to search results."""

    def __init__(self, store: DenseVectorStore) -> None:
        self.store = store

    def search(self, query: str, top_k: int = 5) -> list[EngineeringSearchResult]:
        results: list[EngineeringSearchResult] = []
        for raw in self.store.search_dense(query, top_k=top_k):
            metadata = dict(raw.get("metadata") or {})
            source = str(
                metadata.get("citation_source")
                or metadata.get("document_path")
                or metadata.get("source_uri")
                or "unknown-source"
            )
            results.append(
                EngineeringSearchResult(
                    content=str(raw.get("text") or ""),
                    source=source,
                    score=float(raw.get("score") or 0.0),
                    corpus=str(metadata.get("corpus") or "internal"),
                    authority=str(metadata.get("authority") or "unknown"),
                    line_start=_optional_int(metadata.get("line_start") or metadata.get("chunk_line_start")),
                    line_end=_optional_int(metadata.get("line_end") or metadata.get("chunk_line_end")),
                    symbol=_optional_text(
                        metadata.get("symbol_qualified_name")
                        or metadata.get("symbol")
                        or metadata.get("symbol_name")
                    ),
                    retriever="dense",
                    metadata=metadata,
                )
            )
        return results


class HybridPartitionRetriever:
    """Fuse dense semantic retrieval and exact-token BM25 inside one partition."""

    def __init__(
        self,
        dense: DenseVectorRetriever,
        bm25: BM25Retriever,
        *,
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        fail_open: bool = True,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.candidate_multiplier = max(1, candidate_multiplier)
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.fail_open = bool(fail_open)

    def search(self, query: str, top_k: int = 5) -> list[EngineeringSearchResult]:
        if top_k <= 0:
            return []
        candidate_k = max(top_k, top_k * self.candidate_multiplier)
        degraded: list[str] = []
        failures: list[Exception] = []
        try:
            dense_results = self.dense.search(query, top_k=candidate_k)
        except Exception as exc:
            dense_results = []
            degraded.append("dense")
            failures.append(exc)
        try:
            sparse_results = self.bm25.search(query, top_k=candidate_k)
        except Exception as exc:
            sparse_results = []
            degraded.append("bm25")
            failures.append(exc)
        if degraded and not self.fail_open:
            # Preserve the classified backend exception so callers and health
            # checks can distinguish availability, integrity and programming
            # failures.  Generic retrievers still receive the established
            # fail-closed boundary while retaining their cause via chaining.
            first_failure = failures[0]
            if isinstance(first_failure, MilvusBackendError):
                raise first_failure
            raise RuntimeError("hybrid retrieval failed closed") from first_failure
        fused = rrf_fusion(
            [dense_results, sparse_results],
            k=self.rrf_k,
            top_k=top_k,
            weights=[self.dense_weight, self.bm25_weight],
        )
        if not degraded:
            return fused
        return [
            result.updated(
                metadata={
                    **result.metadata,
                    "retrieval_degraded_components": list(degraded),
                }
            )
            for result in fused
        ]


class EngineeringIndex:
    """Build separated partitions with portable FAISS and optional Milvus."""

    CATALOG_FILENAME = "partitions.json"

    def __init__(
        self,
        root: str | Path,
        embedding_manager: EmbeddingManager,
        specs: Iterable[PartitionSpec],
        *,
        build_id: str = "",
        embedding_model: str = "",
        runtime_backend: str = "faiss",
        milvus_settings: MilvusConnectionSettings | None = None,
        milvus_client_factory: MilvusClientFactory | None = None,
        catalog_schema_version: int = INDEX_SCHEMA_VERSION,
        requested_build_backend: str = "faiss",
        embedding_dimension: int | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.embedding_manager = embedding_manager
        self.specs = tuple(specs)
        self.build_id = build_id
        self.embedding_model = embedding_model or _manager_model_name(embedding_manager)
        self.catalog_schema_version = int(catalog_schema_version)
        self.requested_build_backend = requested_build_backend
        self.embedding_dimension = (
            int(embedding_dimension) if embedding_dimension is not None else None
        )
        self.runtime_backend = _resolve_runtime_backend(runtime_backend)
        self.milvus_settings = milvus_settings
        self.milvus_client_factory = milvus_client_factory
        # Distributed modes must surface partition failures.  Otherwise the
        # outer federated layer could swallow a strict Milvus error even though
        # the inner hybrid retriever correctly failed closed.
        self.federated = FederatedRetriever(
            fail_open=self.runtime_backend == "faiss"
        )
        self._stores: list[DenseVectorStore] = []
        self._backend_by_partition: dict[str, str] = {}
        self._fallback_reasons: dict[str, str] = {}
        # Service health keeps a shallow copy of index stats; this nested state
        # intentionally remains shared so query-time auto failover is visible.
        self._backend_state: dict[str, Any] = {
            "configured_backend": self.runtime_backend,
            "requested_mode": self.runtime_backend,
            "primary_backend": (
                "milvus" if self.runtime_backend in {"milvus", "auto"} else "faiss"
            ),
            "active_backend": "unloaded",
            "available_backends": [
                name
                for name in ("faiss", "milvus")
                if name == "faiss"
                or any(spec.milvus_collection_name for spec in self.specs)
            ],
            "fallback_used": False,
            "degraded": False,
            "reason_code": None,
            "partitions": self._backend_by_partition,
            "fallback_reasons": self._fallback_reasons,
        }
        self._load_partitions()

    @classmethod
    def build(
        cls,
        manifest: BuildManifest,
        root: str | Path,
        *,
        embedding_manager: EmbeddingManager | None = None,
        batch_size: int = 128,
        backend: str | None = None,
        milvus_settings: MilvusConnectionSettings | None = None,
        milvus_client_factory: MilvusClientFactory | None = None,
    ) -> "EngineeringIndex":
        if str(manifest.schema_version) != SOURCE_SCHEMA_VERSION:
            raise ValueError(
                "engineering manifest schema is stale; run sources-sync before "
                f"engineering-build (expected {SOURCE_SCHEMA_VERSION}, got "
                f"{manifest.schema_version})"
            )
        manager = embedding_manager or _default_embedding_manager()
        build_backend = _resolve_build_backend(backend)
        settings = milvus_settings
        if build_backend in {"milvus", "both"} and settings is None:
            settings = MilvusConnectionSettings.from_env()
        index_root = Path(root).resolve()
        index_root.parent.mkdir(parents=True, exist_ok=True)
        lock_path = index_root.parent / f".{index_root.name}.build.lock"
        try:
            build_lock = _acquire_build_lock(lock_path)
        except FileExistsError as exc:
            raise RuntimeError(
                f"engineering index build is already running: {lock_path}"
            ) from exc
        build_token = uuid.uuid4().hex
        staging_root = index_root.parent / f".{index_root.name}.staging-{build_token}"
        backup_root = index_root.parent / f".{index_root.name}.backup-{build_token}"
        created_milvus: list[EngineeringMilvusVectorStore] = []
        published = False
        build_error: BaseException | None = None
        try:
            _recover_interrupted_publish(index_root)
            staging_root.mkdir()
            result = cls._build_and_publish(
                manifest,
                index_root,
                staging_root,
                backup_root,
                manager,
                batch_size=batch_size,
                build_backend=build_backend,
                build_token=build_token,
                milvus_settings=settings,
                milvus_client_factory=milvus_client_factory,
                created_milvus=created_milvus,
            )
            published = True
            return result
        except BaseException as exc:
            build_error = exc
            raise
        finally:
            cleanup_errors: list[Exception] = []
            try:
                if not published:
                    for store in reversed(created_milvus):
                        try:
                            store.drop_collection(ignore_missing=True)
                        except Exception as cleanup_error:
                            # An unpublished collection is unreachable from a
                            # reader catalog. Preserve the build failure and
                            # report remote garbage-collection trouble as an
                            # exception note instead of replacing the cause.
                            cleanup_errors.append(cleanup_error)
                if staging_root.exists():
                    try:
                        shutil.rmtree(staging_root)
                    except Exception as cleanup_error:
                        cleanup_errors.append(cleanup_error)
            finally:
                # A backup is deleted only by _build_and_publish after the
                # newly published index has loaded successfully. Most
                # importantly, cleanup failures can never bypass lock release.
                try:
                    _release_build_lock(build_lock)
                except Exception as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                summary = "; ".join(
                    f"{type(error).__name__}: {error}" for error in cleanup_errors
                )
                note = f"unpublished engineering-index cleanup failed: {summary}"
                active_error = build_error or sys.exc_info()[1]
                if active_error is not None:
                    active_error.add_note(note)
                else:
                    raise RuntimeError(note) from cleanup_errors[0]

    @classmethod
    def _build_and_publish(
        cls,
        manifest: BuildManifest,
        index_root: Path,
        staging_root: Path,
        backup_root: Path,
        manager: EmbeddingManager,
        *,
        batch_size: int,
        build_backend: str,
        build_token: str,
        milvus_settings: MilvusConnectionSettings | None,
        milvus_client_factory: MilvusClientFactory | None,
        created_milvus: list[EngineeringMilvusVectorStore],
    ) -> "EngineeringIndex":
        sources = {record.source_id: record for record in manifest.sources}
        documents = {record.doc_id: record for record in manifest.documents}
        if not sources or not documents or not manifest.chunks:
            raise ValueError("refusing to publish an empty engineering index")
        if len(sources) != len(manifest.sources):
            raise ValueError("manifest contains duplicate source IDs")
        if len(documents) != len(manifest.documents):
            raise ValueError("manifest contains duplicate document IDs")
        partitions: dict[tuple[str, str], list[Document]] = {}
        seen_chunk_ids: set[str] = set()
        for chunk in manifest.chunks:
            if chunk.chunk_id in seen_chunk_ids:
                raise ValueError(f"manifest contains duplicate chunk ID: {chunk.chunk_id}")
            seen_chunk_ids.add(chunk.chunk_id)
            document = documents.get(chunk.doc_id)
            source = sources.get(chunk.source_id)
            if document is None or source is None:
                raise ValueError(f"orphan chunk in manifest: {chunk.chunk_id}")
            if document.source_id != chunk.source_id:
                raise ValueError(
                    f"chunk/document source mismatch: {chunk.chunk_id}"
                )
            indexed = _chunk_document(chunk, document, source)
            key = (str(indexed.metadata["corpus"]), str(indexed.metadata["authority"]))
            partitions.setdefault(key, []).append(indexed)

        specs: list[PartitionSpec] = []
        embedding_dimensions: set[int] = set()
        milvus_owner_namespace = (
            resolve_milvus_owner_namespace(required=True)
            if build_backend in {"milvus", "both"}
            else None
        )
        for (corpus, authority), indexed_documents in sorted(partitions.items()):
            directory = _safe_partition_name(corpus, authority)
            store = LocalVectorStore(manager, persist_dir=str(staging_root / directory))
            store.replace_documents(indexed_documents, batch_size=batch_size)
            partition_root = staging_root / directory
            faiss_stats = store.get_stats()
            embedding_dimensions.add(int(faiss_stats["dimension"]))
            file_sha256 = {
                name: _sha256_file(partition_root / name)
                for name in ("index.faiss", "documents.json", "index_meta.json")
            }
            milvus_collection_name: str | None = None
            milvus_dimension: int | None = None
            milvus_metric_type: str | None = None
            if build_backend in {"milvus", "both"}:
                milvus_collection_name = build_collection_name(
                    manifest.build_id,
                    directory,
                    build_token,
                )
                milvus_dimension = int(faiss_stats["dimension"])
                remote = EngineeringMilvusVectorStore(
                    manager,
                    milvus_collection_name,
                    expected_dimension=milvus_dimension,
                    expected_count=len(indexed_documents),
                    settings=milvus_settings,
                    client_factory=milvus_client_factory,
                    build_id=manifest.build_id,
                    embedding_model=_manager_model_name(manager),
                    owner_namespace=milvus_owner_namespace,
                )
                created_milvus.append(remote)
                remote.replace_documents(indexed_documents, batch_size=batch_size)
                milvus_metric_type = "COSINE"
            specs.append(
                PartitionSpec(
                    corpus=corpus,
                    authority=authority,
                    directory=directory,
                    document_count=len(indexed_documents),
                    weight=_partition_weight(corpus, authority),
                    file_sha256=file_sha256,
                    milvus_collection_name=milvus_collection_name,
                    milvus_dimension=milvus_dimension,
                    milvus_metric_type=milvus_metric_type,
                    milvus_owner_namespace=milvus_owner_namespace,
                )
            )

        if len(embedding_dimensions) != 1:
            raise RuntimeError("engineering partitions used inconsistent embedding dimensions")
        embedding_dimension = next(iter(embedding_dimensions))
        catalog = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "build_id": manifest.build_id,
            "embedding_model": _manager_model_name(manager),
            "embedding_dimension": embedding_dimension,
            "requested_build_backend": build_backend,
            "available_backends": (
                ["faiss", "milvus"]
                if build_backend in {"milvus", "both"}
                else ["faiss"]
            ),
            "partitions": [asdict(spec) for spec in specs],
        }
        catalog_path = staging_root / cls.CATALOG_FILENAME
        temporary_catalog = staging_root / f".{cls.CATALOG_FILENAME}.tmp"
        temporary_catalog.write_text(
            json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_catalog, catalog_path)
        # Load every staged partition before changing the currently published
        # root. A missing/corrupt FAISS or document file fails here and leaves
        # the old index untouched.
        validation_backend = (
            "milvus" if build_backend in {"milvus", "both"} else "faiss"
        )
        staged = cls.load(
            staging_root,
            embedding_manager=manager,
            runtime_backend=validation_backend,
            milvus_settings=milvus_settings,
            milvus_client_factory=milvus_client_factory,
        )
        if staged.stats()["document_count"] != len(manifest.chunks):
            raise RuntimeError("staged engineering index count does not match manifest")

        if index_root.exists():
            os.replace(index_root, backup_root)
        try:
            os.replace(staging_root, index_root)
        except Exception:
            if backup_root.exists() and not index_root.exists():
                os.replace(backup_root, index_root)
            raise
        try:
            published = cls.load(
                index_root,
                embedding_manager=manager,
                runtime_backend=validation_backend,
                milvus_settings=milvus_settings,
                milvus_client_factory=milvus_client_factory,
            )
        except Exception as publish_error:
            if backup_root.exists():
                failed_root = index_root.parent / (
                    f".{index_root.name}.failed-{uuid.uuid4().hex}"
                )
                try:
                    if index_root.exists():
                        os.replace(index_root, failed_root)
                    os.replace(backup_root, index_root)
                except Exception as rollback_error:
                    raise RuntimeError(
                        "published index validation failed and rollback could not "
                        f"complete; preserved backup at {backup_root}"
                    ) from rollback_error
                finally:
                    if failed_root.exists() and index_root.exists():
                        shutil.rmtree(failed_root, ignore_errors=True)
            else:
                shutil.rmtree(index_root, ignore_errors=True)
            raise
        if backup_root.exists():
            # Cleanup is not part of the publish transaction. A transient
            # antivirus/file-handle failure must not turn a successful build
            # into a reported failure.
            shutil.rmtree(backup_root, ignore_errors=True)
        return published

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        embedding_manager: EmbeddingManager | None = None,
        runtime_backend: str | None = None,
        milvus_settings: MilvusConnectionSettings | None = None,
        milvus_client_factory: MilvusClientFactory | None = None,
    ) -> "EngineeringIndex":
        index_root = Path(root).resolve()
        catalog_path = index_root / cls.CATALOG_FILENAME
        if not catalog_path.is_file():
            raise FileNotFoundError(f"engineering index catalog not found: {catalog_path}")
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog = _validate_catalog_payload(payload)
        schema_version = catalog.schema_version
        specs = catalog.specs
        requested_build_backend = catalog.requested_build_backend
        build_id = catalog.build_id
        model_name = catalog.embedding_model
        manager = embedding_manager or _default_embedding_manager(model_name)
        if embedding_manager is not None and _manager_model_name(manager) != model_name:
            raise ValueError(
                "engineering index embedding model does not match the provided manager"
            )
        return cls(
            index_root,
            manager,
            specs,
            build_id=build_id,
            embedding_model=model_name,
            runtime_backend=_resolve_runtime_backend(runtime_backend),
            milvus_settings=milvus_settings,
            milvus_client_factory=milvus_client_factory,
            catalog_schema_version=schema_version,
            requested_build_backend=requested_build_backend,
            embedding_dimension=catalog.embedding_dimension,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "build_id": self.build_id,
            "index_root": str(self.root),
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "catalog_schema_version": self.catalog_schema_version,
            "requested_build_backend": self.requested_build_backend,
            "vector_backend": self._backend_state,
            "partitions": [asdict(spec) for spec in self.specs],
            "document_count": sum(spec.document_count for spec in self.specs),
        }

    def _load_partitions(self) -> None:
        seen_chunk_ids: set[str] = set()
        for spec in self.specs:
            partition_path = (self.root / spec.directory).resolve()
            try:
                partition_path.relative_to(self.root)
            except ValueError as exc:
                raise ValueError(f"partition escapes index root: {spec.directory}") from exc
            required_files = ("index.faiss", "documents.json", "index_meta.json")
            missing = [
                name for name in required_files if not (partition_path / name).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"engineering partition {spec.directory} is incomplete: {missing}"
                )
            if set(spec.file_sha256) != set(required_files):
                raise ValueError(
                    f"engineering partition {spec.directory} has incomplete checksums"
                )
            for name in required_files:
                actual_hash = _sha256_file(partition_path / name)
                if actual_hash != spec.file_sha256[name]:
                    raise ValueError(
                        f"engineering partition {spec.directory} checksum mismatch: {name}"
                    )
            local_store = LocalVectorStore(
                self.embedding_manager, persist_dir=str(partition_path)
            )
            local_stats = local_store.get_stats()
            if (
                self.embedding_dimension is not None
                and int(local_stats["dimension"]) != self.embedding_dimension
            ):
                raise ValueError(
                    f"engineering partition {spec.directory} does not match catalog embedding dimension"
                )
            milvus_fields = (
                spec.milvus_collection_name,
                spec.milvus_dimension,
                spec.milvus_metric_type,
                spec.milvus_owner_namespace,
            )
            if any(value is not None for value in milvus_fields):
                if any(value is None for value in milvus_fields):
                    raise ValueError(
                        f"engineering partition {spec.directory} has an incomplete Milvus artifact"
                    )
                if spec.milvus_metric_type != "COSINE":
                    raise ValueError(
                        f"engineering partition {spec.directory} has an unsupported Milvus metric"
                    )
                if int(spec.milvus_dimension or -1) != int(local_stats["dimension"]):
                    raise ValueError(
                        f"engineering partition {spec.directory} FAISS/Milvus dimension mismatch"
                    )
            documents = [
                _as_search_result(document) for document in local_store.all_documents()
            ]
            if len(documents) != spec.document_count:
                raise ValueError(
                    f"engineering partition {spec.directory} count mismatch: "
                    f"catalog={spec.document_count}, documents={len(documents)}"
                )
            for document in documents:
                if document.corpus != spec.corpus or document.authority != spec.authority:
                    raise ValueError(
                        f"engineering partition metadata mismatch: {spec.directory}"
                    )
                chunk_id = str(document.metadata.get("chunk_id") or "")
                if not chunk_id or chunk_id in seen_chunk_ids:
                    raise ValueError(
                        f"missing or duplicate indexed chunk ID in {spec.directory}: {chunk_id}"
                    )
                seen_chunk_ids.add(chunk_id)
            dense_store = self._select_dense_store(spec, local_store)
            retriever = HybridPartitionRetriever(
                DenseVectorRetriever(dense_store),
                BM25Retriever(documents),
                # A distributed or auto backend must not hide integrity,
                # authentication or programming failures by silently using
                # sparse-only results. Auto failover is implemented narrowly
                # inside AvailabilityFailoverDenseStore.
                fail_open=self.runtime_backend == "faiss",
            )
            self.federated.add_partition(
                spec.corpus,
                spec.authority,
                retriever,
                weight=spec.weight,
            )
            self._stores.append(dense_store)
        self._refresh_active_backend()

    def _select_dense_store(
        self,
        spec: PartitionSpec,
        local_store: LocalVectorStore,
    ) -> DenseVectorStore:
        if self.runtime_backend == "faiss":
            self._backend_by_partition[spec.directory] = "faiss"
            return local_store

        collection_name = spec.milvus_collection_name
        dimension = spec.milvus_dimension
        if not collection_name or not dimension:
            if self.runtime_backend == "milvus":
                raise ValueError(
                    f"engineering partition {spec.directory} has no Milvus artifact"
                )
            self._backend_by_partition[spec.directory] = "faiss"
            self._fallback_reasons[spec.directory] = "milvus_artifact_not_built"
            self._backend_state["reason_code"] = "milvus_artifact_not_built"
            return local_store

        remote = EngineeringMilvusVectorStore(
            self.embedding_manager,
            collection_name,
            expected_dimension=int(dimension),
            expected_count=spec.document_count,
            settings=self.milvus_settings,
            client_factory=self.milvus_client_factory,
            build_id=self.build_id,
            embedding_model=self.embedding_model,
            owner_namespace=spec.milvus_owner_namespace,
        )
        if self.runtime_backend == "milvus":
            remote.validate()
            self._backend_by_partition[spec.directory] = "milvus"
            return remote

        try:
            remote.validate()
        except MilvusAvailabilityError as exc:
            self._backend_by_partition[spec.directory] = "faiss"
            self._fallback_reasons[spec.directory] = str(exc)
            self._backend_state["reason_code"] = "milvus_unavailable"
            return local_store

        self._backend_by_partition[spec.directory] = "milvus"

        def mark_fallback(reason: str, *, partition: str = spec.directory) -> None:
            self._backend_by_partition[partition] = "faiss"
            self._fallback_reasons[partition] = reason
            self._backend_state["reason_code"] = "milvus_unavailable"
            self._refresh_active_backend()

        return AvailabilityFailoverDenseStore(
            remote,
            local_store,
            on_fallback=mark_fallback,
        )

    def _refresh_active_backend(self) -> None:
        values = set(self._backend_by_partition.values())
        if not values:
            active = "unloaded"
        elif len(values) == 1:
            active = next(iter(values))
        else:
            active = "mixed"
        self._backend_state["active_backend"] = active
        fallback_used = bool(self._fallback_reasons)
        self._backend_state["fallback_used"] = fallback_used
        self._backend_state["degraded"] = fallback_used
        if not fallback_used:
            self._backend_state["reason_code"] = None


class _BuildFileLock:
    """Persistent-file advisory lock released automatically on process exit."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    def acquire(self) -> None:
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
                os.fsync(fd)
            os.lseek(fd, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise FileExistsError(self.path) from exc
        self.fd = fd

    def release(self) -> None:
        fd, self.fd = self.fd, None
        if fd is None:
            return
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _acquire_build_lock(lock_path: Path) -> _BuildFileLock:
    """Acquire an OS lock; the stable lock path is never unlinked."""

    lock = _BuildFileLock(lock_path)
    lock.acquire()
    return lock


def _release_build_lock(lock: _BuildFileLock) -> None:
    lock.release()


@contextmanager
def engineering_index_maintenance_lock(root: str | Path):
    """Serialize destructive maintenance with index publication.

    The stable lock file is never deleted; the operating system releases the
    advisory lock if the process exits unexpectedly.
    """

    index_root = Path(root).resolve()
    lock_path = index_root.parent / f".{index_root.name}.build.lock"
    try:
        lock = _acquire_build_lock(lock_path)
    except FileExistsError as exc:
        raise RuntimeError(
            f"engineering index build or maintenance is already running: {lock_path}"
        ) from exc
    try:
        yield
    finally:
        _release_build_lock(lock)


def _process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        return _windows_process_is_running(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _windows_process_is_running(pid: int) -> bool:
    """Query process state without sending a signal on Windows.

    ``os.kill(pid, 0)`` is unsafe on Windows because signal 0 is passed to
    ``TerminateProcess`` rather than acting as the POSIX existence probe.
    """

    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    open_process.restype = wintypes.HANDLE
    get_exit_code = kernel32.GetExitCodeProcess
    get_exit_code.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    get_exit_code.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    handle = open_process(process_query_limited_information, False, pid)
    if not handle:
        # Access denied means the process exists but is not queryable. Invalid
        # parameter is the normal response for a PID that does not exist.
        return ctypes.get_last_error() == 5
    try:
        exit_code = wintypes.DWORD()
        if not get_exit_code(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == still_active
    finally:
        close_handle(handle)


def _recover_interrupted_publish(index_root: Path) -> None:
    parent = index_root.parent
    backups = sorted(parent.glob(f".{index_root.name}.backup-*"))
    stagings = sorted(parent.glob(f".{index_root.name}.staging-*"))
    root_complete = _index_tree_looks_complete(index_root)

    if root_complete:
        # The current root already committed. Compatible backups are cleanup
        # debris; legacy/incomplete backups are preserved under a quarantine
        # name so they cannot block a later current-schema build.
        for backup in backups:
            if _index_tree_looks_complete(backup):
                shutil.rmtree(backup, ignore_errors=True)
            else:
                quarantine = parent / (
                    f".{index_root.name}.quarantine-{uuid.uuid4().hex}"
                )
                os.replace(backup, quarantine)
        for path in stagings:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        return

    if len(backups) > 1:
        raise RuntimeError(
            "ambiguous interrupted engineering-index publish; multiple backups "
            f"must be inspected manually: {[str(path) for path in backups]}"
        )
    backup = backups[0] if backups else None
    if backup is not None and not _index_tree_looks_complete(backup):
        quarantine = parent / f".{index_root.name}.quarantine-{uuid.uuid4().hex}"
        os.replace(backup, quarantine)
        backup = None

    if backup is not None:
        failed_root = parent / f".{index_root.name}.failed-{uuid.uuid4().hex}"
        if index_root.exists():
            os.replace(index_root, failed_root)
        try:
            os.replace(backup, index_root)
        except Exception:
            if failed_root.exists() and not index_root.exists():
                os.replace(failed_root, index_root)
            raise
        if failed_root.exists():
            shutil.rmtree(failed_root, ignore_errors=True)
        backup = None

    for path in stagings:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def _index_tree_looks_complete(root: Path) -> bool:
    catalog_path = root / EngineeringIndex.CATALOG_FILENAME
    if not catalog_path.is_file():
        return False
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog = _validate_catalog_payload(payload)
        resolved_root = root.resolve()
        for spec in catalog.specs:
            directory = spec.directory
            partition = (resolved_root / directory).resolve()
            try:
                partition.relative_to(resolved_root)
            except ValueError:
                return False
            required = ("index.faiss", "documents.json", "index_meta.json")
            checksums = spec.file_sha256
            for name in required:
                path = partition / name
                if not path.is_file() or _sha256_file(path) != checksums[name]:
                    return False
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _validate_catalog_payload(payload: Any) -> _ValidatedCatalog:
    """Validate v3/v4 catalog semantics without filesystem or network I/O."""

    if not isinstance(payload, Mapping):
        raise ValueError("engineering index catalog must be a JSON object")
    try:
        schema_version = int(payload.get("schema_version", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("unsupported engineering index schema") from exc
    if schema_version not in {INDEX_SCHEMA_VERSION, *LEGACY_INDEX_SCHEMA_VERSIONS}:
        raise ValueError("unsupported engineering index schema")

    raw_build_id = payload.get("build_id")
    if not isinstance(raw_build_id, str) or not raw_build_id.strip():
        raise ValueError("engineering index catalog has no build_id")
    build_id = raw_build_id.strip()
    raw_embedding_model = payload.get("embedding_model")
    if not isinstance(raw_embedding_model, str) or not raw_embedding_model.strip():
        raise ValueError("engineering index catalog has no embedding_model")
    embedding_model = raw_embedding_model.strip()

    raw_specs = payload.get("partitions")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("engineering index catalog has no partitions")
    specs: list[PartitionSpec] = []
    seen_directories: set[str] = set()
    seen_collections: set[str] = set()
    required_files = {"index.faiss", "documents.json", "index_meta.json"}
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, Mapping):
            raise ValueError("engineering index partition entry must be an object")
        try:
            spec = PartitionSpec(**dict(raw_spec))
        except (TypeError, ValueError) as exc:
            raise ValueError("engineering index partition entry is invalid") from exc
        if not isinstance(spec.corpus, str) or not isinstance(spec.authority, str):
            raise ValueError("engineering index partition identity must be text")
        if not spec.corpus or not spec.authority:
            raise ValueError("engineering index partition identity is empty")
        if not isinstance(spec.directory, str):
            raise ValueError("engineering index partition directory must be text")
        if not re.fullmatch(r"[a-z0-9_-]+", spec.directory):
            raise ValueError(f"unsafe engineering partition directory: {spec.directory}")
        if spec.directory in seen_directories:
            raise ValueError(f"duplicate engineering partition directory: {spec.directory}")
        seen_directories.add(spec.directory)
        if isinstance(spec.document_count, bool) or not isinstance(spec.document_count, int):
            raise ValueError("engineering index partition count must be an integer")
        if spec.document_count < 0:
            raise ValueError("engineering index partition count must not be negative")
        if isinstance(spec.weight, bool) or not isinstance(spec.weight, (int, float)):
            raise ValueError("engineering index partition weight must be numeric")
        if not math.isfinite(float(spec.weight)) or float(spec.weight) <= 0:
            raise ValueError("engineering index partition weight must be positive")
        if not isinstance(spec.file_sha256, dict) or set(spec.file_sha256) != required_files:
            raise ValueError(
                f"engineering partition {spec.directory} has incomplete checksums"
            )
        if any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in spec.file_sha256.values()
        ):
            raise ValueError(
                f"engineering partition {spec.directory} has an invalid checksum"
            )
        if spec.milvus_collection_name is not None and not isinstance(
            spec.milvus_collection_name, str
        ):
            raise ValueError("engineering Milvus collection name must be text")
        if spec.milvus_collection_name:
            if not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]{0,254}", spec.milvus_collection_name
            ):
                raise ValueError("engineering index catalog has an unsafe Milvus collection")
            if spec.milvus_collection_name in seen_collections:
                raise ValueError("engineering index catalog repeats a Milvus collection")
            seen_collections.add(spec.milvus_collection_name)
        specs.append(spec)

    raw_requested_backend = payload.get("requested_build_backend") or "faiss"
    if not isinstance(raw_requested_backend, str):
        raise ValueError("engineering index catalog has an invalid build backend")
    requested_build_backend = raw_requested_backend.strip().casefold()
    if schema_version == INDEX_SCHEMA_VERSION:
        if requested_build_backend not in BUILD_BACKENDS:
            raise ValueError("engineering index catalog has an invalid build backend")
        expected_available = (
            ("faiss", "milvus")
            if requested_build_backend in {"milvus", "both"}
            else ("faiss",)
        )
        raw_available = payload.get("available_backends")
        if not isinstance(raw_available, list) or tuple(raw_available) != expected_available:
            raise ValueError("engineering index catalog backend artifacts are inconsistent")
        raw_dimension = payload.get("embedding_dimension")
        if isinstance(raw_dimension, bool) or not isinstance(raw_dimension, int):
            raise ValueError(
                "engineering index catalog has no valid embedding dimension"
            )
        embedding_dimension = raw_dimension
        if embedding_dimension <= 0:
            raise ValueError("engineering index embedding dimension must be positive")
    else:
        if requested_build_backend != "faiss":
            raise ValueError("legacy engineering index catalogs are FAISS-only")
        requested_build_backend = "faiss"
        expected_available = ("faiss",)
        embedding_dimension = None

    owners: set[str] = set()
    for spec in specs:
        fields = (
            spec.milvus_collection_name,
            spec.milvus_dimension,
            spec.milvus_metric_type,
            spec.milvus_owner_namespace,
        )
        if expected_available == ("faiss",):
            if any(value is not None for value in fields):
                raise ValueError("FAISS-only catalog must not contain Milvus artifacts")
            continue
        if any(value is None for value in fields):
            raise ValueError("engineering index catalog omitted a Milvus artifact")
        if isinstance(spec.milvus_dimension, bool) or not isinstance(
            spec.milvus_dimension, int
        ):
            raise ValueError("engineering Milvus dimension must be an integer")
        if spec.milvus_dimension != embedding_dimension:
            raise ValueError("engineering FAISS/Milvus dimensions are inconsistent")
        if not isinstance(spec.milvus_metric_type, str):
            raise ValueError("engineering Milvus metric must be text")
        if spec.milvus_metric_type != "COSINE":
            raise ValueError("engineering index catalog has an unsupported Milvus metric")
        if not isinstance(spec.milvus_owner_namespace, str):
            raise ValueError("engineering index catalog has an invalid Milvus owner")
        owner = spec.milvus_owner_namespace
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,63}", owner):
            raise ValueError("engineering index catalog has an invalid Milvus owner")
        owners.add(owner)
    if expected_available == ("faiss", "milvus") and len(owners) != 1:
        raise ValueError("engineering Milvus artifacts have inconsistent owners")

    return _ValidatedCatalog(
        schema_version=schema_version,
        specs=tuple(specs),
        build_id=build_id,
        embedding_model=embedding_model,
        requested_build_backend=requested_build_backend,
        available_backends=expected_available,
        embedding_dimension=embedding_dimension,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 1 else None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _default_embedding_manager(model_name: str | None = None) -> EmbeddingManager:
    selected = model_name or os.getenv(
        "ENGINEERING_EMBEDDING_MODEL_PATH", "BAAI/bge-small-zh-v1.5"
    )
    return EmbeddingManager(EmbeddingConfig(model_path=selected))


def _manager_model_name(manager: Any) -> str:
    config = getattr(manager, "config", None)
    value = getattr(config, "model_path", None)
    return str(value or manager.__class__.__name__)


def _resolve_build_backend(value: str | None) -> str:
    selected = str(
        value
        or os.getenv("ENGINEERING_INDEX_BUILD_BACKEND", "faiss")
    ).strip().casefold()
    if selected not in BUILD_BACKENDS:
        raise ValueError(
            "engineering build backend must be one of: faiss, milvus, both"
        )
    return selected


def _resolve_runtime_backend(value: str | None) -> str:
    selected = str(
        value
        or os.getenv("ENGINEERING_VECTOR_BACKEND", "faiss")
    ).strip().casefold()
    if selected not in RUNTIME_BACKENDS:
        raise ValueError(
            "engineering vector backend must be one of: faiss, milvus, auto"
        )
    return selected
