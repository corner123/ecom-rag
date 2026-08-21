"""Small workflow functions intended for ``main.py`` and automation scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rag_core.ingestion import BuildManifest, IngestionPipeline, IngestionResult
from rag_core.sources import SourceCatalog
from rag_core.sources.official_web import Fetcher
from rag_core.index.engineering_milvus_store import (
    MilvusConnectionSettings,
    cleanup_stale_engineering_collections,
)

from .index import EngineeringIndex, engineering_index_maintenance_lock
from .service import EngineeringRAGService
from .target import resolve_target_repository


DEFAULT_CATALOG = Path("data/sources/ecommerce_demo.yaml")
DEFAULT_MANIFEST = Path("data/manifests/builds/ecommerce_demo.json")
DEFAULT_INDEX = Path("data/indexes/ecommerce_demo")


def sync_engineering_sources(
    catalog_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    *,
    chunk_size: int = 1_200,
    chunk_overlap: int = 120,
    web_fetcher: Fetcher | None = None,
    full_rebuild: bool = False,
) -> IngestionResult:
    """Collect sources and atomically publish a current-schema manifest.

    Normal syncs read the previous manifest to produce an incremental diff.
    ``full_rebuild`` is an explicit migration boundary: it ignores an older
    snapshot rather than silently coercing its schema or metadata.
    """

    catalog_file = Path(
        catalog_path or os.getenv("SOURCE_CATALOG_PATH", str(DEFAULT_CATALOG))
    )
    manifest_file = Path(manifest_path or DEFAULT_MANIFEST)
    catalog = SourceCatalog.load(catalog_file)
    previous = None if full_rebuild else (manifest_file if manifest_file.is_file() else None)
    pipeline = IngestionPipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return pipeline.build(
        catalog.create_sources(web_fetcher=web_fetcher),
        previous=previous,
        output_path=manifest_file,
    )


def build_engineering_index(
    manifest_path: str | Path | None = None,
    index_root: str | Path | None = None,
    *,
    embedding_manager=None,
    batch_size: int = 128,
    backend: str | None = None,
    milvus_settings=None,
    milvus_client_factory=None,
) -> EngineeringIndex:
    """Build all internal/official partitions from a completed manifest."""

    manifest = BuildManifest.read(manifest_path or DEFAULT_MANIFEST)
    root = index_root or os.getenv("ENGINEERING_INDEX_DIR", str(DEFAULT_INDEX))
    return EngineeringIndex.build(
        manifest,
        root,
        embedding_manager=embedding_manager,
        batch_size=batch_size,
        backend=backend,
        milvus_settings=milvus_settings,
        milvus_client_factory=milvus_client_factory,
    )


def load_engineering_service(
    index_root: str | Path | None = None,
    *,
    target_repo: str | Path | None = None,
    mini_nanobot_repo: str | Path | None = None,
    manifest_path: str | Path | None = None,
    embedding_manager=None,
    answerer=None,
    runtime_backend: str | None = None,
    milvus_settings=None,
    milvus_client_factory=None,
) -> EngineeringRAGService:
    """Load the query service with a fixed, configured live-code root."""

    root = index_root or os.getenv("ENGINEERING_INDEX_DIR", str(DEFAULT_INDEX))
    return EngineeringRAGService.from_index(
        root,
        target_repo=resolve_target_repository(
            target_repo,
            legacy_repo=mini_nanobot_repo,
        ),
        manifest_path=manifest_path or os.getenv(
            "ENGINEERING_MANIFEST_PATH", str(DEFAULT_MANIFEST)
        ),
        embedding_manager=embedding_manager,
        answerer=answerer,
        runtime_backend=runtime_backend,
        milvus_settings=milvus_settings,
        milvus_client_factory=milvus_client_factory,
    )


def query_engineering_knowledge(
    query: str,
    *,
    top_k: int = 5,
    answer: bool = False,
    service: EngineeringRAGService | None = None,
    **load_options: Any,
) -> dict[str, Any]:
    """Return a JSON-ready retrieval or grounded-answer payload."""

    active = service or load_engineering_service(**load_options)
    if answer:
        return active.answer(query, top_k=top_k).to_dict()
    return active.retrieve(query, top_k=top_k).to_dict()


def cleanup_engineering_milvus(
    index_root: str | Path | None = None,
    *,
    execute: bool = False,
    namespace: str | None = None,
    milvus_settings: MilvusConnectionSettings | None = None,
    milvus_client_factory=None,
) -> dict[str, Any]:
    """Dry-run or delete stale build-scoped Milvus collections explicitly."""

    root = Path(
        index_root or os.getenv("ENGINEERING_INDEX_DIR", str(DEFAULT_INDEX))
    ).resolve()
    catalog_path = root / EngineeringIndex.CATALOG_FILENAME
    with engineering_index_maintenance_lock(root):
        # Loading through the FAISS path validates catalog schema, checksums,
        # counts, dimensions and partition metadata without contacting Milvus.
        index = EngineeringIndex.load(root, runtime_backend="faiss")
        active = sorted(
            str(spec.milvus_collection_name)
            for spec in index.specs
            if spec.milvus_collection_name
        )
        owners = {
            str(spec.milvus_owner_namespace)
            for spec in index.specs
            if spec.milvus_owner_namespace
        }
        if active and len(owners) != 1:
            raise ValueError("engineering Milvus artifacts have inconsistent owners")
        catalog_owner = next(iter(owners), None)
        if execute and not active:
            raise ValueError(
                "refusing Milvus deletion without an active v4 Milvus artifact; "
                "run a dry-run or point --index-dir at the currently published "
                "dual-backend index"
            )
        if not active and not namespace:
            # A FAISS-only catalog contains no deployment ownership proof.
            # A default/global prefix is not a safe substitute, even for a
            # preview, because it would encourage deleting another deployment's
            # collections. Operators may provide a namespace for an explicit
            # owner-scoped dry run.
            return {
                "dry_run": True,
                "active_collections": [],
                "owner_namespace": None,
                "candidates": [],
                "deleted": [],
                "warning": (
                    "FAISS-only catalog has no Milvus owner; pass --namespace "
                    "for an owner-scoped dry run"
                ),
            }
        if execute and not namespace:
            raise ValueError(
                "--namespace is required for Milvus deletion; use the owner "
                "shown by the dry-run"
            )
        if namespace and catalog_owner and namespace != catalog_owner:
            raise ValueError("cleanup namespace does not match the active catalog owner")
        catalog_hash = _sha256_file(catalog_path)
        preview = cleanup_stale_engineering_collections(
            active,
            execute=False,
            settings=milvus_settings,
            client_factory=milvus_client_factory,
            owner_namespace=catalog_owner or namespace,
        )
        if not execute:
            return preview
        # Re-check immediately before deletion. Cooperative publishers are
        # excluded by the shared OS lock; this also catches direct file edits.
        if _sha256_file(catalog_path) != catalog_hash:
            raise RuntimeError("engineering index catalog changed during cleanup")
        return cleanup_stale_engineering_collections(
            active,
            execute=True,
            settings=milvus_settings,
            client_factory=milvus_client_factory,
            owner_namespace=catalog_owner,
            expected_candidates=preview["candidates"],
        )


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
