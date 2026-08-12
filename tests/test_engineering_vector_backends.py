from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pytest

from rag_core.engineering import EngineeringIndex
from rag_core.engineering.index import _recover_interrupted_publish
from rag_core.evaluation import engineering_adapter
from rag_core.engineering.workflows import cleanup_engineering_milvus
from rag_core.index.engineering_milvus_store import (
    EngineeringMilvusVectorStore,
    MilvusBackendError,
    MilvusConnectionSettings,
    MilvusAvailabilityError,
    MilvusIntegrityError,
    cleanup_stale_engineering_collections,
)
from rag_core.ingestion import BuildManifest
from rag_core.sources.schema import ChunkRecord, DocumentRecord, SourceRecord


class _EmbeddingManager:
    dimension = 8

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        vector = [0.0] * cls.dimension
        for token in text.casefold().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[digest[0] % cls.dimension] += 1.0
        if not any(vector):
            vector[0] = 1.0
        return vector

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]


def _manifest() -> BuildManifest:
    source = SourceRecord(
        source_id="mini",
        source_type="git_repository",
        uri="https://example.test/mini.git",
        commit_sha="a" * 40,
        metadata={"corpus": "internal"},
    )
    documents = [
        DocumentRecord(
            "mini",
            "docs/architecture.md",
            "checkpoint design rationale",
            doc_id="design",
            metadata={"custom_nested": {"owner": "runtime"}},
        ),
        DocumentRecord(
            "mini",
            "mini_nanobot/checkpoint.py",
            "CheckpointStore persists agent state",
            doc_id="code",
        ),
    ]
    chunks = [
        ChunkRecord(
            source.source_id,
            document.doc_id,
            0,
            document.content,
            chunk_id=f"stable_chunk_{index}",
        )
        for index, document in enumerate(documents)
    ]
    return BuildManifest(
        build_id="build_dual_backend_test",
        created_at="2026-08-13T00:00:00+00:00",
        sources=[source],
        documents=documents,
        chunks=chunks,
    )


class _Schema:
    def __init__(self, description: str):
        self.description = description
        self.fields: list[dict[str, Any]] = []

    def add_field(self, **kwargs: Any) -> None:
        self.fields.append(dict(kwargs))


class _IndexParams:
    def __init__(self):
        self.items: list[dict[str, Any]] = []

    def add_index(self, **kwargs: Any) -> None:
        self.items.append(dict(kwargs))


@dataclass
class _MilvusState:
    collections: dict[str, dict[str, Any]] = field(default_factory=dict)
    create_failure_after_commit: str | None = None
    inspect_failure: str | None = None
    search_failure: str | None = None
    drop_failure: str | None = None


class _FakeMilvusClient:
    data_types = {"VARCHAR": "VARCHAR", "FLOAT_VECTOR": "FLOAT_VECTOR"}

    def __init__(self, state: _MilvusState):
        self.state = state

    def has_collection(self, *, collection_name: str) -> bool:
        return collection_name in self.state.collections

    def create_schema(self, *, description: str, **_: Any) -> _Schema:
        return _Schema(description)

    def prepare_index_params(self) -> _IndexParams:
        return _IndexParams()

    def create_collection(
        self,
        *,
        collection_name: str,
        schema: _Schema,
        index_params: _IndexParams,
    ) -> None:
        vector_field = next(
            item for item in schema.fields if item["field_name"] == "dense_vector"
        )
        self.state.collections[collection_name] = {
            "description": schema.description,
            "dimension": int(vector_field["dim"]),
            "index": list(index_params.items),
            "rows": [],
        }
        if self.state.create_failure_after_commit:
            raise RuntimeError(self.state.create_failure_after_commit)

    def insert(self, *, collection_name: str, data: list[dict[str, Any]]) -> None:
        self.state.collections[collection_name]["rows"].extend(
            json.loads(json.dumps(data))
        )

    def flush(self, **_: Any) -> None:
        return None

    def load_collection(self, **_: Any) -> None:
        return None

    def get_collection_stats(self, *, collection_name: str) -> dict[str, int]:
        self._raise_inspect_failure()
        return {"row_count": len(self.state.collections[collection_name]["rows"])}

    def describe_collection(self, *, collection_name: str) -> dict[str, Any]:
        self._raise_inspect_failure()
        collection = self.state.collections[collection_name]
        return {
            "description": collection["description"],
            "fields": [
                {"name": "dense_vector", "params": {"dim": collection["dimension"]}}
            ],
        }

    def search(
        self,
        *,
        collection_name: str,
        data: list[list[float]],
        limit: int,
        **_: Any,
    ) -> list[list[dict[str, Any]]]:
        if self.state.search_failure:
            raise RuntimeError(self.state.search_failure)
        query = np.asarray(data[0], dtype="float32")
        rows = self.state.collections[collection_name]["rows"]
        scored = sorted(
            rows,
            key=lambda row: float(np.dot(query, np.asarray(row["dense_vector"]))),
            reverse=True,
        )[:limit]
        return [[
            {
                "id": row["chunk_id"],
                "distance": float(np.dot(query, np.asarray(row["dense_vector"]))),
                "entity": {
                    "chunk_id": row["chunk_id"],
                    "text": row["text"],
                    "metadata_json": row["metadata_json"],
                },
            }
            for row in scored
        ]]

    def drop_collection(self, *, collection_name: str) -> None:
        if self.state.drop_failure:
            raise RuntimeError(self.state.drop_failure)
        self.state.collections.pop(collection_name, None)

    def list_collections(self) -> list[str]:
        return list(self.state.collections)

    def _raise_inspect_failure(self) -> None:
        if self.state.inspect_failure:
            raise RuntimeError(self.state.inspect_failure)


def _stored_collection(
    *,
    owner: str = "pytest_dual_backend",
    build_id: str = "cleanup-build",
    dimension: int = 8,
) -> dict[str, Any]:
    return {
        "description": "engineering-rag:"
        + json.dumps(
            {
                "build_id": build_id,
                "embedding_model": "test-deterministic-embedding",
                "dimension": dimension,
                "owner_namespace": owner,
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        "dimension": dimension,
        "index": [],
        "rows": [],
    }


def _factory(state: _MilvusState):
    def create(_settings):
        return _FakeMilvusClient(state)

    return create


@pytest.fixture(autouse=True)
def _unique_test_milvus_owner(monkeypatch: pytest.MonkeyPatch):
    """Give every fake/opt-in Milvus artifact an explicit test deployment."""

    monkeypatch.setenv("ENGINEERING_MILVUS_NAMESPACE", "pytest_dual_backend")


def test_both_build_keeps_faiss_and_round_trips_milvus_metadata(tmp_path: Path):
    state = _MilvusState()
    manager = _EmbeddingManager()
    root = tmp_path / "index"
    index = EngineeringIndex.build(
        _manifest(),
        root,
        embedding_manager=manager,
        backend="both",
        milvus_client_factory=_factory(state),
    )

    catalog = json.loads((root / "partitions.json").read_text(encoding="utf-8"))
    assert catalog["schema_version"] == 4
    assert catalog["available_backends"] == ["faiss", "milvus"]
    assert index.stats()["vector_backend"]["active_backend"] == "milvus"
    assert all((root / spec.directory / "index.faiss").is_file() for spec in index.specs)
    assert all(spec.milvus_collection_name in state.collections for spec in index.specs)

    loaded = EngineeringIndex.load(
        root,
        embedding_manager=manager,
        runtime_backend="milvus",
        milvus_client_factory=_factory(state),
    )
    results = loaded.federated.search(
        "checkpoint design", corpora="internal", authorities="design", top_k=2
    )
    assert results
    assert results[0].metadata["chunk_id"] == "stable_chunk_0"
    assert results[0].metadata["custom_nested"] == {"owner": "runtime"}
    assert loaded.stats()["vector_backend"]["active_backend"] == "milvus"
    public_connection = next(iter(loaded._stores)).get_stats()["connection"]
    assert "uri" not in public_connection


def test_auto_falls_back_only_for_transport_availability(tmp_path: Path):
    state = _MilvusState()
    manager = _EmbeddingManager()
    root = tmp_path / "index"
    EngineeringIndex.build(
        _manifest(),
        root,
        embedding_manager=manager,
        backend="milvus",
        milvus_client_factory=_factory(state),
    )
    loaded = EngineeringIndex.load(
        root,
        embedding_manager=manager,
        runtime_backend="auto",
        milvus_client_factory=_factory(state),
    )
    assert loaded.stats()["vector_backend"]["active_backend"] == "milvus"

    state.search_failure = "StatusCode.UNAVAILABLE: connection refused"
    results = loaded.federated.search("CheckpointStore", top_k=2)
    assert results
    backend = loaded.stats()["vector_backend"]
    assert backend["active_backend"] == "faiss"
    assert backend["fallback_used"] is True
    assert backend["degraded"] is True
    assert backend["reason_code"] == "milvus_unavailable"
    assert backend["fallback_reasons"]


@pytest.mark.parametrize(
    "runtime_backend,error_text,error_type",
    [
        ("milvus", "connection refused", MilvusAvailabilityError),
        ("milvus", "Permission denied: invalid token", MilvusIntegrityError),
        ("auto", "Permission denied: invalid token", MilvusIntegrityError),
        ("auto", "unexpected SDK programming failure", MilvusBackendError),
    ],
)
def test_distributed_query_errors_are_not_swallowed_by_federation(
    tmp_path: Path,
    runtime_backend: str,
    error_text: str,
    error_type: type[Exception],
):
    state = _MilvusState()
    manager = _EmbeddingManager()
    root = tmp_path / runtime_backend
    EngineeringIndex.build(
        _manifest(),
        root,
        embedding_manager=manager,
        backend="both",
        milvus_client_factory=_factory(state),
    )
    loaded = EngineeringIndex.load(
        root,
        embedding_manager=manager,
        runtime_backend=runtime_backend,
        milvus_client_factory=_factory(state),
    )
    state.search_failure = error_text

    with pytest.raises(error_type):
        loaded.federated.search("CheckpointStore", top_k=2)

    backend = loaded.stats()["vector_backend"]
    assert backend["fallback_used"] is False
    assert backend["active_backend"] == "milvus"


def test_auto_does_not_hide_auth_or_schema_failures(tmp_path: Path):
    state = _MilvusState()
    manager = _EmbeddingManager()
    root = tmp_path / "index"
    EngineeringIndex.build(
        _manifest(),
        root,
        embedding_manager=manager,
        backend="both",
        milvus_client_factory=_factory(state),
    )

    state.inspect_failure = "Permission denied: invalid token"
    with pytest.raises(MilvusIntegrityError):
        EngineeringIndex.load(
            root,
            embedding_manager=manager,
            runtime_backend="auto",
            milvus_client_factory=_factory(state),
        )


def test_auto_load_falls_back_when_milvus_service_is_unavailable(tmp_path: Path):
    state = _MilvusState()
    manager = _EmbeddingManager()
    root = tmp_path / "index"
    EngineeringIndex.build(
        _manifest(),
        root,
        embedding_manager=manager,
        backend="both",
        milvus_client_factory=_factory(state),
    )
    state.inspect_failure = "deadline exceeded: service unavailable"

    loaded = EngineeringIndex.load(
        root,
        embedding_manager=manager,
        runtime_backend="auto",
        milvus_client_factory=_factory(state),
    )
    assert loaded.stats()["vector_backend"]["active_backend"] == "faiss"
    health = loaded.stats()["vector_backend"]
    assert len(health["fallback_reasons"]) == 2
    assert health["requested_mode"] == "auto"
    assert health["primary_backend"] == "milvus"
    assert health["fallback_used"] is True
    assert health["reason_code"] == "milvus_unavailable"


def test_strict_milvus_rejects_faiss_only_catalog(tmp_path: Path):
    manager = _EmbeddingManager()
    root = tmp_path / "index"
    EngineeringIndex.build(
        _manifest(), root, embedding_manager=manager, backend="faiss"
    )
    with pytest.raises(ValueError, match="no Milvus artifact"):
        EngineeringIndex.load(
            root, embedding_manager=manager, runtime_backend="milvus"
        )


def test_schema_v3_catalog_remains_loadable_as_faiss(tmp_path: Path):
    manager = _EmbeddingManager()
    root = tmp_path / "index"
    EngineeringIndex.build(
        _manifest(), root, embedding_manager=manager, backend="faiss"
    )
    path = root / "partitions.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    catalog["schema_version"] = 3
    catalog.pop("requested_build_backend")
    catalog.pop("available_backends")
    for partition in catalog["partitions"]:
        partition.pop("milvus_collection_name")
        partition.pop("milvus_dimension")
        partition.pop("milvus_metric_type")
        partition.pop("milvus_owner_namespace")
    path.write_text(json.dumps(catalog), encoding="utf-8")

    loaded = EngineeringIndex.load(
        root, embedding_manager=manager, runtime_backend="faiss"
    )
    assert loaded.catalog_schema_version == 3
    assert loaded.stats()["vector_backend"]["active_backend"] == "faiss"


def test_faiss_catalog_rejects_embedded_milvus_artifact(tmp_path: Path):
    root = tmp_path / "faiss-contract"
    manager = _EmbeddingManager()
    EngineeringIndex.build(
        _manifest(), root, embedding_manager=manager, backend="faiss"
    )
    path = root / "partitions.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    partition = catalog["partitions"][0]
    partition.update(
        {
            "milvus_collection_name": "engineering_rag_injected_collection",
            "milvus_dimension": manager.dimension,
            "milvus_metric_type": "COSINE",
            "milvus_owner_namespace": "pytest_dual_backend",
        }
    )
    path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(ValueError, match="FAISS-only catalog"):
        EngineeringIndex.load(
            root, embedding_manager=manager, runtime_backend="faiss"
        )


@pytest.mark.parametrize(
    "corruption,error_match",
    [
        ("unsafe_collection", "unsafe Milvus collection"),
        ("inconsistent_owner", "inconsistent owners"),
        ("top_dimension", "dimensions are inconsistent"),
    ],
)
def test_dual_catalog_rejects_unsafe_or_inconsistent_remote_contract(
    tmp_path: Path,
    corruption: str,
    error_match: str,
):
    state = _MilvusState()
    manager = _EmbeddingManager()
    root = tmp_path / corruption
    EngineeringIndex.build(
        _manifest(),
        root,
        embedding_manager=manager,
        backend="both",
        milvus_client_factory=_factory(state),
    )
    path = root / "partitions.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if corruption == "unsafe_collection":
        catalog["partitions"][0]["milvus_collection_name"] = "../unsafe"
    elif corruption == "inconsistent_owner":
        catalog["partitions"][1]["milvus_owner_namespace"] = "another_deployment"
    else:
        catalog["embedding_dimension"] = manager.dimension + 1
    path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(ValueError, match=error_match):
        EngineeringIndex.load(
            root, embedding_manager=manager, runtime_backend="faiss"
        )


def test_milvus_build_requires_explicit_deployment_namespace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.delenv("ENGINEERING_MILVUS_NAMESPACE", raising=False)

    with pytest.raises(ValueError, match="ENGINEERING_MILVUS_NAMESPACE is required"):
        EngineeringIndex.build(
            _manifest(),
            tmp_path / "missing-owner",
            embedding_manager=_EmbeddingManager(),
            backend="both",
            milvus_client_factory=_factory(_MilvusState()),
        )

    assert not (tmp_path / "missing-owner").exists()


def test_stale_manifest_must_be_resynced_before_build(tmp_path: Path):
    manifest = _manifest()
    manifest.schema_version = "1.0"

    with pytest.raises(ValueError, match="run sources-sync"):
        EngineeringIndex.build(
            manifest,
            tmp_path / "index",
            embedding_manager=_EmbeddingManager(),
            backend="faiss",
        )

    assert not (tmp_path / "index").exists()


def test_formal_evaluation_forces_faiss_despite_runtime_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    observed: list[str | None] = []

    class _Index:
        pass

    def fake_load(*_args, **kwargs):
        observed.append(kwargs.get("runtime_backend"))
        return _Index()

    monkeypatch.setenv("ENGINEERING_VECTOR_BACKEND", "milvus")
    monkeypatch.setattr(engineering_adapter.EngineeringIndex, "load", fake_load)
    monkeypatch.setattr(engineering_adapter, "_assert_current_build", lambda _index: None)
    monkeypatch.setattr(
        engineering_adapter,
        "_public_index_metadata",
        lambda _index: {"build_id": "x"},
    )
    monkeypatch.setattr(
        engineering_adapter,
        "_retriever_view",
        lambda _index, mode, candidate_multiplier=None: mode,
    )

    engineering_adapter.create_index_ablation_predictors(index_root="unused")
    assert observed == ["faiss"]


def test_faiss_build_and_load_ignore_invalid_milvus_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("ENGINEERING_MILVUS_TIMEOUT_SECONDS", "not-a-number")
    root = tmp_path / "faiss"

    built = EngineeringIndex.build(
        _manifest(),
        root,
        embedding_manager=_EmbeddingManager(),
        backend="faiss",
    )
    loaded = EngineeringIndex.load(
        root,
        embedding_manager=_EmbeddingManager(),
        runtime_backend="faiss",
    )

    assert built.stats()["vector_backend"]["active_backend"] == "faiss"
    assert loaded.stats()["vector_backend"]["active_backend"] == "faiss"


def test_auto_faiss_only_load_ignores_invalid_milvus_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    root = tmp_path / "faiss-auto"
    EngineeringIndex.build(
        _manifest(),
        root,
        embedding_manager=_EmbeddingManager(),
        backend="faiss",
    )
    monkeypatch.setenv("ENGINEERING_MILVUS_TIMEOUT_SECONDS", "not-a-number")

    loaded = EngineeringIndex.load(
        root,
        embedding_manager=_EmbeddingManager(),
        runtime_backend="auto",
    )

    backend = loaded.stats()["vector_backend"]
    assert backend["active_backend"] == "faiss"
    assert backend["fallback_used"] is True
    assert backend["reason_code"] == "milvus_artifact_not_built"


def test_strict_milvus_build_does_not_publish_on_unavailable_service(tmp_path: Path):
    state = _MilvusState(inspect_failure="")

    def unavailable_factory(_settings):
        raise RuntimeError("connection refused")

    with pytest.raises(MilvusAvailabilityError):
        EngineeringIndex.build(
            _manifest(),
            tmp_path / "index",
            embedding_manager=_EmbeddingManager(),
            backend="milvus",
            milvus_client_factory=unavailable_factory,
        )
    assert not (tmp_path / "index").exists()


def test_failed_post_write_validation_removes_unpublished_collection(tmp_path: Path):
    state = _MilvusState()
    manager = _EmbeddingManager()
    client = _FakeMilvusClient(state)

    class WrongCountClient:
        """Delegate writes, then expose a corrupt post-flush row count."""

        data_types = client.data_types

        def __getattr__(self, name: str):
            return getattr(client, name)

        def get_collection_stats(self, *, collection_name: str) -> dict[str, int]:
            assert collection_name in state.collections
            return {"row_count": 0}

    store = EngineeringMilvusVectorStore(
        manager,
        "engineering_rag_validation_cleanup_test",
        expected_dimension=manager.dimension,
        expected_count=1,
        client=WrongCountClient(),
        data_types=client.data_types,
        build_id="build_validation_cleanup",
        embedding_model="test-deterministic-embedding",
    )
    document = __import__(
        "langchain_core.documents", fromlist=["Document"]
    ).Document(
        page_content="checkpoint design rationale",
        metadata={"chunk_id": "validation_cleanup_chunk"},
    )

    with pytest.raises(MilvusIntegrityError, match="row count mismatch"):
        store.replace_documents([document])

    assert state.collections == {}


def test_lost_create_response_still_removes_server_side_collection():
    state = _MilvusState(
        create_failure_after_commit="connection reset after collection commit"
    )
    manager = _EmbeddingManager()
    store = EngineeringMilvusVectorStore(
        manager,
        "engineering_rag_create_response_loss_test",
        expected_dimension=manager.dimension,
        expected_count=1,
        client_factory=_factory(state),
        build_id="build_create_response_loss",
        embedding_model="test-deterministic-embedding",
    )
    document = __import__(
        "langchain_core.documents", fromlist=["Document"]
    ).Document(
        page_content="checkpoint design rationale",
        metadata={"chunk_id": "create_response_loss_chunk"},
    )

    with pytest.raises(MilvusAvailabilityError):
        store.replace_documents([document])

    assert state.collections == {}


def test_cleanup_failure_preserves_build_error_and_releases_lock(tmp_path: Path):
    state = _MilvusState(
        inspect_failure="Permission denied: invalid token",
        drop_failure="Permission denied while deleting collection",
    )
    root = tmp_path / "failed-build"

    with pytest.raises(MilvusIntegrityError) as captured:
        EngineeringIndex.build(
            _manifest(),
            root,
            embedding_manager=_EmbeddingManager(),
            backend="both",
            milvus_client_factory=_factory(state),
        )

    notes = getattr(captured.value, "__notes__", [])
    assert any("cleanup failed" in note for note in notes)
    assert not root.exists()
    assert not list(tmp_path.glob(".failed-build.staging-*"))

    # The stable lock file remains by design, but the OS lock must be free.
    state.inspect_failure = None
    state.drop_failure = None
    rebuilt = EngineeringIndex.build(
        _manifest(),
        root,
        embedding_manager=_EmbeddingManager(),
        backend="faiss",
    )
    assert rebuilt.stats()["vector_backend"]["active_backend"] == "faiss"


def test_ignore_missing_does_not_hide_milvus_transport_errors():
    class BrokenInspectionClient:
        def has_collection(self, *, collection_name: str) -> bool:
            raise RuntimeError("connection refused")

    store = EngineeringMilvusVectorStore(
        _EmbeddingManager(),
        "engineering_rag_drop_error_test",
        expected_dimension=_EmbeddingManager.dimension,
        expected_count=0,
        client=BrokenInspectionClient(),
        data_types={"VARCHAR": "VARCHAR", "FLOAT_VECTOR": "FLOAT_VECTOR"},
        build_id="drop_error",
        embedding_model="test-deterministic-embedding",
    )

    with pytest.raises(MilvusAvailabilityError):
        store.drop_collection(ignore_missing=True)


def test_stale_collection_cleanup_is_dry_run_then_explicit():
    state = _MilvusState(
        collections={
            "engineering_rag_aaaaaaaaaaaa_1234567890_internal__code": _stored_collection(),
            "engineering_rag_bbbbbbbbbbbb_1234567890_internal__design": _stored_collection(),
            "engineering_rag_cccccccccccc_1234567890_internal__test": _stored_collection(
                owner="another_deployment"
            ),
            "unrelated_collection": {},
        }
    )
    active = ["engineering_rag_aaaaaaaaaaaa_1234567890_internal__code"]
    preview = cleanup_stale_engineering_collections(
        active,
        client_factory=_factory(state),
    )
    assert preview["deleted"] == []
    assert preview["candidates"] == [
        "engineering_rag_bbbbbbbbbbbb_1234567890_internal__design"
    ]
    assert len(state.collections) == 4

    applied = cleanup_stale_engineering_collections(
        active,
        execute=True,
        client_factory=_factory(state),
        owner_namespace="pytest_dual_backend",
    )
    assert applied["deleted"] == preview["candidates"]
    assert set(state.collections) == {
        *active,
        "engineering_rag_cccccccccccc_1234567890_internal__test",
        "unrelated_collection",
    }


def test_cleanup_refuses_execute_without_active_catalog_artifact(tmp_path: Path):
    state = _MilvusState(
        collections={
            "engineering_rag_bbbbbbbbbbbb_1234567890_internal__design": _stored_collection()
        }
    )
    root = tmp_path / "faiss-only"
    EngineeringIndex.build(
        _manifest(),
        root,
        embedding_manager=_EmbeddingManager(),
        backend="faiss",
    )

    def forbidden_factory(_settings):
        raise AssertionError("FAISS-only ownerless dry-run must not contact Milvus")

    preview = cleanup_engineering_milvus(
        root,
        milvus_client_factory=forbidden_factory,
    )
    assert preview["candidates"] == []
    assert preview["owner_namespace"] is None
    assert "FAISS-only" in preview["warning"]
    with pytest.raises(ValueError, match="refusing Milvus deletion"):
        cleanup_engineering_milvus(
            root,
            execute=True,
            milvus_client_factory=_factory(state),
        )

    assert state.collections


def test_cleanup_describe_transport_error_is_classified_and_raised():
    state = _MilvusState(
        collections={
            "engineering_rag_bbbbbbbbbbbb_1234567890_internal__design": _stored_collection()
        },
        inspect_failure="connection refused",
    )

    with pytest.raises(MilvusAvailabilityError):
        cleanup_stale_engineering_collections(
            [],
            client_factory=_factory(state),
            owner_namespace="pytest_dual_backend",
        )


def test_cleanup_execute_refuses_candidates_not_seen_in_dry_run():
    state = _MilvusState(
        collections={
            "engineering_rag_bbbbbbbbbbbb_1234567890_internal__design": _stored_collection()
        }
    )

    with pytest.raises(RuntimeError, match="candidates changed"):
        cleanup_stale_engineering_collections(
            [],
            execute=True,
            client_factory=_factory(state),
            owner_namespace="pytest_dual_backend",
            expected_candidates=[],
        )

    assert len(state.collections) == 1


def test_recovery_restores_valid_backup_when_current_catalog_is_semantically_corrupt(
    tmp_path: Path,
):
    root = tmp_path / "recoverable"
    manager = _EmbeddingManager()
    EngineeringIndex.build(
        _manifest(), root, embedding_manager=manager, backend="faiss"
    )
    backup = tmp_path / ".recoverable.backup-interrupted"
    shutil.copytree(root, backup)
    catalog_path = root / "partitions.json"
    corrupt = json.loads(catalog_path.read_text(encoding="utf-8"))
    corrupt["available_backends"] = ["faiss", "milvus"]
    catalog_path.write_text(json.dumps(corrupt), encoding="utf-8")

    _recover_interrupted_publish(root)

    recovered = EngineeringIndex.load(
        root,
        embedding_manager=manager,
        runtime_backend="faiss",
    )
    assert recovered.build_id == _manifest().build_id
    assert not backup.exists()
    restored = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert restored["available_backends"] == ["faiss"]


@pytest.mark.milvus
def test_real_milvus_round_trip_when_explicit_test_uri_is_configured():
    """Opt-in real integration; never targets the application's collection."""

    uri = os.getenv("MILVUS_TEST_URI", "").strip()
    if not uri:
        pytest.skip("set MILVUS_TEST_URI to run the real Milvus integration test")
    token = os.getenv("MILVUS_TEST_TOKEN", "").strip() or None
    database = os.getenv("MILVUS_TEST_DATABASE", "default").strip() or "default"
    unique = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
    collection = f"test_engineering_rag_{unique}"
    manager = _EmbeddingManager()
    docs = [
        __import__("langchain_core.documents", fromlist=["Document"]).Document(
            page_content="CheckpointStore persists agent state",
            metadata={"chunk_id": "real_test_chunk", "nested": {"safe": True}},
        )
    ]
    store = EngineeringMilvusVectorStore(
        manager,
        collection,
        expected_dimension=manager.dimension,
        expected_count=1,
        settings=MilvusConnectionSettings(
            uri=uri,
            token=token,
            database=database,
            timeout_seconds=10.0,
        ),
        build_id=f"integration_{unique}",
        embedding_model="test-deterministic-embedding",
    )
    try:
        store.replace_documents(docs)
        store.validate()
        result = store.search_dense("CheckpointStore", top_k=1)[0]
        assert result["chunk_id"] == "real_test_chunk"
        assert result["metadata"]["nested"] == {"safe": True}
    finally:
        store.drop_collection(ignore_missing=True)
