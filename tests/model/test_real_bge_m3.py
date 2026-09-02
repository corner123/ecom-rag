from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import replace
import os
from pathlib import Path
import pickle

from huggingface_hub.constants import HF_HUB_CACHE
import numpy as np
import pytest

import trade_agent.index.contracts as embedding_contracts
from trade_agent.index.contracts import EmbeddingContract
from trade_agent.index.embeddings import (
    ARTIFACT_MANIFEST_SHA256,
    BGE_M3_DIMENSION,
    BGE_M3_REVISION,
    BgeEmbeddingManager,
    TRUSTED_ARTIFACTS,
)


def _model_cache() -> Path:
    return Path(os.environ.get("MODELS__EMBEDDING_CACHE_DIR") or HF_HUB_CACHE)


@pytest.mark.model
def test_real_cached_bge_m3_is_verified_before_producing_vectors() -> None:
    cache = _model_cache()
    snapshot = (
        cache
        / "models--BAAI--bge-m3"
        / "snapshots"
        / BGE_M3_REVISION
    )
    if not snapshot.is_dir():
        pytest.skip("the pinned BAAI/bge-m3 snapshot is not present in the model cache")

    manager = BgeEmbeddingManager(cache_folder=cache, local_files_only=True, batch_size=2)
    documents = manager.embed_documents(
        ["HS 850440 chargers exported to Europe", "采购增长客户需要核验海关月度贸易画像"]
    )
    query = manager.embed_query("查询 HS 850440 采购增长客户")
    contract = manager.contract.require_production()

    assert documents.shape == (2, BGE_M3_DIMENSION)
    assert query.shape == (BGE_M3_DIMENSION,)
    assert documents.dtype == np.float32
    assert query.dtype == np.float32
    assert np.isfinite(documents).all() and np.isfinite(query).all()
    np.testing.assert_allclose(np.linalg.norm(documents, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(query), 1.0, atol=1e-5)
    assert contract.requested_revision == BGE_M3_REVISION
    assert contract.resolved_revision == BGE_M3_REVISION
    assert contract.dimension == BGE_M3_DIMENSION
    assert contract.artifact_manifest_sha256 == ARTIFACT_MANIFEST_SHA256
    assert contract.verified_artifact_count == len(TRUSTED_ARTIFACTS)
    assert manager.snapshot_size_bytes == sum(item.size for item in TRUSTED_ARTIFACTS)
    runtime_view = manager._runtime_view
    assert runtime_view is not None
    visible_runtime_files = {
        path.relative_to(runtime_view.path).as_posix()
        for path in runtime_view.path.rglob("*")
        if path.is_file()
    }
    assert visible_runtime_files == {artifact.path for artifact in TRUSTED_ARTIFACTS}
    assert {
        "README.md",
        "long.jpg",
        "colbert_linear.pt",
        "sparse_linear.pt",
        "model.safetensors",
        "adapter_config.json",
    }.isdisjoint(visible_runtime_files)

    unchanged = contract.model_copy()
    same_value_update = contract.model_copy(update={"dimension": BGE_M3_DIMENSION})
    unchanged.require_production()
    same_value_update.require_production()
    reconstructed = EmbeddingContract.model_validate(contract.model_dump(mode="python"))
    revalidated = EmbeddingContract.model_validate(contract)
    for unattested in (reconstructed, revalidated):
        assert unattested.is_production is False
        with pytest.raises(ValueError, match="verified production"):
            unattested.require_production()
    for field, value in (
        ("provider", "deterministic-test"),
        ("model_name", "BAAI/other"),
        ("requested_revision", "0" * 40),
        ("resolved_revision", "0" * 40),
        ("dimension", BGE_M3_DIMENSION - 1),
        ("normalized", False),
        ("dtype", "float64"),
        ("library_version", "changed"),
        ("artifact_manifest_sha256", "1" * 64),
        ("verified_artifact_count", len(TRUSTED_ARTIFACTS) - 1),
        ("unexpected_field", "not part of the production schema"),
    ):
        mutated = contract.model_copy(update={field: value})
        assert mutated.is_production is False

    receipt = manager._verified_snapshot
    assert receipt is not None

    def issue(candidate: object) -> None:
        embedding_contracts._issue_verified_production_contract(
            receipt=candidate,
        )

    reconstructed_receipt = type(receipt)(
        model_name=receipt.model_name,
        requested_revision=receipt.requested_revision,
        resolved_revision=receipt.resolved_revision,
        trusted_bytes=receipt.trusted_bytes,
        artifact_manifest_sha256=receipt.artifact_manifest_sha256,
        artifact_count=receipt.artifact_count,
    )
    copied_receipts = (
        reconstructed_receipt,
        replace(receipt),
        replace(receipt, trusted_bytes=receipt.trusted_bytes + 1),
        copy(receipt),
        deepcopy(receipt),
        pickle.loads(pickle.dumps(receipt)),
    )
    for copied_receipt in copied_receipts:
        with pytest.raises(ValueError, match="verified snapshot"):
            issue(copied_receipt)

    for field, value in (
        ("model_name", "BAAI/other"),
        ("requested_revision", "0" * 40),
        ("resolved_revision", "0" * 40),
        ("trusted_bytes", receipt.trusted_bytes + 1),
        ("artifact_manifest_sha256", "1" * 64),
        ("artifact_count", len(TRUSTED_ARTIFACTS) - 1),
    ):
        original = getattr(receipt, field)
        object.__setattr__(receipt, field, value)
        with pytest.raises(ValueError, match="verified snapshot"):
            issue(receipt)
        object.__setattr__(receipt, field, original)
