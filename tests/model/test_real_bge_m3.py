from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE
import numpy as np
import pytest

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
    assert manager.snapshot_size_bytes == sum(int(item["size"]) for item in TRUSTED_ARTIFACTS)
