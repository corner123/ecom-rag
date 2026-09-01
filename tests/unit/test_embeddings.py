from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from pydantic import ValidationError

from trade_agent.index.contracts import EmbeddingContract
from trade_agent.index.embeddings import (
    BGE_M3_DIMENSION,
    BGE_M3_REVISION,
    BgeEmbeddingManager,
)


class FakeEncoder:
    """A deliberately narrow test-only encoder."""

    def __init__(self, outputs: Sequence[np.ndarray] | None = None) -> None:
        self.outputs = list(outputs or [])
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, texts: Sequence[str], **kwargs: object) -> np.ndarray:
        self.calls.append((list(texts), kwargs))
        if self.outputs:
            return self.outputs.pop(0)
        rows = np.arange(1, len(texts) + 1, dtype=np.float32)[:, None]
        return np.repeat(rows, BGE_M3_DIMENSION, axis=1)


@pytest.fixture
def encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def manager(encoder: FakeEncoder) -> BgeEmbeddingManager:
    return BgeEmbeddingManager(test_encoder=encoder, test_mode=True, batch_size=1)


def test_embeddings_are_normalized_and_record_a_versioned_contract(manager: BgeEmbeddingManager) -> None:
    vectors = manager.embed_documents(["HS 850440 charger", "采购增长"])
    query = manager.embed_query("采购增长客户")

    assert vectors.shape == (2, BGE_M3_DIMENSION)
    assert vectors.dtype == np.float32
    assert query.shape == (BGE_M3_DIMENSION,)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(query), 1.0, atol=1e-5)
    assert manager.contract.provider == "test-fake"
    assert manager.contract.model_name == "BAAI/bge-m3"
    assert manager.contract.revision == BGE_M3_REVISION
    assert manager.contract.requested_revision == BGE_M3_REVISION
    assert manager.contract.resolved_revision == BGE_M3_REVISION
    assert manager.contract.dimension == BGE_M3_DIMENSION
    assert manager.contract.normalized is True
    assert manager.contract.dtype == "float32"


def test_manager_is_lazy_batches_documents_and_keeps_query_semantics_explicit(encoder: FakeEncoder) -> None:
    manager = BgeEmbeddingManager(test_encoder=encoder, test_mode=True, batch_size=1)

    assert encoder.calls == []
    with pytest.raises(RuntimeError, match="not available"):
        _ = manager.contract

    manager.embed_documents(["first", "second"])
    manager.embed_query("question")

    assert [call[0] for call in encoder.calls] == [["first"], ["second"], ["question"]]
    assert all(call[1]["normalize_embeddings"] is False for call in encoder.calls)
    assert all(call[1]["convert_to_numpy"] is True for call in encoder.calls)


@pytest.mark.parametrize(
    ("documents", "message"),
    [
        ("not-a-sequence", "sequence"),
        ([], "empty"),
        ([""], "blank"),
        (["   "], "blank"),
        ([1], "strings"),
        (["x" * 17], "maximum"),
    ],
)
def test_embed_documents_rejects_invalid_texts(documents: object, message: str) -> None:
    manager = BgeEmbeddingManager(test_encoder=FakeEncoder(), test_mode=True, max_characters=16)

    with pytest.raises((TypeError, ValueError), match=message):
        manager.embed_documents(documents)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "query",
    ["", "   ", 123, ["question"], "x" * 17],
)
def test_embed_query_rejects_invalid_text(query: object) -> None:
    manager = BgeEmbeddingManager(test_encoder=FakeEncoder(), test_mode=True, max_characters=16)

    with pytest.raises((TypeError, ValueError)):
        manager.embed_query(query)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_output",
    [
        np.full((1, BGE_M3_DIMENSION), np.nan, dtype=np.float32),
        np.zeros((1, BGE_M3_DIMENSION), dtype=np.float32),
        np.ones((2, BGE_M3_DIMENSION), dtype=np.float32),
        np.ones((1, BGE_M3_DIMENSION - 1), dtype=np.float32),
        np.ones(BGE_M3_DIMENSION, dtype=np.float32),
    ],
)
def test_embed_query_rejects_malformed_encoder_output(invalid_output: np.ndarray) -> None:
    manager = BgeEmbeddingManager(
        test_encoder=FakeEncoder(outputs=[invalid_output]), test_mode=True
    )

    with pytest.raises(ValueError, match="(finite|non-zero|rows|dimension|two-dimensional)"):
        manager.embed_query("question")


def test_test_encoder_requires_explicit_test_mode() -> None:
    with pytest.raises(ValueError, match="test_mode"):
        BgeEmbeddingManager(test_encoder=FakeEncoder())


def test_embedding_contract_rejects_floating_revisions_and_wrong_bge_m3_dimension() -> None:
    with pytest.raises(ValidationError, match="40-character lowercase SHA"):
        EmbeddingContract(
            provider="sentence-transformers",
            model_name="BAAI/bge-m3",
            requested_revision="main",
            resolved_revision="main",
            dimension=BGE_M3_DIMENSION,
            normalized=True,
            dtype="float32",
            library_version="3.4.1",
        )
    with pytest.raises(ValidationError, match="1024"):
        EmbeddingContract(
            provider="sentence-transformers",
            model_name="BAAI/bge-m3",
            requested_revision=BGE_M3_REVISION,
            resolved_revision=BGE_M3_REVISION,
            dimension=BGE_M3_DIMENSION - 1,
            normalized=True,
            dtype="float32",
            library_version="3.4.1",
        )
