from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import inspect
import pickle

import numpy as np
import pytest
from pydantic import ValidationError

import trade_agent.index.contracts as embedding_contracts
from trade_agent.index.contracts import EmbeddingContract
from trade_agent.index.embeddings import (
    ARTIFACT_MANIFEST_SHA256,
    BGE_M3_DIMENSION,
    BGE_M3_REVISION,
    BgeEmbeddingManager,
    TRUSTED_ARTIFACTS,
    VerifiedEmbeddingSnapshot,
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


class FalseyFakeEncoder(FakeEncoder):
    def __bool__(self) -> bool:
        return False


def _unattested_snapshot_receipt() -> VerifiedEmbeddingSnapshot:
    return VerifiedEmbeddingSnapshot(
        model_name="BAAI/bge-m3",
        requested_revision=BGE_M3_REVISION,
        resolved_revision=BGE_M3_REVISION,
        trusted_bytes=1,
        artifact_manifest_sha256=ARTIFACT_MANIFEST_SHA256,
        artifact_count=len(TRUSTED_ARTIFACTS),
    )


def _issue_from_receipt(receipt: VerifiedEmbeddingSnapshot) -> EmbeddingContract:
    return embedding_contracts._issue_verified_production_contract(
        receipt=receipt,
    )


@pytest.fixture(autouse=True)
def deterministic_test_environment(monkeypatch) -> None:
    monkeypatch.setenv("TEST_EMBEDDING_PROVIDER", "deterministic")


@pytest.fixture
def encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def manager(encoder: FakeEncoder) -> BgeEmbeddingManager:
    return BgeEmbeddingManager(test_encoder=encoder, test_mode=True, batch_size=1)


def test_embeddings_are_normalized_and_record_a_versioned_contract(
    manager: BgeEmbeddingManager,
) -> None:
    vectors = manager.embed_documents(["HS 850440 charger", "采购增长"])
    query = manager.embed_query("采购增长客户")

    assert vectors.shape == (2, BGE_M3_DIMENSION)
    assert vectors.dtype == np.float32
    assert query.shape == (BGE_M3_DIMENSION,)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(query), 1.0, atol=1e-5)
    assert manager.contract.provider == "deterministic-test"
    assert manager.contract.is_production is False
    assert manager.contract.dimension == BGE_M3_DIMENSION
    assert manager.contract.normalized is True
    assert manager.contract.dtype == "float32"


def test_manager_is_lazy_batches_documents_and_keeps_query_semantics_explicit(
    encoder: FakeEncoder,
) -> None:
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


@pytest.mark.parametrize("test_mode,test_encoder", [(True, None), (False, FakeEncoder())])
def test_test_mode_and_test_encoder_must_be_strictly_paired(test_mode, test_encoder) -> None:
    with pytest.raises(ValueError, match="test_mode"):
        BgeEmbeddingManager(test_encoder=test_encoder, test_mode=test_mode)


def test_test_encoder_presence_is_not_inferred_from_its_truthiness() -> None:
    manager = BgeEmbeddingManager(test_encoder=FalseyFakeEncoder(), test_mode=True)

    assert manager.embed_query("HS 850440").shape == (BGE_M3_DIMENSION,)


@pytest.mark.parametrize("test_mode", [None, 0, 1, "true"])
def test_test_mode_requires_a_real_boolean(test_mode: object) -> None:
    with pytest.raises(TypeError, match="boolean"):
        BgeEmbeddingManager(
            test_encoder=FakeEncoder(), test_mode=test_mode  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("provider", [None, "", "hash", "sentence-transformers"])
def test_test_encoder_requires_explicit_deterministic_test_environment(
    monkeypatch, provider
) -> None:
    if provider is None:
        monkeypatch.delenv("TEST_EMBEDDING_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("TEST_EMBEDDING_PROVIDER", provider)
    with pytest.raises(ValueError, match="TEST_EMBEDDING_PROVIDER"):
        BgeEmbeddingManager(test_encoder=FakeEncoder(), test_mode=True)


def test_production_manager_cannot_override_the_pinned_identity() -> None:
    with pytest.raises(ValueError, match="production"):
        BgeEmbeddingManager(model_name="BAAI/bge-small-zh-v1.5")
    with pytest.raises(ValueError, match="production"):
        BgeEmbeddingManager(revision="0" * 40)


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
            artifact_manifest_sha256=ARTIFACT_MANIFEST_SHA256,
            verified_artifact_count=len(TRUSTED_ARTIFACTS),
        )


def test_direct_production_fields_are_valid_identity_but_not_live_attestation() -> None:
    production = EmbeddingContract(
        provider="sentence-transformers",
        model_name="BAAI/bge-m3",
        requested_revision=BGE_M3_REVISION,
        resolved_revision=BGE_M3_REVISION,
        dimension=BGE_M3_DIMENSION,
        normalized=True,
        dtype="float32",
        library_version="3.4.1",
        artifact_manifest_sha256=ARTIFACT_MANIFEST_SHA256,
        verified_artifact_count=len(TRUSTED_ARTIFACTS),
    )
    assert production.provider == "sentence-transformers"
    assert production.is_production is False
    with pytest.raises(ValueError, match="verified production"):
        production.require_production()


def test_production_contract_factory_only_accepts_a_live_snapshot_receipt() -> None:
    signature = inspect.signature(
        embedding_contracts._issue_verified_production_contract
    )

    assert set(signature.parameters) == {"receipt"}
    with pytest.raises(TypeError):
        embedding_contracts._issue_verified_production_contract()
    with pytest.raises(ValueError, match="verified snapshot"):
        _issue_from_receipt(_unattested_snapshot_receipt())


def test_snapshot_receipt_reconstruction_and_copy_cannot_mint_a_contract() -> None:
    direct = _unattested_snapshot_receipt()
    reconstructed = VerifiedEmbeddingSnapshot(
        **{
            "model_name": direct.model_name,
            "requested_revision": direct.requested_revision,
            "resolved_revision": direct.resolved_revision,
            "trusted_bytes": direct.trusted_bytes,
            "artifact_manifest_sha256": direct.artifact_manifest_sha256,
            "artifact_count": direct.artifact_count,
        }
    )
    candidates = (
        direct,
        reconstructed,
        replace(direct),
        replace(direct, trusted_bytes=direct.trusted_bytes + 1),
        pickle.loads(pickle.dumps(direct)),
    )

    for candidate in candidates:
        with pytest.raises(ValueError, match="verified snapshot"):
            _issue_from_receipt(candidate)


def test_manager_cannot_promote_a_directly_constructed_snapshot_receipt() -> None:
    manager = BgeEmbeddingManager()
    manager._verified_snapshot = _unattested_snapshot_receipt()

    with pytest.raises(ValueError, match="verified snapshot"):
        manager._record_contract(BGE_M3_DIMENSION)


def test_changing_only_fake_provider_cannot_forge_production(
    manager: BgeEmbeddingManager,
) -> None:
    manager.embed_query("HS 850440")

    forged = manager.contract.model_copy(update={"provider": "sentence-transformers"})

    assert forged.is_production is False
    with pytest.raises(ValueError, match="verified production"):
        forged.require_production()


def test_embedding_contract_rejects_other_production_and_test_identities() -> None:

    with pytest.raises(ValidationError):
        EmbeddingContract(
            provider="sentence-transformers",
            model_name="other",
            requested_revision=BGE_M3_REVISION,
            resolved_revision=BGE_M3_REVISION,
            dimension=BGE_M3_DIMENSION,
            normalized=True,
            dtype="float32",
            library_version="3.4.1",
            artifact_manifest_sha256=ARTIFACT_MANIFEST_SHA256,
            verified_artifact_count=len(TRUSTED_ARTIFACTS),
        )
    with pytest.raises(ValidationError):
        EmbeddingContract(
            provider="deterministic-test",
            model_name="BAAI/bge-m3",
            requested_revision=BGE_M3_REVISION,
            resolved_revision=BGE_M3_REVISION,
            dimension=BGE_M3_DIMENSION,
            normalized=True,
            dtype="float32",
            library_version="test",
            artifact_manifest_sha256=ARTIFACT_MANIFEST_SHA256,
            verified_artifact_count=len(TRUSTED_ARTIFACTS),
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
            artifact_manifest_sha256=ARTIFACT_MANIFEST_SHA256,
            verified_artifact_count=len(TRUSTED_ARTIFACTS),
        )


@pytest.mark.parametrize(
    ("manifest_sha256", "artifact_count"),
    [("1" * 64, len(TRUSTED_ARTIFACTS)), (ARTIFACT_MANIFEST_SHA256, 1)],
)
def test_production_contract_requires_the_exact_trusted_artifact_manifest(
    manifest_sha256: str, artifact_count: int
) -> None:
    with pytest.raises(ValidationError, match="trusted artifact manifest"):
        EmbeddingContract(
            provider="sentence-transformers",
            model_name="BAAI/bge-m3",
            requested_revision=BGE_M3_REVISION,
            resolved_revision=BGE_M3_REVISION,
            dimension=BGE_M3_DIMENSION,
            normalized=True,
            dtype="float32",
            library_version="3.4.1",
            artifact_manifest_sha256=manifest_sha256,
            verified_artifact_count=artifact_count,
        )


def test_deterministic_contract_cannot_satisfy_a_production_consumer(
    manager: BgeEmbeddingManager,
) -> None:
    manager.embed_query("HS 850440")

    with pytest.raises(ValueError, match="production"):
        manager.contract.require_production()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dimension", BGE_M3_DIMENSION - 1),
        ("normalized", False),
        ("artifact_manifest_sha256", "1" * 64),
        ("verified_artifact_count", 1),
    ],
)
def test_deterministic_contract_has_a_strict_test_only_identity(field: str, value: object) -> None:
    values = {
        "provider": "deterministic-test",
        "model_name": "deterministic-test",
        "requested_revision": "0" * 40,
        "resolved_revision": "0" * 40,
        "dimension": BGE_M3_DIMENSION,
        "normalized": True,
        "dtype": "float32",
        "library_version": "deterministic-test-v1",
        "artifact_manifest_sha256": "0" * 64,
        "verified_artifact_count": 0,
    }
    values[field] = value

    with pytest.raises(ValidationError, match="test-only identity"):
        EmbeddingContract.model_validate(values)
