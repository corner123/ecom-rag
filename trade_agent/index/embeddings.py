"""Pinned, validated BGE-M3 embeddings for the trade retrieval pipeline."""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

import numpy as np

from .contracts import EmbeddingContract, _issue_verified_production_contract
from .provenance import (
    TrustedArtifact,
    TrustedManifest,
    VerifiedEmbeddingSnapshot,
    _RuntimeView,
    _load_trusted_manifest,
    _prepare_verified_runtime_view,
    _require_trusted_manifest,
    _validate_manifest_schema,
    _verify_loaded_runtime_and_issue_receipt,
    _verify_snapshot_contents,
)

_PINNED_BGE_M3_MODEL = "BAAI/bge-m3"
_PINNED_BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
_PINNED_BGE_M3_DIMENSION = 1024
_PINNED_ARTIFACT_MANIFEST_SHA256 = (
    "3a862f1d0a8543acc13e9faa5e6d6d1f916ee609b264960be8337e6ba509856b"
)
BGE_M3_MODEL = _PINNED_BGE_M3_MODEL
BGE_M3_REVISION = _PINNED_BGE_M3_REVISION
BGE_M3_DIMENSION = _PINNED_BGE_M3_DIMENSION
TEST_EMBEDDING_PROVIDER = "TEST_EMBEDDING_PROVIDER"
_TEST_REVISION = "0" * 40
_MANIFEST_PATH = Path(__file__).with_name("bge_m3_artifact_manifest.json")
_OBSERVABLE_MANIFEST = _load_trusted_manifest(_MANIFEST_PATH)
ARTIFACT_MANIFEST_SHA256 = _PINNED_ARTIFACT_MANIFEST_SHA256
TRUSTED_ARTIFACTS = _OBSERVABLE_MANIFEST.files
TRUSTED_ARTIFACT_PATHS = _OBSERVABLE_MANIFEST.paths


class Encoder(Protocol):
    def encode(self, texts: Sequence[str], **kwargs: Any) -> Any: ...


class BgeEmbeddingManager:
    """Lazily load one immutable BGE-M3 snapshot and produce validated vectors.

    Documents and queries intentionally use the same BGE-M3 encode path.  The
    two public methods preserve call-site intent; neither claims a prompt or
    a query transform that the selected model does not perform.
    """

    def __init__(
        self,
        model_name: str = BGE_M3_MODEL,
        revision: str = BGE_M3_REVISION,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        cache_folder: str | Path | None = None,
        local_files_only: bool = False,
        max_characters: int = 16_000,
        test_encoder: Encoder | None = None,
        test_mode: bool = False,
    ) -> None:
        if not model_name.strip() or not revision.strip() or not device.strip():
            raise ValueError("model name, revision, and device must not be blank")
        if batch_size < 1:
            raise ValueError("batch_size must be at least one")
        if max_characters < 1:
            raise ValueError("max_characters must be at least one")
        if not isinstance(test_mode, bool):
            raise TypeError("test_mode must be a boolean")
        if test_mode != (test_encoder is not None):
            raise ValueError("test_mode and test_encoder must be supplied together")
        if test_mode and os.environ.get(TEST_EMBEDDING_PROVIDER) != "deterministic":
            raise ValueError("TEST_EMBEDDING_PROVIDER=deterministic is required for a test encoder")
        if not test_mode and (
            model_name != _PINNED_BGE_M3_MODEL
            or revision != _PINNED_BGE_M3_REVISION
        ):
            raise ValueError("production embeddings must use the pinned BAAI/bge-m3 revision")

        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.cache_folder = Path(cache_folder) if cache_folder is not None else None
        self.local_files_only = local_files_only
        self.max_characters = max_characters
        self._test_encoder = test_encoder
        self._test_mode = test_mode
        self._encoder: Encoder | None = None
        self._contract: EmbeddingContract | None = None
        self._verified_snapshot: VerifiedEmbeddingSnapshot | None = None
        self._runtime_view: _RuntimeView | None = None
        self._lock = RLock()

    @property
    def contract(self) -> EmbeddingContract:
        if self._contract is None:
            raise RuntimeError("embedding contract is not available until a vector is observed")
        return self._contract

    @property
    def snapshot_size_bytes(self) -> int | None:
        """Downloaded snapshot byte count, without exposing its local path."""

        if self._verified_snapshot is None:
            return None
        return self._verified_snapshot.trusted_bytes

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        validated = self._validate_documents(texts)
        with self._lock:
            encoder = self._load_encoder()
            batches = [
                self._encode_batch(encoder, validated[start : start + self.batch_size])
                for start in range(0, len(validated), self.batch_size)
            ]
            return np.vstack(batches).astype(np.float32, copy=False)

    def embed_query(self, text: str) -> np.ndarray:
        validated = self._validate_text(text, field="query")
        with self._lock:
            encoder = self._load_encoder()
            vectors = self._encode_batch(encoder, [validated])
            return vectors[0]

    def _load_encoder(self) -> Encoder:
        if self._encoder is not None:
            return self._encoder
        if self._test_encoder is not None:
            self._encoder = self._test_encoder
            return self._encoder

        trusted_manifest = _load_trusted_manifest(_MANIFEST_PATH)

        from huggingface_hub import HfApi, snapshot_download
        from sentence_transformers import SentenceTransformer

        if not self.local_files_only:
            self._verify_official_metadata(HfApi(), trusted_manifest)
        snapshot_path = Path(
            snapshot_download(
                repo_id=self.model_name,
                repo_type="model",
                revision=self.revision,
                cache_dir=str(self.cache_folder) if self.cache_folder is not None else None,
                local_files_only=self.local_files_only,
                allow_patterns=trusted_manifest.paths,
            )
        )
        trusted_bytes, runtime_view = self._verify_snapshot(
            snapshot_path,
            trusted_manifest,
        )
        try:
            encoder = SentenceTransformer(
                str(runtime_view.path),
                revision=self.revision,
                cache_folder=(
                    str(self.cache_folder) if self.cache_folder is not None else None
                ),
                device=self.device,
                trust_remote_code=False,
                local_files_only=True,
            )
            verified_snapshot = self._verify_loaded_snapshot(
                snapshot_path,
                trusted_manifest,
                runtime_view,
                trusted_bytes,
            )
        except BaseException:
            runtime_view.cleanup()
            raise
        self._verified_snapshot = verified_snapshot
        self._runtime_view = runtime_view
        self._encoder = encoder
        return self._encoder

    def _encode_batch(self, encoder: Encoder, texts: Sequence[str]) -> np.ndarray:
        raw = encoder.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        vectors = np.asarray(raw)
        if vectors.ndim != 2:
            raise ValueError("encoder output must be two-dimensional")
        if vectors.shape[0] != len(texts):
            raise ValueError("encoder output rows must match input rows")
        if vectors.shape[1] != _PINNED_BGE_M3_DIMENSION:
            raise ValueError("encoder output dimension must be 1024 for BAAI/bge-m3")
        vectors = vectors.astype(np.float32, copy=False)
        if not np.isfinite(vectors).all():
            raise ValueError("encoder output must be finite")
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
        if not np.isfinite(norms).all() or np.any(norms == 0.0):
            raise ValueError("encoder output must have non-zero finite norms")
        normalized = (vectors / norms[:, None]).astype(np.float32, copy=False)
        if not np.allclose(np.linalg.norm(normalized, axis=1), 1.0, rtol=0.0, atol=1e-5):
            raise ValueError("encoder output normalization failed")
        self._record_contract(normalized.shape[1])
        return normalized

    def _record_contract(self, dimension: int) -> None:
        if self._test_encoder is not None:
            observed = EmbeddingContract(
                provider="deterministic-test",
                model_name="deterministic-test",
                requested_revision=_TEST_REVISION,
                resolved_revision=_TEST_REVISION,
                dimension=dimension,
                normalized=True,
                dtype="float32",
                library_version="deterministic-test-v1",
                artifact_manifest_sha256="0" * 64,
                verified_artifact_count=0,
            )
        else:
            if self._verified_snapshot is None:
                raise RuntimeError("production vectors require a verified model snapshot")
            observed = _issue_verified_production_contract(
                receipt=self._verified_snapshot,
            )
        if self._contract is not None:
            if self._contract.model_dump(mode="python") != observed.model_dump(mode="python"):
                raise ValueError("embedding contract changed after initial observation")
            if self._test_encoder is None:
                self._contract.require_production()
            return
        self._contract = observed

    def _validate_documents(self, texts: Sequence[str]) -> list[str]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("documents must be a sequence of strings, not a string")
        if not texts:
            raise ValueError("documents must not be empty")
        return [self._validate_text(text, field="document") for text in texts]

    def _validate_text(self, text: object, *, field: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"{field} texts must be strings")
        if not text.strip():
            raise ValueError(f"{field} text must not be blank")
        if len(text) > self.max_characters:
            raise ValueError(f"{field} text exceeds the configured maximum length")
        return text

    def _verify_official_metadata(
        self,
        api: Any,
        trusted_manifest: TrustedManifest,
    ) -> None:
        manifest = _require_trusted_manifest(trusted_manifest)
        info = api.model_info(
            manifest.model_name,
            revision=manifest.revision,
            files_metadata=True,
        )
        if info.sha != manifest.revision:
            raise ValueError("Hugging Face resolved revision does not match the pinned revision")
        by_path = {item.rfilename: item for item in info.siblings}
        for expected in manifest.files:
            sibling = by_path.get(expected.path)
            if sibling is None or sibling.size != expected.size:
                raise ValueError(
                    "Hugging Face metadata does not match the trusted artifact manifest"
                )
            lfs_sha = getattr(getattr(sibling, "lfs", None), "sha256", None)
            if expected.lfs_sha256 is not None:
                if lfs_sha != expected.lfs_sha256:
                    raise ValueError(
                        "Hugging Face LFS checksum does not match the trusted artifact manifest"
                    )
            elif getattr(sibling, "blob_id", None) != expected.blob_id:
                raise ValueError(
                    "Hugging Face blob metadata does not match the trusted artifact manifest"
                )

    def _verify_snapshot(
        self,
        snapshot_path: Path,
        trusted_manifest: TrustedManifest,
    ) -> tuple[int, _RuntimeView]:
        return _prepare_verified_runtime_view(
            snapshot_path,
            trusted_manifest,
        )

    def _verify_loaded_snapshot(
        self,
        snapshot_path: Path,
        trusted_manifest: TrustedManifest,
        runtime_view: _RuntimeView,
        trusted_bytes: int,
    ) -> VerifiedEmbeddingSnapshot:
        return _verify_loaded_runtime_and_issue_receipt(
            snapshot_path,
            trusted_manifest,
            runtime_view,
            trusted_bytes,
        )
