"""Pinned, validated BGE-M3 embeddings for the trade retrieval pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

import numpy as np

from .contracts import EmbeddingContract

BGE_M3_MODEL = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_M3_DIMENSION = 1024
MODEL_SNAPSHOT_IGNORE_PATTERNS = (
    "onnx/**",
    "imgs/**",
    "*.onnx*",
)


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
        if test_encoder is not None and not test_mode:
            raise ValueError("test_encoder requires explicit test_mode=True")
        if not test_mode and (model_name != BGE_M3_MODEL or revision != BGE_M3_REVISION):
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
        self._snapshot_size_bytes: int | None = None
        self._lock = RLock()

    @property
    def contract(self) -> EmbeddingContract:
        if self._contract is None:
            raise RuntimeError("embedding contract is not available until a vector is observed")
        return self._contract

    @property
    def snapshot_size_bytes(self) -> int | None:
        """Downloaded snapshot byte count, without exposing its local path."""

        return self._snapshot_size_bytes

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

        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        snapshot_path = Path(
            snapshot_download(
                repo_id=self.model_name,
                repo_type="model",
                revision=self.revision,
                cache_dir=str(self.cache_folder) if self.cache_folder is not None else None,
                local_files_only=self.local_files_only,
                ignore_patterns=MODEL_SNAPSHOT_IGNORE_PATTERNS,
            )
        )
        if snapshot_path.name != self.revision:
            raise ValueError("model cache snapshot does not match requested immutable revision")
        self._snapshot_size_bytes = sum(
            entry.stat().st_size for entry in snapshot_path.rglob("*") if entry.is_file()
        )
        self._encoder = SentenceTransformer(
            str(snapshot_path),
            revision=self.revision,
            cache_folder=str(self.cache_folder) if self.cache_folder is not None else None,
            device=self.device,
            trust_remote_code=False,
            local_files_only=True,
        )
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
        if vectors.shape[1] != BGE_M3_DIMENSION:
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
        provider = "test-fake" if self._test_encoder is not None else "sentence-transformers"
        library_version = "test-fake" if self._test_encoder is not None else version("sentence-transformers")
        observed = EmbeddingContract(
            provider=provider,
            model_name=self.model_name,
            requested_revision=self.revision,
            resolved_revision=self.revision,
            dimension=dimension,
            normalized=True,
            dtype="float32",
            library_version=library_version,
        )
        if self._contract is not None and self._contract != observed:
            raise ValueError("embedding contract changed after initial observation")
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
