"""Pinned, validated BGE-M3 embeddings for the trade retrieval pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import hmac
from importlib.metadata import version
import json
import os
from pathlib import Path, PurePosixPath
import re
from threading import RLock
from typing import Any, Protocol

import numpy as np

from .contracts import EmbeddingContract, _issue_verified_production_contract

BGE_M3_MODEL = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_M3_DIMENSION = 1024
TEST_EMBEDDING_PROVIDER = "TEST_EMBEDDING_PROVIDER"
_TEST_REVISION = "0" * 40
_MANIFEST_PATH = Path(__file__).with_name("bge_m3_artifact_manifest.json")
_EXPECTED_ARTIFACT_MANIFEST_SHA256 = (
    "3a862f1d0a8543acc13e9faa5e6d6d1f916ee609b264960be8337e6ba509856b"
)
_EXPECTED_ARTIFACT_COUNT = 10
_MANIFEST_KEYS = {"schema_version", "model_name", "revision", "files"}
_ARTIFACT_COMMON_KEYS = {"path", "size", "sha256"}
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
_LOWER_GIT_BLOB = re.compile(r"[0-9a-f]{40}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError("trusted manifest contains a duplicate JSON key")
        document[key] = value
    return document


def _is_safe_relative_posix_path(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if "\\" in value or "//" in value or any(ord(character) < 32 for character in value):
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _validate_manifest_schema(document: object) -> None:
    if not isinstance(document, dict) or set(document) != _MANIFEST_KEYS:
        raise ValueError("trusted manifest has invalid top-level fields")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("trusted manifest schema_version must be exactly 1")
    if document["model_name"] != BGE_M3_MODEL:
        raise ValueError("trusted manifest model_name does not match BAAI/bge-m3")
    if document["revision"] != BGE_M3_REVISION:
        raise ValueError("trusted manifest revision does not match the pinned revision")
    files = document["files"]
    if not isinstance(files, list) or len(files) != _EXPECTED_ARTIFACT_COUNT:
        raise ValueError("trusted manifest must contain exactly 10 artifacts")

    paths: set[str] = set()
    for artifact in files:
        if not isinstance(artifact, dict):
            raise ValueError("trusted manifest artifact must be an object")
        has_blob = "blob_id" in artifact
        has_lfs = "lfs_sha256" in artifact
        if has_blob == has_lfs:
            raise ValueError("trusted manifest artifact requires exactly one blob or LFS identity")
        lineage_key = "blob_id" if has_blob else "lfs_sha256"
        if set(artifact) != _ARTIFACT_COMMON_KEYS | {lineage_key}:
            raise ValueError("trusted manifest artifact has invalid fields")

        path = artifact["path"]
        if not _is_safe_relative_posix_path(path):
            raise ValueError("trusted manifest path must be a safe relative POSIX path")
        if path in paths:
            raise ValueError("trusted manifest artifact paths must be unique")
        paths.add(path)

        size = artifact["size"]
        if type(size) is not int or size <= 0:
            raise ValueError("trusted manifest artifact size must be a positive integer")
        sha256 = artifact["sha256"]
        if not isinstance(sha256, str) or not _LOWER_SHA256.fullmatch(sha256):
            raise ValueError("trusted manifest artifact sha256 is invalid")
        lineage = artifact[lineage_key]
        lineage_pattern = _LOWER_GIT_BLOB if has_blob else _LOWER_SHA256
        if not isinstance(lineage, str) or not lineage_pattern.fullmatch(lineage):
            raise ValueError(f"trusted manifest artifact {lineage_key} is invalid")


def _load_trusted_manifest(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("trusted manifest package data is unreadable") from error
    canonical_json = json.dumps(
        document,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    observed_digest = hashlib.sha256(canonical_json).hexdigest()
    if not hmac.compare_digest(observed_digest, _EXPECTED_ARTIFACT_MANIFEST_SHA256):
        raise ValueError("trusted manifest digest does not match the embedded trust anchor")
    _validate_manifest_schema(document)
    return document


_MANIFEST = _load_trusted_manifest(_MANIFEST_PATH)
ARTIFACT_MANIFEST_SHA256 = _EXPECTED_ARTIFACT_MANIFEST_SHA256
TRUSTED_ARTIFACTS = tuple(_MANIFEST["files"])
TRUSTED_ARTIFACT_PATHS = tuple(item["path"] for item in TRUSTED_ARTIFACTS)


class Encoder(Protocol):
    def encode(self, texts: Sequence[str], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class VerifiedEmbeddingSnapshot:
    """Content-authenticated local artifacts for one resolved model revision."""

    resolved_revision: str
    trusted_bytes: int
    artifact_manifest_sha256: str
    artifact_count: int


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
        self._verified_snapshot: VerifiedEmbeddingSnapshot | None = None
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

        _load_trusted_manifest(_MANIFEST_PATH)

        from huggingface_hub import HfApi, snapshot_download
        from sentence_transformers import SentenceTransformer

        if not self.local_files_only:
            self._verify_official_metadata(HfApi())
        snapshot_path = Path(
            snapshot_download(
                repo_id=self.model_name,
                repo_type="model",
                revision=self.revision,
                cache_dir=str(self.cache_folder) if self.cache_folder is not None else None,
                local_files_only=self.local_files_only,
                allow_patterns=TRUSTED_ARTIFACT_PATHS,
            )
        )
        self._verified_snapshot = self._verify_snapshot(snapshot_path)
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
        if self._test_encoder is not None:
            provider = "deterministic-test"
            model_name = "deterministic-test"
            requested_revision = _TEST_REVISION
            resolved_revision = _TEST_REVISION
            library_version = "deterministic-test-v1"
            manifest_sha256 = "0" * 64
            artifact_count = 0
        else:
            if self._verified_snapshot is None:
                raise RuntimeError("production vectors require a verified model snapshot")
            provider = "sentence-transformers"
            model_name = self.model_name
            requested_revision = self.revision
            resolved_revision = self._verified_snapshot.resolved_revision
            library_version = version("sentence-transformers")
            manifest_sha256 = self._verified_snapshot.artifact_manifest_sha256
            artifact_count = self._verified_snapshot.artifact_count
        contract_fields = {
            "model_name": model_name,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "dimension": dimension,
            "normalized": True,
            "dtype": "float32",
            "library_version": library_version,
            "artifact_manifest_sha256": manifest_sha256,
            "verified_artifact_count": artifact_count,
        }
        if provider == "sentence-transformers":
            observed = _issue_verified_production_contract(**contract_fields)
        else:
            observed = EmbeddingContract(provider=provider, **contract_fields)
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

    def _verify_official_metadata(self, api: Any) -> None:
        info = api.model_info(self.model_name, revision=self.revision, files_metadata=True)
        if info.sha != self.revision:
            raise ValueError("Hugging Face resolved revision does not match the pinned revision")
        by_path = {item.rfilename: item for item in info.siblings}
        for expected in TRUSTED_ARTIFACTS:
            sibling = by_path.get(expected["path"])
            if sibling is None or sibling.size != expected["size"]:
                raise ValueError(
                    "Hugging Face metadata does not match the trusted artifact manifest"
                )
            lfs_sha = getattr(getattr(sibling, "lfs", None), "sha256", None)
            if "lfs_sha256" in expected:
                if lfs_sha != expected["lfs_sha256"]:
                    raise ValueError(
                        "Hugging Face LFS checksum does not match the trusted artifact manifest"
                    )
            elif getattr(sibling, "blob_id", None) != expected["blob_id"]:
                raise ValueError(
                    "Hugging Face blob metadata does not match the trusted artifact manifest"
                )

    def _verify_snapshot(self, snapshot_path: Path) -> VerifiedEmbeddingSnapshot:
        if snapshot_path.name != self.revision or snapshot_path.parent.name != "snapshots":
            raise ValueError("model cache snapshot does not match requested immutable revision")
        repository_root = snapshot_path.parent.parent
        blobs_root = repository_root / "blobs"
        trusted_bytes = 0
        for expected in TRUSTED_ARTIFACTS:
            artifact = snapshot_path / expected["path"]
            if not artifact.exists() or not artifact.is_file():
                raise ValueError("model snapshot is missing a required trusted artifact")
            resolved_artifact = artifact.resolve(strict=True)
            if artifact.is_symlink():
                if not _is_within(resolved_artifact, blobs_root):
                    raise ValueError(
                        "model snapshot contains a symlink outside the trusted blob store"
                    )
            elif not _is_within(resolved_artifact, snapshot_path):
                raise ValueError(
                    "model snapshot contains an intermediate symlink outside the snapshot"
                )
            size, digest = _sha256_file(artifact)
            if size != expected["size"] or digest != expected["sha256"]:
                raise ValueError(
                    "model snapshot artifact checksum does not match the trusted manifest"
                )
            trusted_bytes += size
        return VerifiedEmbeddingSnapshot(
            resolved_revision=snapshot_path.name,
            trusted_bytes=trusted_bytes,
            artifact_manifest_sha256=ARTIFACT_MANIFEST_SHA256,
            artifact_count=len(TRUSTED_ARTIFACTS),
        )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError:
        return False
    return True


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()
