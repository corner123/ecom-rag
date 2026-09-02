"""Content provenance for the pinned BGE-M3 embedding snapshot."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
from tempfile import TemporaryDirectory
from typing import Any, BinaryIO
import weakref

_BGE_M3_MODEL = "BAAI/bge-m3"
_BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
_ARTIFACT_MANIFEST_SHA256 = (
    "3a862f1d0a8543acc13e9faa5e6d6d1f916ee609b264960be8337e6ba509856b"
)
_ARTIFACT_COUNT = 10
_MANIFEST_KEYS = {"schema_version", "model_name", "revision", "files"}
_ARTIFACT_COMMON_KEYS = {"path", "size", "sha256"}
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
_LOWER_GIT_BLOB = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class TrustedArtifact:
    """One immutable dense SentenceTransformer runtime artifact."""

    path: str
    size: int
    sha256: str
    blob_id: str | None = None
    lfs_sha256: str | None = None


@dataclass(frozen=True, slots=True, weakref_slot=True)
class TrustedManifest:
    """Deeply immutable values loaded from the anchored package manifest."""

    schema_version: int
    model_name: str
    revision: str
    files: tuple[TrustedArtifact, ...]
    canonical_sha256: str

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(artifact.path for artifact in self.files)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class VerifiedEmbeddingSnapshot:
    """A live receipt minted only after all pinned model files are hashed."""

    model_name: str
    requested_revision: str
    resolved_revision: str
    trusted_bytes: int
    artifact_manifest_sha256: str
    artifact_count: int


class _RuntimeView:
    """Private temporary directory exposing only content-verified model files."""

    def __init__(
        self,
        temporary_directory: TemporaryDirectory[str],
        path: Path,
        file_handles: list[BinaryIO],
    ) -> None:
        self._temporary_directory = temporary_directory
        self._file_handles = file_handles
        self._closed = False
        self.path = path

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._temporary_directory.cleanup()
        finally:
            for handle in self._file_handles:
                handle.close()
            self._file_handles.clear()


class _ObjectAttestor:
    """Bind an HMAC to one exact live object identity and its current fields."""

    def __init__(self) -> None:
        self._key = secrets.token_bytes(32)
        self._records: dict[
            int,
            tuple[weakref.ReferenceType[object], bytes],
        ] = {}

    def attest(self, value: object, payload: bytes) -> None:
        identity = id(value)

        def discard(reference: weakref.ReferenceType[object]) -> None:
            current = self._records.get(identity)
            if current is not None and current[0] is reference:
                self._records.pop(identity, None)

        reference = weakref.ref(value, discard)
        mac = hmac.digest(self._key, payload, "sha256")
        self._records[identity] = (reference, mac)

    def verifies(self, value: object, payload: bytes) -> bool:
        record = self._records.get(id(value))
        if record is None or record[0]() is not value:
            return False
        expected = hmac.digest(self._key, payload, "sha256")
        return hmac.compare_digest(record[1], expected)


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


def _validate_manifest_schema(document: object) -> tuple[TrustedArtifact, ...]:
    if not isinstance(document, dict) or set(document) != _MANIFEST_KEYS:
        raise ValueError("trusted manifest has invalid top-level fields")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("trusted manifest schema_version must be exactly 1")
    if document["model_name"] != _BGE_M3_MODEL:
        raise ValueError("trusted manifest model_name does not match BAAI/bge-m3")
    if document["revision"] != _BGE_M3_REVISION:
        raise ValueError("trusted manifest revision does not match the pinned revision")
    files = document["files"]
    if not isinstance(files, list) or len(files) != _ARTIFACT_COUNT:
        raise ValueError("trusted manifest must contain exactly 10 artifacts")

    paths: set[str] = set()
    immutable_artifacts: list[TrustedArtifact] = []
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
        immutable_artifacts.append(
            TrustedArtifact(
                path=path,
                size=size,
                sha256=sha256,
                blob_id=lineage if has_blob else None,
                lfs_sha256=lineage if has_lfs else None,
            )
        )
    return tuple(immutable_artifacts)


def _manifest_payload(manifest: TrustedManifest) -> bytes:
    artifacts: list[dict[str, object]] = []
    for artifact in manifest.files:
        item: dict[str, object] = {
            "path": artifact.path,
            "size": artifact.size,
            "sha256": artifact.sha256,
        }
        if artifact.blob_id is not None:
            item["blob_id"] = artifact.blob_id
        if artifact.lfs_sha256 is not None:
            item["lfs_sha256"] = artifact.lfs_sha256
        artifacts.append(item)
    return json.dumps(
        {
            "schema_version": manifest.schema_version,
            "model_name": manifest.model_name,
            "revision": manifest.revision,
            "files": artifacts,
            "canonical_sha256": manifest.canonical_sha256,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _receipt_payload(receipt: VerifiedEmbeddingSnapshot) -> bytes:
    return json.dumps(
        {
            "model_name": receipt.model_name,
            "requested_revision": receipt.requested_revision,
            "resolved_revision": receipt.resolved_revision,
            "trusted_bytes": receipt.trusted_bytes,
            "artifact_manifest_sha256": receipt.artifact_manifest_sha256,
            "artifact_count": receipt.artifact_count,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _prepared_view_payload(
    runtime_view: _RuntimeView,
    snapshot_path: Path,
    manifest: TrustedManifest,
    trusted_bytes: int,
) -> bytes:
    return json.dumps(
        {
            "runtime_path": str(runtime_view.path.resolve(strict=True)),
            "snapshot_path": str(snapshot_path.resolve(strict=True)),
            "manifest_sha256": manifest.canonical_sha256,
            "trusted_bytes": trusted_bytes,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


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


def _verify_snapshot_contents(
    snapshot_path: Path,
    *,
    expected_revision: str,
    artifacts: tuple[TrustedArtifact, ...],
) -> int:
    """Verify immutable artifact expectations without issuing authority."""

    if snapshot_path.name != expected_revision or snapshot_path.parent.name != "snapshots":
        raise ValueError("model cache snapshot does not match requested immutable revision")
    if type(artifacts) is not tuple or not artifacts:
        raise ValueError("snapshot verification requires immutable artifact expectations")
    repository_root = snapshot_path.parent.parent
    blobs_root = repository_root / "blobs"
    trusted_bytes = 0
    for expected in artifacts:
        if type(expected) is not TrustedArtifact:
            raise TypeError("snapshot artifact expectations must be TrustedArtifact values")
        artifact = snapshot_path / expected.path
        if not artifact.exists() or not artifact.is_file():
            raise ValueError("model snapshot is missing a required trusted artifact")
        resolved_artifact = artifact.resolve(strict=True)
        if artifact.is_symlink():
            if not _is_within(resolved_artifact, blobs_root):
                raise ValueError("model snapshot contains a symlink outside the trusted blob store")
        elif not _is_within(resolved_artifact, snapshot_path):
            raise ValueError(
                "model snapshot contains an intermediate symlink outside the snapshot"
            )
        size, digest = _sha256_file(artifact)
        if size != expected.size or digest != expected.sha256:
            raise ValueError(
                "model snapshot artifact checksum does not match the trusted manifest"
            )
        trusted_bytes += size
    return trusted_bytes


def _materialize_runtime_view(
    snapshot_path: Path,
    artifacts: tuple[TrustedArtifact, ...],
) -> _RuntimeView:
    """Expose only trusted filenames to the downstream model loader."""

    if type(artifacts) is not tuple or not artifacts:
        raise ValueError("runtime view requires immutable artifact expectations")
    temporary_directory = TemporaryDirectory(prefix="trade-agent-bge-runtime-")
    runtime_root = Path(temporary_directory.name) / "model"
    runtime_root.mkdir(mode=0o700)
    file_handles: list[BinaryIO] = []
    descriptor_root = Path("/proc/self/fd")
    if not descriptor_root.is_dir():
        descriptor_root = Path("/dev/fd")
    if not descriptor_root.is_dir():
        temporary_directory.cleanup()
        raise RuntimeError("the platform does not expose stable process file descriptors")
    fallback_errors = {
        errno.EACCES,
        errno.EPERM,
        errno.EROFS,
        errno.EXDEV,
    }
    try:
        for expected in artifacts:
            if type(expected) is not TrustedArtifact:
                raise TypeError("runtime artifacts must be TrustedArtifact values")
            if not _is_safe_relative_posix_path(expected.path):
                raise ValueError("runtime artifact path must be a safe relative POSIX path")
            source = (snapshot_path / expected.path).resolve(strict=True)
            destination = runtime_root / expected.path
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                os.link(source, destination)
            except OSError as error:
                if error.errno not in fallback_errors:
                    raise
                descriptor = os.open(
                    source,
                    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
                )
                handle = os.fdopen(descriptor, "rb", closefd=True)
                try:
                    destination.symlink_to(descriptor_root / str(handle.fileno()))
                except BaseException:
                    handle.close()
                    raise
                file_handles.append(handle)
        return _RuntimeView(temporary_directory, runtime_root, file_handles)
    except BaseException:
        temporary_directory.cleanup()
        for handle in file_handles:
            handle.close()
        raise


def _verify_runtime_view_contents(
    runtime_root: Path,
    artifacts: tuple[TrustedArtifact, ...],
) -> int:
    """Hash the exact sanitized file set that the model loader will see."""

    expected_paths = {artifact.path for artifact in artifacts}
    visible_paths = {
        path.relative_to(runtime_root).as_posix()
        for path in runtime_root.rglob("*")
        if path.is_file()
    }
    if visible_paths != expected_paths:
        raise ValueError("verified runtime view does not contain the exact trusted file set")
    trusted_bytes = 0
    for expected in artifacts:
        artifact = runtime_root / expected.path
        size, digest = _sha256_file(artifact)
        if size != expected.size or digest != expected.sha256:
            raise ValueError("verified runtime view checksum does not match the trusted manifest")
        trusted_bytes += size
    return trusted_bytes


def _build_provenance_authority() -> tuple[object, ...]:
    manifest_attestor = _ObjectAttestor()
    prepared_view_attestor = _ObjectAttestor()
    receipt_attestor = _ObjectAttestor()
    manifest_payload = _manifest_payload
    prepared_view_payload = _prepared_view_payload
    receipt_payload = _receipt_payload
    verify_contents = _verify_snapshot_contents
    materialize_runtime_view = _materialize_runtime_view
    verify_runtime_view = _verify_runtime_view_contents

    def require_manifest(manifest: TrustedManifest) -> TrustedManifest:
        if type(manifest) is not TrustedManifest:
            raise ValueError("a live trusted manifest is required")
        if (
            manifest.schema_version != 1
            or manifest.model_name != _BGE_M3_MODEL
            or manifest.revision != _BGE_M3_REVISION
            or manifest.canonical_sha256 != _ARTIFACT_MANIFEST_SHA256
            or len(manifest.files) != _ARTIFACT_COUNT
        ):
            raise ValueError("a live trusted manifest is required")
        try:
            payload = manifest_payload(manifest)
        except (AttributeError, TypeError, ValueError):
            raise ValueError("a live trusted manifest is required") from None
        if not manifest_attestor.verifies(manifest, payload):
            raise ValueError("a live trusted manifest is required")
        return manifest

    def load_manifest(path: Path) -> TrustedManifest:
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
        if not hmac.compare_digest(observed_digest, _ARTIFACT_MANIFEST_SHA256):
            raise ValueError("trusted manifest digest does not match the embedded trust anchor")
        artifacts = _validate_manifest_schema(document)
        manifest = TrustedManifest(
            schema_version=document["schema_version"],
            model_name=document["model_name"],
            revision=document["revision"],
            files=artifacts,
            canonical_sha256=observed_digest,
        )
        manifest_attestor.attest(manifest, manifest_payload(manifest))
        return manifest

    def prepare_runtime_view(
        snapshot_path: Path,
        trusted_manifest: TrustedManifest,
    ) -> tuple[int, _RuntimeView]:
        manifest = require_manifest(trusted_manifest)
        trusted_bytes = verify_contents(
            snapshot_path,
            expected_revision=manifest.revision,
            artifacts=manifest.files,
        )
        runtime_view = materialize_runtime_view(snapshot_path, manifest.files)
        try:
            runtime_bytes = verify_runtime_view(runtime_view.path, manifest.files)
            if runtime_bytes != trusted_bytes:
                raise ValueError("verified runtime view byte count changed after snapshot check")
            prepared_view_attestor.attest(
                runtime_view,
                prepared_view_payload(
                    runtime_view,
                    snapshot_path,
                    manifest,
                    runtime_bytes,
                ),
            )
            return runtime_bytes, runtime_view
        except BaseException:
            runtime_view.cleanup()
            raise

    def verify_loaded_runtime_and_issue_receipt(
        snapshot_path: Path,
        trusted_manifest: TrustedManifest,
        runtime_view: _RuntimeView,
        trusted_bytes: int,
    ) -> VerifiedEmbeddingSnapshot:
        manifest = require_manifest(trusted_manifest)
        if type(runtime_view) is not _RuntimeView or type(trusted_bytes) is not int:
            raise ValueError("a live prepared runtime view is required")
        try:
            preparation = prepared_view_payload(
                runtime_view,
                snapshot_path,
                manifest,
                trusted_bytes,
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            raise ValueError("a live prepared runtime view is required") from None
        if not prepared_view_attestor.verifies(runtime_view, preparation):
            raise ValueError("a live prepared runtime view is required")

        runtime_bytes = verify_runtime_view(runtime_view.path, manifest.files)
        if runtime_bytes != trusted_bytes:
            raise ValueError("verified runtime view byte count changed during model loading")
        receipt = VerifiedEmbeddingSnapshot(
            model_name=manifest.model_name,
            requested_revision=manifest.revision,
            resolved_revision=snapshot_path.name,
            trusted_bytes=runtime_bytes,
            artifact_manifest_sha256=manifest.canonical_sha256,
            artifact_count=len(manifest.files),
        )
        receipt_attestor.attest(receipt, receipt_payload(receipt))
        return receipt

    def require_receipt(
        receipt: VerifiedEmbeddingSnapshot,
    ) -> VerifiedEmbeddingSnapshot:
        if type(receipt) is not VerifiedEmbeddingSnapshot:
            raise ValueError("a live verified snapshot receipt is required")
        if (
            receipt.model_name != _BGE_M3_MODEL
            or receipt.requested_revision != _BGE_M3_REVISION
            or receipt.resolved_revision != _BGE_M3_REVISION
            or type(receipt.trusted_bytes) is not int
            or receipt.trusted_bytes <= 0
            or receipt.artifact_manifest_sha256 != _ARTIFACT_MANIFEST_SHA256
            or receipt.artifact_count != _ARTIFACT_COUNT
        ):
            raise ValueError("a live verified snapshot receipt is required")
        try:
            payload = receipt_payload(receipt)
        except (AttributeError, TypeError, ValueError):
            raise ValueError("a live verified snapshot receipt is required") from None
        if not receipt_attestor.verifies(receipt, payload):
            raise ValueError("a live verified snapshot receipt is required")
        return receipt

    return (
        load_manifest,
        require_manifest,
        prepare_runtime_view,
        verify_loaded_runtime_and_issue_receipt,
        require_receipt,
    )


(
    _load_trusted_manifest,
    _require_trusted_manifest,
    _prepare_verified_runtime_view,
    _verify_loaded_runtime_and_issue_receipt,
    _require_verified_snapshot,
) = _build_provenance_authority()
del _build_provenance_authority
