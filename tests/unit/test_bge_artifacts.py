from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import trade_agent.index.embeddings as embeddings


def _snapshot_with_one_trusted_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[embeddings.BgeEmbeddingManager, Path, Path, dict[str, object]]:
    payload = b"trusted dense model artifact"
    expected: dict[str, object] = {
        "path": "config.json",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "blob_id": "a" * 40,
    }
    monkeypatch.setattr(embeddings, "TRUSTED_ARTIFACTS", (expected,))

    repository = tmp_path / "models--BAAI--bge-m3"
    blob = repository / "blobs" / "trusted-blob"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(payload)
    snapshot = repository / "snapshots" / embeddings.BGE_M3_REVISION
    snapshot.mkdir(parents=True)
    artifact = snapshot / "config.json"
    artifact.symlink_to(blob)
    return embeddings.BgeEmbeddingManager(local_files_only=True), snapshot, artifact, expected


def test_snapshot_verification_hashes_trusted_files_and_ignores_extras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, snapshot, _, expected = _snapshot_with_one_trusted_file(tmp_path, monkeypatch)
    (snapshot / "long.jpg").write_bytes(b"untrusted image asset")
    (snapshot / "onnx").mkdir()
    (snapshot / "onnx" / "model.onnx").write_bytes(b"duplicate weights")

    verified = manager._verify_snapshot(snapshot)

    assert verified.resolved_revision == embeddings.BGE_M3_REVISION
    assert verified.trusted_bytes == expected["size"]
    assert verified.artifact_manifest_sha256 == embeddings.ARTIFACT_MANIFEST_SHA256
    assert verified.artifact_count == 1


def test_snapshot_directory_name_alone_cannot_establish_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, snapshot, artifact, _ = _snapshot_with_one_trusted_file(tmp_path, monkeypatch)
    artifact.unlink()

    with pytest.raises(ValueError, match="missing"):
        manager._verify_snapshot(snapshot)


def test_snapshot_rejects_a_non_regular_required_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, snapshot, artifact, _ = _snapshot_with_one_trusted_file(tmp_path, monkeypatch)
    artifact.unlink()
    artifact.mkdir()

    with pytest.raises(ValueError, match="missing"):
        manager._verify_snapshot(snapshot)


def test_snapshot_rejects_a_required_symlink_outside_the_blob_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, snapshot, artifact, _ = _snapshot_with_one_trusted_file(tmp_path, monkeypatch)
    payload = artifact.read_bytes()
    artifact.unlink()
    outside = tmp_path / "outside-model-cache.bin"
    outside.write_bytes(payload)
    artifact.symlink_to(outside)

    with pytest.raises(ValueError, match="outside"):
        manager._verify_snapshot(snapshot)


def test_snapshot_rejects_an_intermediate_directory_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"trusted nested artifact"
    expected = {
        "path": "nested/config.json",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "blob_id": "a" * 40,
    }
    monkeypatch.setattr(embeddings, "TRUSTED_ARTIFACTS", (expected,))
    repository = tmp_path / "models--BAAI--bge-m3"
    (repository / "blobs").mkdir(parents=True)
    snapshot = repository / "snapshots" / embeddings.BGE_M3_REVISION
    snapshot.mkdir(parents=True)
    outside = tmp_path / "outside-directory"
    outside.mkdir()
    (outside / "config.json").write_bytes(payload)
    (snapshot / "nested").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="outside"):
        embeddings.BgeEmbeddingManager(local_files_only=True)._verify_snapshot(snapshot)


@pytest.mark.parametrize("corruption", ["size", "sha256"])
def test_snapshot_rejects_wrong_size_or_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    manager, snapshot, _, original = _snapshot_with_one_trusted_file(tmp_path, monkeypatch)
    expected = dict(original)
    if corruption == "size":
        expected["size"] = int(expected["size"]) + 1
    else:
        expected["sha256"] = "0" * 64
    monkeypatch.setattr(embeddings, "TRUSTED_ARTIFACTS", (expected,))

    with pytest.raises(ValueError, match="checksum"):
        manager._verify_snapshot(snapshot)


def _official_model_info(**changes: object) -> SimpleNamespace:
    siblings = []
    for expected in embeddings.TRUSTED_ARTIFACTS:
        lfs = (
            SimpleNamespace(sha256=expected["lfs_sha256"])
            if "lfs_sha256" in expected
            else None
        )
        siblings.append(
            SimpleNamespace(
                rfilename=expected["path"],
                size=expected["size"],
                blob_id=expected.get("blob_id"),
                lfs=lfs,
            )
        )
    values = {"sha": embeddings.BGE_M3_REVISION, "siblings": siblings}
    values.update(changes)
    return SimpleNamespace(**values)


class _FakeHfApi:
    def __init__(self, info: SimpleNamespace) -> None:
        self.info = info

    def model_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
        return self.info


def test_official_metadata_accepts_the_pinned_revision_and_manifest() -> None:
    manager = embeddings.BgeEmbeddingManager()

    manager._verify_official_metadata(_FakeHfApi(_official_model_info()))


def test_official_metadata_rejects_a_different_resolved_revision() -> None:
    manager = embeddings.BgeEmbeddingManager()

    with pytest.raises(ValueError, match="resolved revision"):
        manager._verify_official_metadata(_FakeHfApi(_official_model_info(sha="0" * 40)))


@pytest.mark.parametrize("corruption", ["missing", "size", "lfs_sha256", "blob_id"])
def test_official_metadata_rejects_manifest_mismatches(corruption: str) -> None:
    info = _official_model_info()
    if corruption == "missing":
        info.siblings.pop()
    elif corruption == "size":
        info.siblings[0].size += 1
    elif corruption == "lfs_sha256":
        sibling = next(item for item in info.siblings if item.lfs is not None)
        sibling.lfs.sha256 = "0" * 64
    else:
        sibling = next(item for item in info.siblings if item.lfs is None)
        sibling.blob_id = "0" * 40

    with pytest.raises(ValueError, match="metadata|checksum"):
        embeddings.BgeEmbeddingManager()._verify_official_metadata(_FakeHfApi(info))


def test_dense_sentence_transformer_allowlist_excludes_nonruntime_assets() -> None:
    paths = set(embeddings.TRUSTED_ARTIFACT_PATHS)

    assert {
        "modules.json",
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "1_Pooling/config.json",
    } <= paths
    assert not any(
        path.startswith("onnx/")
        or path.endswith((".jpg", ".webp", ".onnx"))
        or path in {"README.md", "colbert_linear.pt", "sparse_linear.pt"}
        for path in paths
    )
