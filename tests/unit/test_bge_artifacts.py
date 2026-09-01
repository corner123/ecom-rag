from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import trade_agent.index.embeddings as embeddings


def _manifest_document() -> dict[str, object]:
    return json.loads(embeddings._MANIFEST_PATH.read_text(encoding="utf-8"))


def test_trusted_manifest_loader_accepts_the_committed_package_data() -> None:
    manifest = embeddings._load_trusted_manifest(embeddings._MANIFEST_PATH)

    assert manifest["schema_version"] == 1
    assert manifest["model_name"] == embeddings.BGE_M3_MODEL
    assert manifest["revision"] == embeddings.BGE_M3_REVISION
    assert len(manifest["files"]) == 10


def test_trusted_manifest_loader_rejects_altered_canonical_json(tmp_path: Path) -> None:
    manifest = _manifest_document()
    manifest["files"][0]["size"] += 1  # type: ignore[index]
    altered = tmp_path / "bge_m3_artifact_manifest.json"
    altered.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="trusted manifest digest"):
        embeddings._load_trusted_manifest(altered)


def test_trusted_manifest_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    original = embeddings._MANIFEST_PATH.read_text(encoding="utf-8")
    duplicate = original.replace(
        '"schema_version":1',
        '"schema_version":1,"schema_version":1',
        1,
    )
    altered = tmp_path / "bge_m3_artifact_manifest.json"
    altered.write_text(duplicate, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        embeddings._load_trusted_manifest(altered)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("model_name", "BAAI/other", "model_name"),
        ("revision", "0" * 40, "revision"),
    ],
)
def test_manifest_schema_rejects_wrong_identity(
    field: str, value: object, message: str
) -> None:
    manifest = _manifest_document()
    manifest[field] = value

    with pytest.raises(ValueError, match=message):
        embeddings._validate_manifest_schema(manifest)


def test_manifest_schema_requires_the_exact_artifact_count() -> None:
    manifest = _manifest_document()
    manifest["files"].pop()  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="exactly 10"):
        embeddings._validate_manifest_schema(manifest)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "/absolute/config.json",
        "../config.json",
        "nested/../config.json",
        "nested//config.json",
        "nested/./config.json",
        "nested\\config.json",
        "",
        "nested/\x00config.json",
    ],
)
def test_manifest_schema_rejects_unsafe_posix_paths(unsafe_path: str) -> None:
    manifest = _manifest_document()
    manifest["files"][0]["path"] = unsafe_path  # type: ignore[index]

    with pytest.raises(ValueError, match="safe relative POSIX path"):
        embeddings._validate_manifest_schema(manifest)


def test_manifest_schema_rejects_duplicate_artifact_paths() -> None:
    manifest = _manifest_document()
    first_path = manifest["files"][0]["path"]  # type: ignore[index]
    manifest["files"][1]["path"] = first_path  # type: ignore[index]

    with pytest.raises(ValueError, match="unique"):
        embeddings._validate_manifest_schema(manifest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"size": True}, "size"),
        ({"size": 0}, "size"),
        ({"sha256": "bad"}, "sha256"),
        ({"blob_id": "bad"}, "blob_id"),
        ({"unexpected": "field"}, "fields"),
    ],
)
def test_manifest_schema_rejects_invalid_artifact_fields(
    mutation: dict[str, object], message: str
) -> None:
    manifest = _manifest_document()
    manifest["files"][0].update(mutation)  # type: ignore[index]

    with pytest.raises(ValueError, match=message):
        embeddings._validate_manifest_schema(manifest)


@pytest.mark.parametrize("lineage", ["both", "neither"])
def test_manifest_schema_requires_exactly_one_blob_or_lfs_identity(lineage: str) -> None:
    manifest = _manifest_document()
    artifact = manifest["files"][0]  # type: ignore[index]
    if lineage == "both":
        artifact["lfs_sha256"] = "1" * 64
    else:
        artifact.pop("blob_id")

    with pytest.raises(ValueError, match="exactly one"):
        embeddings._validate_manifest_schema(manifest)


def test_manifest_failure_precedes_snapshot_download_and_model_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest_document()
    manifest["files"][0]["size"] += 1  # type: ignore[index]
    altered = tmp_path / "bge_m3_artifact_manifest.json"
    altered.write_text(json.dumps(manifest), encoding="utf-8")
    called = {"download": False, "model": False}

    def forbidden_download(*args: object, **kwargs: object) -> str:
        called["download"] = True
        return "/untrusted/snapshot"

    def forbidden_model(*args: object, **kwargs: object) -> object:
        called["model"] = True
        return object()

    import huggingface_hub
    import sentence_transformers

    monkeypatch.setattr(embeddings, "_MANIFEST_PATH", altered)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden_download)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", forbidden_model)

    with pytest.raises(ValueError, match="trusted manifest digest"):
        embeddings.BgeEmbeddingManager(local_files_only=True).embed_query("HS 850440")

    assert called == {"download": False, "model": False}


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
