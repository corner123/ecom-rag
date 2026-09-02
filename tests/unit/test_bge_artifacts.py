from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import errno
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import trade_agent.index.embeddings as embeddings
import trade_agent.index.provenance as provenance


def _manifest_document() -> dict[str, object]:
    return json.loads(embeddings._MANIFEST_PATH.read_text(encoding="utf-8"))


def test_trusted_manifest_loader_accepts_the_committed_package_data() -> None:
    manifest = embeddings._load_trusted_manifest(embeddings._MANIFEST_PATH)

    assert manifest.schema_version == 1
    assert manifest.model_name == embeddings.BGE_M3_MODEL
    assert manifest.revision == embeddings.BGE_M3_REVISION
    assert len(manifest.files) == 10


def test_loaded_manifest_is_deeply_immutable_and_copies_lose_trust() -> None:
    manifest = embeddings._load_trusted_manifest(embeddings._MANIFEST_PATH)

    with pytest.raises(FrozenInstanceError):
        manifest.revision = "0" * 40  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        manifest.files[0].size += 1  # type: ignore[misc]
    copied = replace(manifest)
    with pytest.raises(ValueError, match="live trusted manifest"):
        embeddings._require_trusted_manifest(copied)


def test_loader_returns_fresh_equivalent_trusted_manifest_values() -> None:
    first = embeddings._load_trusted_manifest(embeddings._MANIFEST_PATH)
    second = embeddings._load_trusted_manifest(embeddings._MANIFEST_PATH)

    assert first == second
    assert first is not second
    embeddings._require_trusted_manifest(first)
    embeddings._require_trusted_manifest(second)


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


def test_load_encoder_threads_one_local_manifest_through_every_runtime_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = embeddings._load_trusted_manifest(embeddings._MANIFEST_PATH)
    snapshot = tmp_path / "models--BAAI--bge-m3" / "snapshots" / embeddings.BGE_M3_REVISION
    snapshot.mkdir(parents=True)
    seen: dict[str, object] = {}
    encoder = object()
    runtime_path = tmp_path / "verified-runtime-view"
    runtime_path.mkdir()

    class FakeRuntimeView:
        path = runtime_path

        def cleanup(self) -> None:
            seen["cleaned"] = True

    runtime_view = FakeRuntimeView()
    receipt = embeddings.VerifiedEmbeddingSnapshot(
        model_name=embeddings.BGE_M3_MODEL,
        requested_revision=embeddings.BGE_M3_REVISION,
        resolved_revision=embeddings.BGE_M3_REVISION,
        trusted_bytes=1,
        artifact_manifest_sha256=embeddings.ARTIFACT_MANIFEST_SHA256,
        artifact_count=len(manifest.files),
    )

    class FakeApi:
        pass

    def fake_download(**kwargs: object) -> str:
        seen["allow_patterns"] = kwargs["allow_patterns"]
        return str(snapshot)

    def fake_metadata(api: object, trusted_manifest: object) -> None:
        seen["metadata_manifest"] = trusted_manifest

    def fake_snapshot(path: Path, trusted_manifest: object) -> object:
        seen["snapshot_manifest"] = trusted_manifest
        return 1, runtime_view

    def fake_loaded_snapshot(
        path: Path,
        trusted_manifest: object,
        loaded_runtime_view: object,
        trusted_bytes: int,
    ) -> object:
        seen["post_load_manifest"] = trusted_manifest
        seen["post_load_runtime_view"] = loaded_runtime_view
        seen["post_load_trusted_bytes"] = trusted_bytes
        return receipt

    def fake_model(model_path: str, **kwargs: object) -> object:
        seen["model_path"] = Path(model_path)
        return encoder

    import huggingface_hub
    import sentence_transformers

    monkeypatch.setattr(embeddings, "_load_trusted_manifest", lambda _: manifest)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fake_model)
    manager = embeddings.BgeEmbeddingManager(local_files_only=False)
    monkeypatch.setattr(manager, "_verify_official_metadata", fake_metadata)
    monkeypatch.setattr(manager, "_verify_snapshot", fake_snapshot)
    monkeypatch.setattr(manager, "_verify_loaded_snapshot", fake_loaded_snapshot)

    assert manager._load_encoder() is encoder
    assert seen["allow_patterns"] == manifest.paths
    assert seen["metadata_manifest"] is manifest
    assert seen["snapshot_manifest"] is manifest
    assert seen["post_load_manifest"] is manifest
    assert seen["post_load_runtime_view"] is runtime_view
    assert seen["post_load_trusted_bytes"] == 1
    assert seen["model_path"] == runtime_path
    assert manager._runtime_view is runtime_view


def test_model_construction_failure_cleans_the_verified_runtime_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = embeddings._load_trusted_manifest(embeddings._MANIFEST_PATH)
    snapshot = tmp_path / "models--BAAI--bge-m3" / "snapshots" / embeddings.BGE_M3_REVISION
    snapshot.mkdir(parents=True)
    runtime_path = tmp_path / "verified-runtime-view"
    runtime_path.mkdir()
    state = {"cleaned": False}
    class FakeRuntimeView:
        path = runtime_path

        def cleanup(self) -> None:
            state["cleaned"] = True

    import huggingface_hub
    import sentence_transformers

    def fail_model_construction(*args: object, **kwargs: object) -> object:
        raise RuntimeError("load failed")

    monkeypatch.setattr(embeddings, "_load_trusted_manifest", lambda _: manifest)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_: str(snapshot))
    monkeypatch.setattr(
        sentence_transformers,
        "SentenceTransformer",
        fail_model_construction,
    )
    manager = embeddings.BgeEmbeddingManager(local_files_only=True)
    monkeypatch.setattr(
        manager,
        "_verify_snapshot",
        lambda path, trusted_manifest: (1, FakeRuntimeView()),
    )

    with pytest.raises(RuntimeError, match="load failed"):
        manager._load_encoder()

    assert state["cleaned"] is True
    assert manager._verified_snapshot is None
    assert manager._runtime_view is None


def test_model_construction_in_place_mutation_is_rejected_before_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, artifact, expected = _snapshot_with_one_trusted_file(tmp_path)
    runtime_view = provenance._materialize_runtime_view(snapshot, (expected,))
    trusted_bytes = provenance._verify_runtime_view_contents(
        runtime_view.path,
        (expected,),
    )
    verified_blob = artifact.resolve(strict=True)

    def mutate_during_model_load(*args: object, **kwargs: object) -> object:
        assert manager._verified_snapshot is None
        assert manager._encoder is None
        with pytest.raises(RuntimeError, match="not available"):
            _ = manager.contract
        verified_blob.write_bytes(b"attacker changed model bytes")
        return object()

    def verify_after_model_load(*args: object, **kwargs: object) -> object:
        provenance._verify_runtime_view_contents(runtime_view.path, (expected,))
        pytest.fail("a receipt must not be issued for the mutated runtime view")

    import huggingface_hub
    import sentence_transformers

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **_: str(snapshot),
    )
    monkeypatch.setattr(
        sentence_transformers,
        "SentenceTransformer",
        mutate_during_model_load,
    )
    manager = embeddings.BgeEmbeddingManager(local_files_only=True)
    monkeypatch.setattr(
        manager,
        "_verify_snapshot",
        lambda path, trusted_manifest: (trusted_bytes, runtime_view),
    )
    monkeypatch.setattr(
        manager,
        "_verify_loaded_snapshot",
        verify_after_model_load,
        raising=False,
    )

    with pytest.raises(ValueError, match="runtime view checksum"):
        manager._load_encoder()

    assert not runtime_view.path.exists()
    assert manager._verified_snapshot is None
    assert manager._runtime_view is None
    assert manager._encoder is None
    with pytest.raises(RuntimeError, match="not available"):
        _ = manager.contract


def test_rebinding_public_allowlists_cannot_authorize_attacker_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted = embeddings._load_trusted_manifest(embeddings._MANIFEST_PATH)
    payload = b"attacker-controlled model bytes"
    attacker = {
        "path": "config.json",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "blob_id": "a" * 40,
    }
    snapshot = tmp_path / "models--BAAI--bge-m3" / "snapshots" / embeddings.BGE_M3_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(payload)
    called: dict[str, object] = {"model": False}

    def fake_download(**kwargs: object) -> str:
        called["allow_patterns"] = kwargs["allow_patterns"]
        return str(snapshot)

    def fake_model(*args: object, **kwargs: object) -> object:
        called["model"] = True

        class AttackerEncoder:
            def encode(self, texts: object, **options: object) -> np.ndarray:
                return np.ones((1, embeddings.BGE_M3_DIMENSION), dtype=np.float32)

        return AttackerEncoder()

    import huggingface_hub
    import sentence_transformers

    monkeypatch.setattr(embeddings, "TRUSTED_ARTIFACTS", (attacker,))
    monkeypatch.setattr(embeddings, "TRUSTED_ARTIFACT_PATHS", ("config.json",))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fake_model)

    with pytest.raises(ValueError, match="missing|required|checksum"):
        embeddings.BgeEmbeddingManager(local_files_only=True).embed_query("HS 850440")

    assert called["allow_patterns"] == trusted.paths
    assert called["model"] is False


def _snapshot_with_one_trusted_file(
    tmp_path: Path,
) -> tuple[Path, Path, embeddings.TrustedArtifact]:
    payload = b"trusted dense model artifact"
    expected = embeddings.TrustedArtifact(
        path="config.json",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        blob_id="a" * 40,
    )

    repository = tmp_path / "models--BAAI--bge-m3"
    blob = repository / "blobs" / "trusted-blob"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(payload)
    snapshot = repository / "snapshots" / embeddings.BGE_M3_REVISION
    snapshot.mkdir(parents=True)
    artifact = snapshot / "config.json"
    artifact.symlink_to(blob)
    return snapshot, artifact, expected


def test_snapshot_verification_hashes_trusted_files_and_ignores_extras(
    tmp_path: Path,
) -> None:
    snapshot, _, expected = _snapshot_with_one_trusted_file(tmp_path)
    (snapshot / "long.jpg").write_bytes(b"untrusted image asset")
    (snapshot / "onnx").mkdir()
    (snapshot / "onnx" / "model.onnx").write_bytes(b"duplicate weights")

    verified_bytes = embeddings._verify_snapshot_contents(
        snapshot,
        expected_revision=embeddings.BGE_M3_REVISION,
        artifacts=(expected,),
    )

    assert verified_bytes == expected.size


def test_runtime_view_hides_unverified_loader_alternates_from_the_model(
    tmp_path: Path,
) -> None:
    snapshot, _, expected = _snapshot_with_one_trusted_file(tmp_path)
    dangerous_extras = {
        "model.safetensors",
        "model.safetensors.index.json",
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    for relative_path in dangerous_extras:
        (snapshot / relative_path).write_bytes(b"attacker-controlled loader input")

    runtime_view = provenance._materialize_runtime_view(snapshot, (expected,))
    seen: dict[str, Path] = {}

    def fake_loader(model_path: str) -> object:
        seen["model_path"] = Path(model_path)
        return object()

    try:
        fake_loader(str(runtime_view.path))

        assert seen["model_path"] == runtime_view.path
        assert (runtime_view.path / expected.path).read_bytes() == (
            b"trusted dense model artifact"
        )
        assert not any((runtime_view.path / path).exists() for path in dangerous_extras)
        visible_files = {
            path.relative_to(runtime_view.path).as_posix()
            for path in runtime_view.path.rglob("*")
            if path.is_file()
        }
        assert visible_files == {expected.path}
    finally:
        runtime_view.cleanup()


def test_cross_filesystem_runtime_view_pins_the_verified_file_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, artifact, expected = _snapshot_with_one_trusted_file(tmp_path)
    verified_blob = artifact.resolve(strict=True)

    def cross_filesystem_link(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(provenance.os, "link", cross_filesystem_link)
    runtime_view = provenance._materialize_runtime_view(snapshot, (expected,))
    replacement = verified_blob.with_name("attacker-replacement")
    replacement.write_bytes(b"same path, different inode")
    replacement.replace(verified_blob)

    try:
        assert (runtime_view.path / expected.path).read_bytes() == (
            b"trusted dense model artifact"
        )
    finally:
        runtime_view.cleanup()


def test_directly_constructed_manifest_cannot_issue_a_snapshot_receipt(
    tmp_path: Path,
) -> None:
    snapshot, _, expected = _snapshot_with_one_trusted_file(tmp_path)
    forged = embeddings.TrustedManifest(
        schema_version=1,
        model_name=embeddings.BGE_M3_MODEL,
        revision=embeddings.BGE_M3_REVISION,
        files=(expected,),
        canonical_sha256=embeddings.ARTIFACT_MANIFEST_SHA256,
    )

    with pytest.raises(ValueError, match="live trusted manifest"):
        embeddings.BgeEmbeddingManager(local_files_only=True)._verify_snapshot(
            snapshot,
            forged,
        )


def test_snapshot_directory_name_alone_cannot_establish_provenance(
    tmp_path: Path,
) -> None:
    snapshot, artifact, expected = _snapshot_with_one_trusted_file(tmp_path)
    artifact.unlink()

    with pytest.raises(ValueError, match="missing"):
        embeddings._verify_snapshot_contents(
            snapshot,
            expected_revision=embeddings.BGE_M3_REVISION,
            artifacts=(expected,),
        )


def test_snapshot_rejects_a_non_regular_required_artifact(
    tmp_path: Path,
) -> None:
    snapshot, artifact, expected = _snapshot_with_one_trusted_file(tmp_path)
    artifact.unlink()
    artifact.mkdir()

    with pytest.raises(ValueError, match="missing"):
        embeddings._verify_snapshot_contents(
            snapshot,
            expected_revision=embeddings.BGE_M3_REVISION,
            artifacts=(expected,),
        )


def test_snapshot_rejects_a_required_symlink_outside_the_blob_store(
    tmp_path: Path,
) -> None:
    snapshot, artifact, expected = _snapshot_with_one_trusted_file(tmp_path)
    payload = artifact.read_bytes()
    artifact.unlink()
    outside = tmp_path / "outside-model-cache.bin"
    outside.write_bytes(payload)
    artifact.symlink_to(outside)

    with pytest.raises(ValueError, match="outside"):
        embeddings._verify_snapshot_contents(
            snapshot,
            expected_revision=embeddings.BGE_M3_REVISION,
            artifacts=(expected,),
        )


def test_snapshot_rejects_an_intermediate_directory_symlink_escape(
    tmp_path: Path,
) -> None:
    payload = b"trusted nested artifact"
    expected = embeddings.TrustedArtifact(
        path="nested/config.json",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        blob_id="a" * 40,
    )
    repository = tmp_path / "models--BAAI--bge-m3"
    (repository / "blobs").mkdir(parents=True)
    snapshot = repository / "snapshots" / embeddings.BGE_M3_REVISION
    snapshot.mkdir(parents=True)
    outside = tmp_path / "outside-directory"
    outside.mkdir()
    (outside / "config.json").write_bytes(payload)
    (snapshot / "nested").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="outside"):
        embeddings._verify_snapshot_contents(
            snapshot,
            expected_revision=embeddings.BGE_M3_REVISION,
            artifacts=(expected,),
        )


@pytest.mark.parametrize("corruption", ["size", "sha256"])
def test_snapshot_rejects_wrong_size_or_sha256(
    tmp_path: Path, corruption: str
) -> None:
    snapshot, _, original = _snapshot_with_one_trusted_file(tmp_path)
    if corruption == "size":
        expected = replace(original, size=original.size + 1)
    else:
        expected = replace(original, sha256="0" * 64)

    with pytest.raises(ValueError, match="checksum"):
        embeddings._verify_snapshot_contents(
            snapshot,
            expected_revision=embeddings.BGE_M3_REVISION,
            artifacts=(expected,),
        )


def _official_model_info(
    manifest: embeddings.TrustedManifest,
    **changes: object,
) -> SimpleNamespace:
    siblings = []
    for expected in manifest.files:
        lfs = (
            SimpleNamespace(sha256=expected.lfs_sha256)
            if expected.lfs_sha256 is not None
            else None
        )
        siblings.append(
            SimpleNamespace(
                rfilename=expected.path,
                size=expected.size,
                blob_id=expected.blob_id,
                lfs=lfs,
            )
        )
    values = {"sha": manifest.revision, "siblings": siblings}
    values.update(changes)
    return SimpleNamespace(**values)


class _FakeHfApi:
    def __init__(self, info: SimpleNamespace) -> None:
        self.info = info

    def model_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
        return self.info


def test_official_metadata_accepts_the_pinned_revision_and_manifest() -> None:
    manager = embeddings.BgeEmbeddingManager()
    manifest = embeddings._load_trusted_manifest(embeddings._MANIFEST_PATH)

    manager._verify_official_metadata(
        _FakeHfApi(_official_model_info(manifest)),
        manifest,
    )


def test_official_metadata_rejects_a_different_resolved_revision() -> None:
    manager = embeddings.BgeEmbeddingManager()
    manifest = embeddings._load_trusted_manifest(embeddings._MANIFEST_PATH)

    with pytest.raises(ValueError, match="resolved revision"):
        manager._verify_official_metadata(
            _FakeHfApi(_official_model_info(manifest, sha="0" * 40)),
            manifest,
        )


@pytest.mark.parametrize("corruption", ["missing", "size", "lfs_sha256", "blob_id"])
def test_official_metadata_rejects_manifest_mismatches(corruption: str) -> None:
    manifest = embeddings._load_trusted_manifest(embeddings._MANIFEST_PATH)
    info = _official_model_info(manifest)
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
        embeddings.BgeEmbeddingManager()._verify_official_metadata(
            _FakeHfApi(info),
            manifest,
        )


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
