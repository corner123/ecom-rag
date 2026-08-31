from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def _catalog(tmp_path: Path, *, include_binary: bool = False) -> Path:
    root = tmp_path / "corpus"
    (root / "manifests").mkdir(parents=True)
    (root / "site.html").write_text("<h1>Example exporter</h1><p>Verified trade evidence.</p>", encoding="utf-8")
    records = [{
        "path": "site.html", "content_hash": hashlib.sha256((root / "site.html").read_bytes()).hexdigest(),
        "expected_entity": "Example exporter", "fact_type": "company_status", "file_type": "html",
        "source_type": "official_website", "ingested_at": "2026-08-30T00:00:00+00:00",
        "publish_time": "2026-08-30T00:00:00+00:00", "valid_from": "2026-08-30T00:00:00+00:00",
        "is_synthetic": True, "locator": {"section": "overview"}, "reference_claim_ids": ["CLAIM-1"],
    }]
    paths = ["site.html"]
    if include_binary:
        (root / "unsupported.bin").write_bytes(b"\x00\x01not-a-source")
        paths.append("unsupported.bin")
    (root / "manifests" / "corpus_manifest.json").write_text(json.dumps({"records": records}, sort_keys=True), encoding="utf-8")
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text("\n".join([
        "catalog_version: 1", f"root: {root.as_posix()}", "synthetic_notice: test-only", "sources:",
        "  - source_type: official_website", "    file_types: [html]", f"    paths: [{', '.join(paths)}]",
        "    url_pattern: https://company-*.example",
    ]), encoding="utf-8")
    return catalog


def test_same_inputs_produce_same_build_id_and_chunk_ids(tmp_path: Path) -> None:
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog

    catalog = SourceCatalog.from_yaml(_catalog(tmp_path))
    pipeline = IngestionPipeline()
    first = pipeline.run(catalog, tmp_path / "one.json")
    second = pipeline.run(catalog, tmp_path / "two.json")

    assert first.build_id == second.build_id
    assert [chunk.chunk_id for chunk in first.chunks] == [chunk.chunk_id for chunk in second.chunks]
    assert first.chunks[0].restore().content == first.chunks[0].content
    assert (tmp_path / "one.json").read_bytes() == (tmp_path / "two.json").read_bytes()
    assert first.metadata_complete is True


def test_unsupported_file_is_counted_not_dropped(tmp_path: Path) -> None:
    from trade_agent.data.manifest import BuildManifest
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog

    catalog = SourceCatalog.from_yaml(_catalog(tmp_path, include_binary=True))
    result = IngestionPipeline().run(catalog, tmp_path / "build.json")

    assert result.quarantined[0].error_code == "unsupported_file_type"
    assert result.file_type_counts["unsupported"] == 1
    assert BuildManifest.model_validate_json(result.model_dump_json()) == result


def test_tampered_manifest_hash_is_quarantined_and_output_refuses_symlink(tmp_path: Path) -> None:
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog

    catalog_path = _catalog(tmp_path)
    catalog = SourceCatalog.from_yaml(catalog_path)
    manifest_path = catalog.corpus_root / "manifests" / "corpus_manifest.json"
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8").replace('"content_hash": "', '"content_hash": "f'), encoding="utf-8")
    output = tmp_path / "linked.json"
    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text("do not replace", encoding="utf-8")
    output.symlink_to(sentinel)

    result = IngestionPipeline().run(SourceCatalog.from_yaml(catalog_path), tmp_path / "build.json")
    assert result.quarantined[0].error_code == "manifest_hash_mismatch"

    try:
        IngestionPipeline().run(catalog, output)
    except ValueError as exc:
        assert "symlink" in str(exc)
    else:  # pragma: no cover - documents the non-negotiable safety boundary
        raise AssertionError("symlink output must be rejected")
    assert sentinel.read_text(encoding="utf-8") == "do not replace"


def test_bad_catalog_paths_are_individually_quarantined(tmp_path: Path) -> None:
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog

    catalog_path = _catalog(tmp_path)
    catalog_path.write_text(catalog_path.read_text(encoding="utf-8") + "\n  - source_type: official_website\n    file_types: [html]\n    paths: [../outside.html, missing-*.html]\n    url_pattern: https://company-*.example\n", encoding="utf-8")
    result = IngestionPipeline().run(SourceCatalog.from_yaml(catalog_path), tmp_path / "build.json")
    assert {item.error_code for item in result.quarantined} >= {"unsafe_catalog_path", "missing_catalog_path"}
    assert len(result.documents) == 1


def test_demo_cli_builds_all_frozen_records(tmp_path: Path) -> None:
    output = tmp_path / "demo.json"
    result = subprocess.run([sys.executable, "-m", "scripts.ingest_trade_sources", "--catalog", "data/sources/trade_intel_demo.yaml", "--output", str(output)], check=True, capture_output=True, text=True)
    summary = json.loads(result.stdout)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert len(payload["sources"]) == 74 and len(payload["documents"]) == 116 and len(payload["chunks"]) == 120
    assert payload["quarantined"][0]["source_path"] == "pdf/scanned-regulator-notice.pdf"
    scanned = next(item for item in payload["sources"] if item["path"] == "pdf/scanned-regulator-notice.pdf")
    assert scanned["parser_backend"] == "pymupdf" and scanned["degraded"] is True
    assert payload["parser_backends"]["mineru_statuses"] == ["unavailable"]
    assert "mineru_unavailable" in payload["parser_backends"]["degraded_components"]
    reloaded = __import__("trade_agent.data.manifest", fromlist=["BuildManifest"]).BuildManifest.model_validate_json(output.read_text(encoding="utf-8"))
    assert len(reloaded.documents) == 116 and len(reloaded.chunks) == 120
    serialized = output.read_text(encoding="utf-8")
    assert "/Users/" not in serialized and str(Path.cwd().resolve()) not in serialized


def test_catalog_rule_and_field_order_is_semantic_invariant(tmp_path: Path) -> None:
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog

    catalog_path = _catalog(tmp_path)
    first_catalog = SourceCatalog.from_yaml(catalog_path)
    second = first_catalog.corpus_root / "site2.html"
    second.write_text("<h1>Second exporter</h1><p>More evidence.</p>", encoding="utf-8")
    third = first_catalog.corpus_root / "site3.html"
    third.write_text("<h1>Third exporter</h1><p>More evidence.</p>", encoding="utf-8")
    manifest_path = first_catalog.corpus_root / "manifests" / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"].append({
        **manifest["records"][0],
        "path": "site2.html",
        "content_hash": hashlib.sha256(second.read_bytes()).hexdigest(),
        "expected_entity": "Second exporter",
        "reference_claim_ids": ["CLAIM-2"],
    })
    manifest["records"].append({
        **manifest["records"][0],
        "path": "site3.html",
        "content_hash": hashlib.sha256(third.read_bytes()).hexdigest(),
        "expected_entity": "Third exporter",
        "reference_claim_ids": ["CLAIM-3"],
    })
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    root = first_catalog.corpus_root.as_posix()
    raw = "\n".join([
        "catalog_version: 1", f"root: {root}", "synthetic_notice: test-only", "sources:",
        "  - source_type: official_website", "    file_types: [html, markdown]", "    paths: [site.html, site2.html]", "    url_pattern: https://company-*.example",
        "  - source_type: official_website", "    file_types: [markdown, html]", "    paths: [site3.html]", "    url_pattern: https://company-*.example",
    ])
    first_path = tmp_path / "catalog-first.yaml"
    first_path.write_text(raw, encoding="utf-8")
    first_catalog = SourceCatalog.from_yaml(first_path)
    reordered = "\n".join([
        "catalog_version: 1", f"root: {root}", "synthetic_notice: test-only", "sources:",
        "  - source_type: official_website", "    file_types: [markdown, html]", "    paths: [site3.html]", "    url_pattern: https://company-*.example",
        "  - source_type: official_website", "    file_types: [markdown, html]", "    paths: [site2.html, site.html]", "    url_pattern: https://company-*.example",
    ])
    second_path = tmp_path / "catalog-reordered.yaml"
    second_path.write_text(reordered, encoding="utf-8")
    second_catalog = SourceCatalog.from_yaml(second_path)
    first = IngestionPipeline().run(first_catalog, tmp_path / "first.json")
    second = IngestionPipeline().run(second_catalog, tmp_path / "second.json")
    assert first.config_hash == second.config_hash
    assert first.build_id == second.build_id
    assert first.chunk_ids == second.chunk_ids


def test_copied_corpus_under_new_root_has_identical_build_and_chunks(tmp_path: Path) -> None:
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog

    first_catalog_path = _catalog(tmp_path / "first")
    first_catalog = SourceCatalog.from_yaml(first_catalog_path)
    copied_root = tmp_path / "second" / "corpus"
    shutil.copytree(first_catalog.corpus_root, copied_root)
    second_catalog_path = tmp_path / "second" / "catalog.yaml"
    second_catalog_path.write_text(first_catalog_path.read_text(encoding="utf-8").replace(str(first_catalog.corpus_root), str(copied_root)), encoding="utf-8")
    first = IngestionPipeline().run(first_catalog, tmp_path / "first-build.json")
    second = IngestionPipeline().run(SourceCatalog.from_yaml(second_catalog_path), tmp_path / "second-build.json")
    assert first.config_hash == second.config_hash
    assert first.build_id == second.build_id
    assert first.chunk_ids == second.chunk_ids


def test_output_symlinked_ancestor_is_rejected_without_external_write(tmp_path: Path) -> None:
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog

    catalog = SourceCatalog.from_yaml(_catalog(tmp_path))
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        IngestionPipeline().run(catalog, linked / "build.json")
    assert not (outside / "build.json").exists()


def test_persisted_manifest_rejects_snapshot_count_and_build_id_tampering(tmp_path: Path) -> None:
    from pydantic import ValidationError
    from trade_agent.data.manifest import BuildManifest
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog

    result = IngestionPipeline().run(SourceCatalog.from_yaml(_catalog(tmp_path)), tmp_path / "build.json")
    payload = result.model_dump(mode="json")
    payload["chunks"][0]["chunk_id"] = "wrong"
    with pytest.raises(ValidationError):
        BuildManifest.model_validate(payload)
    payload = result.model_dump(mode="json")
    payload["counts_by_file_type"][0]["count"] = 99
    with pytest.raises(ValidationError):
        BuildManifest.model_validate(payload)
    payload = result.model_dump(mode="json")
    payload["build_id"] = "build_" + "f" * 32
    with pytest.raises(ValidationError):
        BuildManifest.model_validate(payload)
    payload = result.model_dump(mode="json")
    payload["parser_backends"]["document_router_version"] = "tampered"
    with pytest.raises(ValidationError):
        BuildManifest.model_validate(payload)
    payload = result.model_dump(mode="json")
    payload["metadata_schema_version"] = "tampered"
    with pytest.raises(ValidationError):
        BuildManifest.model_validate(payload)
    payload = result.model_dump(mode="json")
    payload["counts_by_file_type"][0]["count"] = "1"
    with pytest.raises(ValidationError):
        BuildManifest.model_validate(payload)
    assert len(result.fingerprint) == 64


def test_oversized_source_is_quarantined_without_aborting_batch(tmp_path: Path) -> None:
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog
    from trade_agent.data.router import DocumentRouter

    result = IngestionPipeline(document_router=DocumentRouter(max_bytes=1)).run(SourceCatalog.from_yaml(_catalog(tmp_path)), tmp_path / "build.json")
    assert result.quarantined[0].error_code == "input_too_large"
    assert result.sources[0].status == "quarantined"
    assert result.metadata_complete is False


def test_dangling_symlink_and_replace_failure_preserve_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog

    catalog = SourceCatalog.from_yaml(_catalog(tmp_path))
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing.json")
    with pytest.raises(ValueError):
        IngestionPipeline().run(catalog, dangling)

    output = tmp_path / "existing.json"
    output.write_text("original", encoding="utf-8")
    monkeypatch.setattr("trade_agent.data.pipeline.os.replace", lambda *_: (_ for _ in ()).throw(OSError("injected")))
    with pytest.raises(OSError):
        IngestionPipeline().run(catalog, output)
    assert output.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".existing.json.*.tmp"))


def test_frozen_manifest_direct_symlink_is_rejected_before_json_parse(tmp_path: Path) -> None:
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog

    catalog_path = _catalog(tmp_path)
    root = SourceCatalog.from_yaml(catalog_path).corpus_root
    target = tmp_path / "external-manifest.json"
    target.write_text("not-json", encoding="utf-8")
    manifest = root / "manifests" / "corpus_manifest.json"
    manifest.unlink()
    manifest.symlink_to(target)
    with pytest.raises(ValueError, match="frozen corpus manifest") as exc:
        IngestionPipeline().run(SourceCatalog.from_yaml(catalog_path), tmp_path / "build.json")
    assert str(target) not in str(exc.value)


def test_frozen_manifest_symlinked_manifests_ancestor_is_rejected(tmp_path: Path) -> None:
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog

    catalog_path = _catalog(tmp_path)
    root = SourceCatalog.from_yaml(catalog_path).corpus_root
    external = tmp_path / "external-manifests"
    external.mkdir()
    (external / "corpus_manifest.json").write_text("not-json", encoding="utf-8")
    real_manifests = root / "manifests"
    (real_manifests / "corpus_manifest.json").unlink()
    real_manifests.rmdir()
    real_manifests.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="frozen corpus manifest") as exc:
        IngestionPipeline().run(SourceCatalog.from_yaml(catalog_path), tmp_path / "build.json")
    assert str(external) not in str(exc.value)


def test_frozen_manifest_is_bounded_before_parsing(tmp_path: Path) -> None:
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog
    from trade_agent.data.router import DocumentRouter

    catalog_path = _catalog(tmp_path)
    root = SourceCatalog.from_yaml(catalog_path).corpus_root
    (root / "manifests" / "corpus_manifest.json").write_bytes(b"{" + b" " * 100 + b"}")
    pipeline = IngestionPipeline(document_router=DocumentRouter(max_bytes=32))
    with pytest.raises(ValueError, match="frozen corpus manifest"):
        pipeline._frozen_records(root, maximum=32)


def test_frozen_source_file_type_mismatch_is_quarantined_before_router(tmp_path: Path) -> None:
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog

    catalog_path = _catalog(tmp_path)
    root = SourceCatalog.from_yaml(catalog_path).corpus_root
    manifest_path = root / "manifests" / "corpus_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["records"][0]["file_type"] = "json"
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    result = IngestionPipeline().run(SourceCatalog.from_yaml(catalog_path), tmp_path / "build.json")
    assert result.quarantined[0].error_code == "manifest_record_mismatch"
    assert result.documents == ()


def test_pipeline_preserves_deep_relative_quarantine_path_and_scan_backend_state(tmp_path: Path) -> None:
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog

    root = tmp_path / "corpus"
    (root / "manifests").mkdir(parents=True)
    deep = root / "a" / "b" / "x.html"
    deep.parent.mkdir(parents=True)
    deep.write_text("<html><body><h1>broken", encoding="utf-8")
    digest = hashlib.sha256(deep.read_bytes()).hexdigest()
    (root / "manifests" / "corpus_manifest.json").write_text(json.dumps({"records": [{
        "path": "a/b/x.html", "content_hash": digest, "expected_entity": "x", "fact_type": "company_status",
        "file_type": "html", "source_type": "official_website", "ingested_at": "2026-08-30T00:00:00+00:00",
        "publish_time": "2026-08-30T00:00:00+00:00", "valid_from": "2026-08-30T00:00:00+00:00", "is_synthetic": True,
        "locator": {"section": "overview"}, "reference_claim_ids": ["C1"]
    }]}, sort_keys=True), encoding="utf-8")
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text("\n".join([
        "catalog_version: 1", f"root: {root}", "synthetic_notice: test-only", "sources:",
        "  - source_type: official_website", "    file_types: [html]", "    paths: [a/b/x.html]", "    url_pattern: https://x.example",
    ]), encoding="utf-8")
    result = IngestionPipeline().run(SourceCatalog.from_yaml(catalog), tmp_path / "build.json")
    assert result.quarantined and result.quarantined[0].source_path == "a/b/x.html"


def test_chunk_failure_is_quarantined_and_reloadable(tmp_path: Path) -> None:
    from trade_agent.data.chunkers import ChunkRouter
    from trade_agent.data.manifest import BuildManifest
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog

    class FailingChunkRouter(ChunkRouter):
        def chunk(self, document):
            raise RuntimeError("chunker injected failure")

    output = tmp_path / "build.json"
    result = IngestionPipeline(chunk_router=FailingChunkRouter()).run(SourceCatalog.from_yaml(_catalog(tmp_path)), output)
    assert result.sources[0].status == "quarantined"
    assert any(item.error_code == "chunk_failed" for item in result.quarantined)
    assert BuildManifest.model_validate_json(output.read_text(encoding="utf-8")) == result
