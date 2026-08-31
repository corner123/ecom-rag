from __future__ import annotations

import hashlib
import json
from pathlib import Path
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
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog

    catalog = SourceCatalog.from_yaml(_catalog(tmp_path, include_binary=True))
    result = IngestionPipeline().run(catalog, tmp_path / "build.json")

    assert result.quarantined[0].error_code == "unsupported_file_type"
    assert result.file_type_counts["unsupported"] == 1


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
