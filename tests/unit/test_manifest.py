from __future__ import annotations

import json
from pathlib import Path

import pytest


def _catalog(tmp_path: Path, *, include_binary: bool = False) -> Path:
    root = tmp_path / "corpus"
    (root / "manifests").mkdir(parents=True)
    (root / "site.html").write_text("<h1>Example exporter</h1><p>Verified trade evidence.</p>", encoding="utf-8")
    records = [
        {
            "path": "site.html",
            "content_hash": __import__("hashlib").sha256((root / "site.html").read_bytes()).hexdigest(),
            "expected_entity": "Example exporter",
            "fact_type": "company_status",
            "file_type": "html",
            "source_type": "official_website",
            "ingested_at": "2026-08-30T00:00:00+00:00",
            "publish_time": "2026-08-30T00:00:00+00:00",
            "valid_from": "2026-08-30T00:00:00+00:00",
            "is_synthetic": True,
            "locator": {"section": "overview"},
            "reference_claim_ids": ["CLAIM-1"],
        }
    ]
    paths = ["site.html"]
    if include_binary:
        (root / "unsupported.bin").write_bytes(b"\x00\x01not-a-source")
        paths.append("unsupported.bin")
    (root / "manifests" / "corpus_manifest.json").write_text(
        json.dumps({"records": records}, sort_keys=True), encoding="utf-8"
    )
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        "\n".join(
            [
                "catalog_version: 1",
                f"root: {root.as_posix()}",
                "synthetic_notice: test-only",
                "sources:",
                "  - source_type: official_website",
                "    file_types: [html]",
                f"    paths: [{', '.join(paths)}]",
                "    url_pattern: https://company-*.example",
            ]
        ),
        encoding="utf-8",
    )
    return catalog


def test_canonical_hash_is_order_independent_and_manifest_round_trips() -> None:
    from trade_agent.data.manifest import BuildManifest, ParserBackends, canonical_hash

    assert canonical_hash({"b": [2, 1], "a": "é"}) == canonical_hash({"a": "é", "b": [2, 1]})
    backend = ParserBackends(document_router_version="router", chunk_router_version="chunker", max_tokens=1, overlap_tokens=0, mineru_statuses=("not_attempted",), degraded_components=())
    fingerprint = {"config_hash": "0" * 64, "parser_backends": backend.model_dump(mode="json"), "metadata_schema_version": "task6-source-metadata-v1", "sources": [], "documents": [], "chunks": [], "quarantined": []}
    manifest = BuildManifest.model_validate({
        "build_id": "build_" + canonical_hash(fingerprint)[:32],
        "config_hash": "0" * 64,
        "sources": [], "documents": [], "chunks": [], "quarantined": [],
        "counts_by_source_type": [], "counts_by_file_type": [],
        "parser_backends": backend,
        "metadata_complete": False,
        "fingerprint": canonical_hash(fingerprint),
    })
    assert BuildManifest.model_validate_json(manifest.model_dump_json()) == manifest
    with pytest.raises(Exception):
        manifest.build_id = "changed"  # type: ignore[misc]


def test_quarantine_diagnostic_redacts_cookie_session_and_signature_assignments() -> None:
    from trade_agent.data.quarantine import sanitize_diagnostic

    diagnostic = sanitize_diagnostic("cookie=abc session_id=def signature=ghi sig=jkl", source_path="relative/file.txt")
    assert "abc" not in diagnostic and "def" not in diagnostic and "ghi" not in diagnostic and "jkl" not in diagnostic
    assert diagnostic.count("[REDACTED]") == 4
