from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

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
    fingerprint = {"config_hash": "0" * 64, "parser_backends": backend.model_dump(mode="json"), "metadata_schema_version": "task7-source-metadata-v2", "sources": [], "documents": [], "chunks": [], "quarantined": []}
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


def test_snapshots_require_exact_canonical_json_payload() -> None:
    from pydantic import ValidationError
    from trade_agent.data.manifest import ChunkSnapshot, DocumentSnapshot, canonical_json
    from trade_agent.schemas.source import DocumentRecord, FileType, SourceType, content_sha256

    document = DocumentRecord(
        document_id="doc-example",
        document_identity="example",
        source_id="source-example",
        file_type=FileType.HTML,
        source_type=SourceType.OFFICIAL_WEBSITE,
        title="Example",
        language="en",
        content="Evidence",
        content_hash=content_sha256("Evidence"),
        fetched_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
        is_synthetic=True,
        units=[{"text": "Evidence", "locator": {"section": "Overview", "raw": {"claim": "C1"}}}],
        source_url="https://example.com/source",
    )
    snapshot = DocumentSnapshot.freeze(document)
    assert snapshot.payload == canonical_json(document.model_dump(mode="json"))
    payload = json.loads(snapshot.payload)
    payload["title"] = "Tampered"
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
    with pytest.raises(ValidationError):
        DocumentSnapshot.model_validate({"document_id": snapshot.document_id, "payload": payload_text})
    with pytest.raises(ValidationError):
        snapshot.model_copy(update={"payload": json.dumps(json.loads(snapshot.payload), ensure_ascii=False)})


def test_chunk_snapshot_rejects_reordered_json_even_when_fingerprint_fields_match() -> None:
    from pydantic import ValidationError
    from trade_agent.data.manifest import ChunkSnapshot
    from trade_agent.schemas.source import ChunkMetadata, ChunkRecord, FileType, SourceLocator, SourceType, content_sha256

    metadata = ChunkMetadata(
        chunk_id="chunk-example",
        chunk_index=0,
        document_id="doc-example",
        file_type=FileType.HTML,
        source_type=SourceType.OFFICIAL_WEBSITE,
        source_weight=0.5,
        ingested_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
        source_locator=SourceLocator(section="Overview"),
        content_hash=content_sha256("Evidence"),
        parent_document_hash="a" * 64,
        language="en",
        is_synthetic=True,
    )
    snapshot = ChunkSnapshot.freeze(ChunkRecord(content="Evidence", metadata=metadata))
    payload = json.loads(snapshot.payload)
    reordered = "{" + ",".join(f"{json.dumps(key)}:{json.dumps(payload[key], ensure_ascii=False)}" for key in reversed(payload)) + "}"
    with pytest.raises(ValidationError):
        ChunkSnapshot.model_validate({"chunk_id": snapshot.chunk_id, "content": snapshot.content, "payload": reordered})


def test_build_manifest_rejects_chunk_locator_not_in_document_units(tmp_path: Path) -> None:
    from pydantic import ValidationError
    from trade_agent.data.manifest import BuildManifest
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog

    manifest = IngestionPipeline().run(SourceCatalog.from_yaml(_catalog(tmp_path)), tmp_path / "build.json")
    payload = manifest.model_dump(mode="json")
    snapshot_payload = json.loads(payload["chunks"][0]["payload"])
    snapshot_payload["metadata"]["source_locator"]["section"] = "Other"
    from trade_agent.data.manifest import canonical_json
    payload["chunks"][0]["payload"] = canonical_json(snapshot_payload)
    with pytest.raises(ValidationError):
        BuildManifest.model_validate(payload)


def test_parser_backend_tuples_are_strict_and_quarantine_copy_revalidates() -> None:
    from pydantic import ValidationError
    from trade_agent.data.manifest import ParserBackends
    from trade_agent.data.quarantine import QuarantineRecord

    with pytest.raises(ValidationError):
        ParserBackends(
            document_router_version="router",
            chunk_router_version="chunker",
            max_tokens=10,
            overlap_tokens=1,
            mineru_statuses=("available", "available"),
            degraded_components=(),
        )
    quarantine = QuarantineRecord(
        error_code="bad",
        source_path="a/file.html",
        parser="router",
        diagnostic="safe",
    )
    with pytest.raises(ValidationError):
        quarantine.model_copy(update={"diagnostic": "token=secret"})


def test_sanitize_diagnostic_redacts_posix_and_windows_host_paths_without_url_damage() -> None:
    from trade_agent.data.quarantine import QuarantineRecord, sanitize_diagnostic

    diagnostic = (
        "failed /Users/alice/project/input.pdf and /home/alice/input.pdf "
        "from /private/var/folders/aa/bb/input.pdf C:\\Users\\alice\\input.pdf "
        "url=https://safe.example/path"
    )
    sanitized = sanitize_diagnostic(diagnostic)
    assert "/Users/" not in sanitized
    assert "/home/" not in sanitized
    assert "/private/var/" not in sanitized
    assert "C:\\Users\\" not in sanitized
    assert "https://safe.example/path" in sanitized
    assert QuarantineRecord(
        error_code="bad", source_path="input.pdf", parser="router", diagnostic=sanitized
    )
