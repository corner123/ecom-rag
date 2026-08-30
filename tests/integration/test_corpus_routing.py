"""Corpus-wide routing/chunking contract for the frozen Task 5 fixtures."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from trade_agent.data.chunkers import ChunkRouter
from trade_agent.data.router import DocumentRouter, SourceInput
from trade_agent.schemas.source import ChunkRecord, FileType, SourceType, stable_id


ROOT = Path(__file__).parents[2] / "demo" / "trade_intel_seed"
MANIFEST = json.loads((ROOT / "manifests" / "corpus_manifest.json").read_text(encoding="utf-8"))


def _ingest():
    router = DocumentRouter()
    documents = []
    for record in sorted(MANIFEST["records"], key=lambda item: item["path"]):
        source_type = SourceType(record["source_type"])
        assert source_type is not SourceType.TRADE_LEDGER
        source = SourceInput(
            path=ROOT / record["path"],
            file_type=FileType(record["file_type"]),
            source_type=source_type,
            source_id=stable_id("source", record["path"]),
            title=record.get("expected_entity", record["path"]),
            language="en",
            fetched_at=datetime.fromisoformat(record["ingested_at"]),
            is_synthetic=record["is_synthetic"],
            source_url="https://ingest.example/manifest",
            publish_time=datetime.fromisoformat(record["publish_time"]),
            valid_from=datetime.fromisoformat(record["valid_from"]),
            manifest_attributes={key: value for key, value in record.items() if key not in {"path", "file_type", "source_type", "is_synthetic"}},
        )
        documents.extend(router.load(source))
    chunker = ChunkRouter()
    chunks = [chunk for document in documents for chunk in chunker.chunk(document)]
    return router, documents, chunks


def test_all_74_manifest_records_route_to_116_documents_and_strict_chunks():
    router, documents, chunks = _ingest()
    assert len(MANIFEST["records"]) == 74
    assert len(documents) == 116
    assert len(chunks) >= len(documents)
    assert [(record.error_code, Path(record.source_path).name) for record in router.quarantines] == [
        ("SCANNED_PDF_OCR_UNAVAILABLE", "scanned-regulator-notice.pdf")
    ]
    assert all(ChunkRecord.model_validate_json(chunk.model_dump_json()) == chunk for chunk in chunks)
    assert all(chunk.metadata.source_locator.raw for chunk in chunks)
    assert not any(chunk.metadata.source_type is SourceType.TRADE_LEDGER for chunk in chunks)

    source_counts = Counter(chunk.metadata.source_type for chunk in chunks)
    assert set(source_counts) == {
        SourceType.OFFICIAL_WEBSITE,
        SourceType.B2B,
        SourceType.INDUSTRY_NEWS,
        SourceType.SOCIAL,
        SourceType.REGULATOR,
        SourceType.CUSTOMS_PROFILE,
    }


def test_corpus_boundaries_and_metadata_are_source_specific():
    _, _, chunks = _ingest()
    by_source = {source: [chunk for chunk in chunks if chunk.metadata.source_type is source] for source in SourceType}

    assert all(chunk.metadata.source_locator.section for chunk in by_source[SourceType.OFFICIAL_WEBSITE])
    assert all(chunk.metadata.source_locator.row and chunk.metadata.source_locator.raw["product_id"] for chunk in by_source[SourceType.B2B])
    assert all(chunk.metadata.source_locator.post_id and chunk.metadata.source_locator.raw["claim_id"] for chunk in by_source[SourceType.SOCIAL])
    assert all(chunk.metadata.source_locator.profile == "monthly_company_hs" for chunk in by_source[SourceType.CUSTOMS_PROFILE])
    assert all(chunk.metadata.aggregation_info["aggregation_grain"] == "company_country_hs_calendar_month" for chunk in by_source[SourceType.CUSTOMS_PROFILE])

    pdf_chunks = [chunk for chunk in chunks if chunk.metadata.file_type is FileType.PDF]
    assert pdf_chunks and all(chunk.metadata.source_locator.page >= 1 for chunk in pdf_chunks)
    assert all(chunk.metadata.publish_time is not None and chunk.metadata.valid_from is not None for chunk in chunks)
    mirror = next(chunk for chunk in by_source[SourceType.INDUSTRY_NEWS] if chunk.metadata.source_locator.raw.get("story_id") == "NEWS-013")
    assert str(mirror.metadata.source_url) == "https://syndication.example/story/013"
    assert str(mirror.metadata.canonical_url) == "https://newsroom.example/story/001"
    assert mirror.metadata.dedupe_cluster_id == "NEWS-CLUSTER-001"


def test_corpus_routing_and_chunking_are_repeat_deterministic():
    first_router, first_documents, first_chunks = _ingest()
    second_router, second_documents, second_chunks = _ingest()
    assert [document.model_dump(mode="json") for document in first_documents] == [document.model_dump(mode="json") for document in second_documents]
    assert [(chunk.metadata.chunk_id, chunk.metadata.content_hash, chunk.content) for chunk in first_chunks] == [
        (chunk.metadata.chunk_id, chunk.metadata.content_hash, chunk.content) for chunk in second_chunks
    ]
    assert [record.model_dump() for record in first_router.quarantines] == [record.model_dump() for record in second_router.quarantines]
