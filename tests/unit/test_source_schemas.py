from datetime import datetime, timezone
from math import inf, nan

import pytest
from pydantic import ValidationError

from trade_agent.schemas.source import (
    ChunkMetadata,
    ChunkRecord,
    DocumentRecord,
    FactType,
    FileType,
    SourceLocator,
    SourceRecord,
    SourceType,
    content_sha256,
    stable_id,
)


NOW = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)


def locator() -> SourceLocator:
    return SourceLocator(section="Products", page=2, block=3, raw="section=Products&page=2")


def metadata(**overrides) -> dict:
    value = {
        "chunk_id": "chk_abc123",
        "document_id": "doc_abc123",
        "file_type": FileType.HTML,
        "source_type": SourceType.OFFICIAL_WEBSITE,
        "source_weight": 0.8,
        "ingested_at": NOW,
        "source_locator": locator(),
        "content_hash": content_sha256("a chunk"),
        "parent_document_hash": content_sha256("a document"),
        "language": "en",
        "is_synthetic": True,
    }
    value.update(overrides)
    return value


def test_enums_cover_required_business_values_and_preserve_strings():
    assert {item.value for item in FileType} >= {"html", "markdown", "json", "jsonl", "pdf", "generated_profile"}
    assert {item.value for item in SourceType} >= {
        "official_website", "b2b", "industry_news", "social", "regulator", "customs_profile"
    }
    assert {item.value for item in FactType} >= {
        "trade_activity", "product_offering", "company_status", "regulation", "risk", "contact", "market_signal"
    }


def test_chunk_metadata_requires_truth_boundary_and_forbids_extra():
    with pytest.raises(ValidationError):
        ChunkMetadata(file_type="html", source_type="official_website")
    with pytest.raises(ValidationError):
        ChunkMetadata(**metadata(unexpected="nope"))
    assert ChunkMetadata(**metadata()).is_synthetic is True


def test_hashes_and_stable_ids_are_deterministic_and_collision_resistant():
    assert content_sha256("é") == content_sha256("é".encode("utf-8"))
    assert len(content_sha256("x")) == 64
    with pytest.raises(TypeError):
        content_sha256(123)  # type: ignore[arg-type]
    assert stable_id("doc", "a", "bc") != stable_id("doc", "ab", "c")
    assert stable_id("doc", "中文", "é") == stable_id("doc", "中文", "é")
    with pytest.raises(ValueError):
        stable_id("bad prefix", "x")


def test_hs_code_is_string_and_preserves_leading_zero():
    assert ChunkMetadata(**metadata(hs_code="010121")).hs_code == "010121"
    with pytest.raises(ValidationError):
        ChunkMetadata(**metadata(hs_code=10121))
    with pytest.raises(ValidationError):
        ChunkMetadata(**metadata(hs_code="10121x"))


def test_country_code_is_normalized_and_validated():
    assert ChunkMetadata(**metadata(country_code="cn")).country_code == "CN"
    with pytest.raises(ValidationError):
        ChunkMetadata(**metadata(country_code="CHN"))


def test_aware_datetimes_ranges_and_bounds_are_enforced():
    with pytest.raises(ValidationError):
        ChunkMetadata(**metadata(ingested_at=datetime(2026, 8, 30, 12, 0)))
    with pytest.raises(ValidationError):
        ChunkMetadata(**metadata(source_weight=1.1))
    with pytest.raises(ValidationError):
        ChunkMetadata(**metadata(ocr_confidence=-0.1))
    with pytest.raises(ValidationError):
        ChunkMetadata(**metadata(valid_from=NOW, valid_to=NOW.replace(day=29)))


def test_synthetic_url_boundary_and_real_public_source():
    synthetic = SourceRecord(
        source_id="src_demo",
        source_type=SourceType.OFFICIAL_WEBSITE,
        file_type=FileType.HTML,
        url="https://acme.synthetic.example/about",
        title="About",
        language="en",
        fetched_at=NOW,
        is_synthetic=True,
    )
    assert str(synthetic.url).startswith("https://acme.synthetic.example")
    with pytest.raises(ValidationError):
        SourceRecord(
            source_id="src_demo",
            source_type="official_website",
            file_type="html",
            url="https://evil.invalid/about",
            title="About",
            language="en",
            fetched_at=NOW,
            is_synthetic=True,
        )
    real = synthetic.model_copy(update={"url": "https://www.example.com/about", "is_synthetic": False})
    assert real.is_synthetic is False


def test_document_to_chunk_fixture_round_trips_and_hashes_match():
    document = DocumentRecord(
        document_id="doc_abc123",
        source_id="src_demo",
        file_type=FileType.HTML,
        source_type=SourceType.OFFICIAL_WEBSITE,
        title="Products",
        language="en",
        content="Acme exports pumps.",
        content_hash=content_sha256("Acme exports pumps."),
        fetched_at=NOW,
        is_synthetic=True,
        units=[{"kind": "section", "heading": "Products", "text": "Acme exports pumps."}],
        attributes={"canonical_url": "https://acme.synthetic.example/products"},
    )
    record = ChunkRecord(
        content="Acme exports pumps.",
        metadata=metadata(
            document_id=document.document_id,
            content_hash=content_sha256("Acme exports pumps."),
            parent_document_hash=document.content_hash,
            source_url="https://acme.synthetic.example/products",
        ),
    )
    assert ChunkRecord.model_validate_json(record.model_dump_json()) == record
    assert record.metadata.parent_document_hash == document.content_hash


def test_source_locator_supports_structured_and_json_serializable_variants():
    value = SourceLocator(table="monthly_trade", row=8, profile="2026-07", post_id="post-1", sql="SELECT 1")
    assert value.model_dump(mode="json")["table"] == "monthly_trade"
    assert value.post_id == "post-1"
    with pytest.raises(ValidationError):
        SourceLocator(page=-1)
    with pytest.raises(ValidationError):
        SourceLocator()
    with pytest.raises(ValidationError):
        SourceLocator(raw={})
    with pytest.raises(ValidationError):
        SourceLocator(raw="   ")
    with pytest.raises(ValidationError):
        SourceLocator(raw={"nested": ["", {"deep": "  "}]})
    assert SourceLocator(raw={"section": "Products", "page": 2}).raw == {"section": "Products", "page": 2}


def test_numeric_contracts_reject_coercion_and_accept_real_boundary_numbers():
    assert SourceLocator(page=1, block=0).page == 1
    assert ChunkMetadata(**metadata(source_weight=0)).source_weight == 0
    assert ChunkMetadata(**metadata(source_weight=1, ocr_confidence=1)).ocr_confidence == 1
    for value in ("1", True, False):
        with pytest.raises(ValidationError):
            SourceLocator(page=value)
    for value in ("0.5", True, False):
        with pytest.raises(ValidationError):
            ChunkMetadata(**metadata(source_weight=value))
        with pytest.raises(ValidationError):
            ChunkMetadata(**metadata(ocr_confidence=value))


def test_structured_json_rejects_non_json_values_and_non_finite_numbers():
    base = {
        "document_id": "doc_abc123", "source_id": "src_demo", "file_type": "html",
        "source_type": "official_website", "title": "Products", "language": "en",
        "content": "Acme exports pumps.", "content_hash": content_sha256("Acme exports pumps."),
        "fetched_at": NOW, "is_synthetic": True,
    }
    with pytest.raises(ValidationError):
        DocumentRecord(**base, units=[{"bad": object()}])
    with pytest.raises(ValidationError):
        DocumentRecord(**base, attributes={"bad": {1, 2}})
    with pytest.raises(ValidationError):
        DocumentRecord(**base, attributes={"bad": nan})
    with pytest.raises(ValidationError):
        ChunkMetadata(**metadata(aggregation_info={"bad": inf}))


def test_synthetic_urls_cover_canonical_and_all_url_bearing_records():
    with pytest.raises(ValidationError):
        SourceRecord(
            source_id="src_demo", source_type="official_website", file_type="html",
            url="https://acme.synthetic.example/about", canonical_url="https://evil.invalid/about",
            title="About", language="en", fetched_at=NOW, is_synthetic=True,
        )
    with pytest.raises(ValidationError):
        DocumentRecord(
            document_id="doc_abc123", source_id="src_demo", file_type="html",
            source_type="official_website", title="Products", language="en",
            content="x", content_hash=content_sha256("x"), fetched_at=NOW,
            is_synthetic=True, canonical_url="https://evil.invalid/products",
        )
    with pytest.raises(ValidationError):
        ChunkMetadata(**metadata(canonical_url="https://evil.invalid/chunk"))
    assert SourceRecord(
        source_id="src_demo", source_type="official_website", file_type="html",
        url="https://WWW.EXAMPLE.COM./about", canonical_url="https://EXAMPLE.ORG/about",
        title="About", language="en", fetched_at=NOW, is_synthetic=True,
    )


def test_model_copy_revalidates_updates_for_nested_contracts():
    document = DocumentRecord(
        document_id="doc_abc123", source_id="src_demo", file_type="html",
        source_type="official_website", title="Products", language="en", content="x",
        content_hash=content_sha256("x"), fetched_at=NOW, is_synthetic=True,
    )
    with pytest.raises(ValidationError):
        document.model_copy(update={"content": "tampered"})
    record = ChunkRecord(content="a chunk", metadata=ChunkMetadata(**metadata()))
    with pytest.raises(ValidationError):
        record.model_copy(update={"content": "tampered"})
    with pytest.raises(ValidationError):
        record.model_copy(update={"metadata": {**metadata(), "is_synthetic": None}})


def test_every_chunk_metadata_required_field_is_explicit():
    required = {
        "chunk_id", "document_id", "file_type", "source_type", "source_weight",
        "ingested_at", "source_locator", "content_hash", "parent_document_hash",
        "language", "is_synthetic",
    }
    for field in required:
        values = metadata()
        values.pop(field)
        with pytest.raises(ValidationError):
            ChunkMetadata(**values)
