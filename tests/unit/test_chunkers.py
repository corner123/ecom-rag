from datetime import datetime, timezone

from trade_agent.schemas.source import DocumentRecord, FileType, SourceType, content_sha256


def doc(source_type, content, **attributes):
    units = attributes.pop("units", [])
    return DocumentRecord(
        document_id="doc-test", source_id="source-test", file_type=FileType.JSON,
        source_type=source_type, title="Factory expansion", language="en", content=content,
        content_hash=content_sha256(content), fetched_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
        is_synthetic=True, units=units, attributes={"source_weight": 0.5, "source_weight_version": "task6-v1", **attributes},
        source_url="https://test.example/a",
    )


def test_news_title_injected_and_overlap_preserves_hs_and_sku():
    from trade_agent.data.chunkers import ChunkRouter

    text = " ".join(["paragraph carries HS 010121 SKU-A7B9." for _ in range(160)])
    chunks = ChunkRouter(max_tokens=40, overlap_tokens=8).chunk(doc(SourceType.INDUSTRY_NEWS, text, hs_code="010121", sku="SKU-A7B9"))
    assert len(chunks) > 1
    assert all(chunk.content.startswith("Factory expansion\n") for chunk in chunks)
    assert any("010121" in chunk.content and "SKU-A7B9" in chunk.content for chunk in chunks)
    assert all(chunk.metadata.source_weight == 0.5 for chunk in chunks)


def test_social_is_one_chunk_unless_genuinely_long():
    from trade_agent.data.chunkers import ChunkRouter

    one = ChunkRouter().chunk(doc(SourceType.SOCIAL, "brief post", post_id="POST-001", row=1))
    many = ChunkRouter(max_tokens=20, overlap_tokens=4).chunk(doc(SourceType.SOCIAL, "word " * 100, post_id="POST-001", row=1))
    assert len(one) == 1 and one[0].metadata.source_locator.post_id == "POST-001"
    assert len(many) > 1 and all(c.metadata.source_locator.post_id == "POST-001" for c in many)


def test_pdf_tables_are_independent_and_keep_page_locator():
    from trade_agent.data.chunkers import ChunkRouter

    value = doc(SourceType.REGULATOR, "fallback", units=[
        {"text": "page text", "locator": {"page": 1, "block": 0}},
        {"text": "A | B\n1 | 2", "kind": "table", "locator": {"page": 1, "block": 1, "table": "table-1"}},
    ])
    value = value.model_copy(update={"file_type": FileType.PDF})
    chunks = ChunkRouter().chunk(value)
    assert len(chunks) == 2
    assert chunks[1].metadata.source_locator.table == "table-1"
    assert all(c.metadata.source_locator.page == 1 for c in chunks)


def test_counter_units_include_injected_heading_and_overlap_exactly():
    from trade_agent.data.chunkers import ChunkRouter
    value = doc(SourceType.INDUSTRY_NEWS, "One short sentence. Two short sentence. Three short sentence.")
    chunks = ChunkRouter(max_tokens=28, overlap_tokens=5, token_counter=len).chunk(value)
    assert len(chunks) > 1
    assert all(len(chunk.content) <= 28 for chunk in chunks)
    assert all(chunk.content.startswith("Factory expansion\n") for chunk in chunks)


def test_fallback_counter_never_exceeds_budget_after_prefix():
    from trade_agent.data.chunkers import ChunkRouter
    chunks = ChunkRouter(max_tokens=12, overlap_tokens=3).chunk(doc(SourceType.OFFICIAL_WEBSITE, "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda."))
    assert len(chunks) > 1
    assert all(ChunkRouter(max_tokens=12, overlap_tokens=3).by[SourceType.OFFICIAL_WEBSITE].counter(c.content) <= 12 for c in chunks)
