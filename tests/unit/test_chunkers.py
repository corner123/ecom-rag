from datetime import datetime, timezone

import pytest

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


def test_protected_identifier_is_never_split_in_content_or_overlap():
    from trade_agent.data.chunkers import ChunkRouter

    identifier = "beta_super-long-SKU-12345"
    text = f"Opening paragraph. Alpha {identifier} omega. Closing paragraph with enough words to split."
    chunks = ChunkRouter(max_tokens=48, overlap_tokens=12, token_counter=len).chunk(
        doc(SourceType.SOCIAL, text, post_id="POST-001", row=1)
    )
    assert len(chunks) > 1
    assert any(identifier in chunk.content for chunk in chunks)
    assert not any(
        0 < len(part) < len(identifier) and part in identifier
        for chunk in chunks
        for part in chunk.content.split()
        if "beta_" in part or "SKU-12345" in part
    )
    assert all(len(chunk.content) <= 48 for chunk in chunks)


def test_atomic_identifier_that_cannot_fit_raises_explicit_error():
    from trade_agent.data.chunkers import ChunkRouter

    with pytest.raises(ValueError, match="atomic identifier.*cannot fit"):
        ChunkRouter(max_tokens=20, overlap_tokens=4, token_counter=len).chunk(
            doc(SourceType.SOCIAL, "beta_super-long-SKU-12345", post_id="POST-001", row=1)
        )


def test_nonadditive_counter_is_applied_to_full_prefix_and_slice():
    from trade_agent.data.chunkers import ChunkRouter

    def nonadditive(value: str) -> int:
        return 0 if not value else len(value) + 10

    value = doc(SourceType.INDUSTRY_NEWS, "Alpha beta gamma. Delta epsilon zeta. Eta theta iota.")
    chunks = ChunkRouter(max_tokens=38, overlap_tokens=16, token_counter=nonadditive).chunk(value)
    assert len(chunks) > 1
    assert all(nonadditive(chunk.content) <= 38 for chunk in chunks)
    assert all(chunk.content.startswith("Factory expansion\n") for chunk in chunks)
    assert all(not chunk.content.endswith(("bet", "gamm", "epsil")) for chunk in chunks)


def test_nonadditive_prefix_equal_to_budget_can_accept_repeated_body_token():
    import re

    from trade_agent.data.chunkers import ChunkRouter

    def unique_words(value: str) -> int:
        return len(set(re.findall(r"[A-Za-z]+", value.casefold())))

    chunks = ChunkRouter(max_tokens=2, overlap_tokens=0, token_counter=unique_words).chunk(
        doc(SourceType.INDUSTRY_NEWS, "Factory")
    )
    assert [chunk.content for chunk in chunks] == ["Factory expansion\nFactory"]
    assert unique_words(chunks[0].content) == 2


def test_nonadditive_prefix_equal_to_budget_still_rejects_new_body_token():
    import re

    from trade_agent.data.chunkers import ChunkRouter

    def unique_words(value: str) -> int:
        return len(set(re.findall(r"[A-Za-z]+", value.casefold())))

    with pytest.raises(ValueError, match="cannot fit|make progress"):
        ChunkRouter(max_tokens=2, overlap_tokens=0, token_counter=unique_words).chunk(
            doc(SourceType.INDUSTRY_NEWS, "Unrelated")
        )


def test_natural_boundaries_preserve_original_punctuation_and_whitespace():
    from trade_agent.data.chunkers import ChunkRouter

    text = "Paragraph one, exact!\n\nParagraph two has punctuation?  Yes; it does.\nFinal line."
    chunks = ChunkRouter(max_tokens=42, overlap_tokens=0, token_counter=len).chunk(
        doc(SourceType.SOCIAL, text, post_id="POST-001", row=1)
    )
    assert "".join(chunk.content for chunk in chunks) == text
    assert chunks[0].content == "Paragraph one, exact!\n\n"


def test_all_long_single_record_sources_honor_the_same_budget():
    from trade_agent.data.chunkers import ChunkRouter

    cases = [
        doc(SourceType.B2B, "Product sentence. " * 30, product_id=1, sku="SKU-1", units=[{"text": "x", "locator": {"row": 1, "raw": {"product_id": 1}}}]),
        doc(SourceType.SOCIAL, "Social sentence. " * 30, post_id="POST-1", row=1),
        doc(SourceType.CUSTOMS_PROFILE, "Profile sentence. " * 30, units=[{"text": "x", "locator": {"profile": "monthly_company_hs", "raw": {"calendar_month": "2026-01"}}}]),
    ]
    router = ChunkRouter(max_tokens=60, overlap_tokens=10, token_counter=len)
    for value in cases:
        chunks = router.chunk(value)
        assert len(chunks) > 1
        assert all(len(chunk.content) <= 60 for chunk in chunks)


def test_b2b_html_heading_is_injected_inside_every_chunk_budget(tmp_path):
    from trade_agent.data.chunkers import ChunkRouter
    from trade_agent.data.router import DocumentRouter, SourceInput

    path = tmp_path / "products.html"
    path.write_text(
        "<html><body><h1>Industrial Chargers</h1><p>Supplier Alpha offers SKU-CHG-100 under HS 850440. "
        "Export evidence remains attached to this product heading.</p></body></html>",
        encoding="utf-8",
    )
    source = SourceInput(
        path=path, file_type=FileType.HTML, source_type=SourceType.B2B, source_id="b2b-html",
        title="Catalog", language="en", fetched_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
        is_synthetic=True, source_url="https://supplier.example/catalog",
    )
    document = DocumentRouter().load(source)[0]
    chunks = ChunkRouter(max_tokens=72, overlap_tokens=8, token_counter=len).chunk(document)
    assert len(chunks) > 1
    assert all(chunk.content.startswith("Industrial Chargers\n") for chunk in chunks)
    assert all(len(chunk.content) <= 72 for chunk in chunks)
    assert all(chunk.metadata.source_locator.section == "Industrial Chargers" for chunk in chunks)
    assert any("SKU-CHG-100" in chunk.content and "HS 850440" in chunk.content for chunk in chunks)


def test_chinese_punctuation_provides_natural_deterministic_overlap_boundaries():
    from trade_agent.data.chunkers import ChunkRouter

    text = "企业提供工业充电器。产品通过出口认证！支持稳定供货；适用海外市场。联系团队获取报价。"
    value = doc(SourceType.SOCIAL, text, post_id="POST-CN-1", row=1)
    router = ChunkRouter(max_tokens=22, overlap_tokens=8, token_counter=len)
    first = router.chunk(value)
    second = router.chunk(value)
    assert len(first) > 1
    assert all(len(chunk.content) <= 22 for chunk in first)
    assert [(chunk.content, chunk.metadata.chunk_id) for chunk in first] == [
        (chunk.content, chunk.metadata.chunk_id) for chunk in second
    ]
    overlaps = []
    for previous, current in zip(first, first[1:]):
        sizes = [size for size in range(1, min(len(previous.content), len(current.content)) + 1)
                 if previous.content[-size:] == current.content[:size]]
        overlaps.append(max(sizes, default=0))
    assert any(size > 0 for size in overlaps)
    assert all(size <= 8 for size in overlaps)


def test_unpunctuated_chinese_uses_safe_codepoint_fallback_with_overlap():
    from trade_agent.data.chunkers import ChunkRouter

    text = "企业持续提供工业充电设备并为海外客户提供交付支持和售后服务" * 3
    value = doc(SourceType.SOCIAL, text, post_id="POST-CN-2", row=1)
    router = ChunkRouter(max_tokens=20, overlap_tokens=5, token_counter=len)
    chunks = router.chunk(value)
    assert len(chunks) > 1
    assert all(len(chunk.content) <= 20 for chunk in chunks)
    assert any(previous.content[-5:] == current.content[:5] for previous, current in zip(chunks, chunks[1:]))
    assert [chunk.content for chunk in chunks] == [chunk.content for chunk in router.chunk(value)]


def test_cjk_codepoint_fallback_never_splits_embedded_trade_identifier():
    from trade_agent.data.chunkers import ChunkRouter

    identifier = "beta_super-long-SKU-12345"
    text = "企业提供稳定交付支持" + identifier + "并持续提供售后服务和海外响应"
    chunks = ChunkRouter(max_tokens=32, overlap_tokens=6, token_counter=len).chunk(
        doc(SourceType.SOCIAL, text, post_id="POST-CN-ID", row=1)
    )
    assert any(identifier in chunk.content for chunk in chunks)
    assert all(
        identifier in chunk.content or "beta_" not in chunk.content and "SKU-12345" not in chunk.content
        for chunk in chunks
    )


def test_website_heading_is_inside_budget_and_full_locator_is_preserved():
    from trade_agent.data.chunkers import ChunkRouter

    heading = "Products > Chargers"
    value = doc(
        SourceType.OFFICIAL_WEBSITE,
        "Sentence one. Sentence two. Sentence three.",
        units=[{"text": "Sentence one. Sentence two. Sentence three.", "locator": {"section": heading, "raw": {"claim_id": "CLAIM-1", "item_source_url": "https://test.example/item"}}}],
    )
    chunks = ChunkRouter(max_tokens=38, overlap_tokens=5, token_counter=len).chunk(value)
    assert all(chunk.content.startswith(heading + "\n") and len(chunk.content) <= 38 for chunk in chunks)
    assert all(chunk.metadata.source_locator.raw["claim_id"] == "CLAIM-1" for chunk in chunks)


def test_pdf_never_crosses_pages_and_keeps_raw_table_block_and_confidence():
    from trade_agent.data.chunkers import ChunkRouter

    units = [
        {"text": "Page one sentence. " * 8, "kind": "text", "confidence": 0.91, "locator": {"page": 1, "block": 4, "raw": {"bbox": [1, 2, 3, 4], "kind": "text"}}},
        {"text": "A | B\n1 | 2", "kind": "table", "confidence": 0.82, "locator": {"page": 2, "block": 0, "table": "table-2-0", "raw": {"bbox": [5, 6, 7, 8], "kind": "table"}}},
    ]
    value = doc(SourceType.INDUSTRY_NEWS, "fallback", units=units).model_copy(update={"file_type": FileType.PDF})
    chunks = ChunkRouter(max_tokens=55, overlap_tokens=8, token_counter=len).chunk(value)
    assert {chunk.metadata.source_locator.page for chunk in chunks} == {1, 2}
    assert all(chunk.metadata.source_locator.page == 1 for chunk in chunks[:-1])
    assert chunks[-1].metadata.source_locator.table == "table-2-0"
    assert chunks[-1].metadata.source_locator.raw["bbox"] == [5, 6, 7, 8]
    assert {chunk.metadata.ocr_confidence for chunk in chunks if chunk.metadata.source_locator.page == 1} == {0.91}
    assert chunks[-1].metadata.ocr_confidence == 0.82


def test_chunk_ids_and_hashes_are_deterministic_for_repeat_runs():
    from trade_agent.data.chunkers import ChunkRouter

    value = doc(SourceType.INDUSTRY_NEWS, "Repeatable sentence. " * 20, dedupe_cluster_id="D-1")
    router = ChunkRouter(max_tokens=55, overlap_tokens=7, token_counter=len)
    first = router.chunk(value)
    second = router.chunk(value)
    assert [(c.metadata.chunk_id, c.metadata.content_hash, c.content) for c in first] == [
        (c.metadata.chunk_id, c.metadata.content_hash, c.content) for c in second
    ]


def test_overlap_uses_counter_units_and_safe_token_boundaries_exactly():
    from trade_agent.data.chunkers import BaseChunker

    chunker = BaseChunker(max_tokens=25, overlap_tokens=15, token_counter=len)
    chunks = chunker.split_text("Alpha sentence. Beta sentence. Gamma sentence.")
    assert all(len(chunk) <= 25 for chunk in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        overlaps = [size for size in range(1, min(len(previous), len(current)) + 1) if previous[-size:] == current[:size]]
        overlap = max(overlaps, default=0)
        assert overlap <= 15
        if overlap:
            assert previous[-overlap:].split() == current[:overlap].split()


@pytest.mark.parametrize("identifier", ["010121", "SKU.ABC/123", "AB12CD34", "https://example.test/a/b?x=1"])
def test_trade_identifiers_with_digits_dots_slashes_and_urls_are_atomic(identifier):
    from trade_agent.data.chunkers import ChunkRouter

    text = f"Before words. {identifier} after words with another sentence."
    chunks = ChunkRouter(max_tokens=45, overlap_tokens=8, token_counter=len).chunk(
        doc(SourceType.SOCIAL, text, post_id="POST-001", row=1)
    )
    assert any(identifier in chunk.content for chunk in chunks)
    marker_size = max(4, len(identifier) // 3)
    partial_markers = {identifier[start:start + marker_size] for start in range(len(identifier) - marker_size + 1)}
    assert all(identifier in chunk.content or not any(marker in chunk.content for marker in partial_markers) for chunk in chunks)
