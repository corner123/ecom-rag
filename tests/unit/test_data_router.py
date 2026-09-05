import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from trade_agent.schemas.source import FileType, SourceType


ROOT = Path(__file__).parents[2] / "demo" / "trade_intel_seed"
NOW = datetime(2026, 8, 30, tzinfo=timezone.utc)


def source(path, file_type, source_type, **extra):
    from trade_agent.data.router import SourceInput

    return SourceInput(
        path=ROOT / path,
        file_type=file_type,
        source_type=source_type,
        source_id=extra.pop("source_id", f"src-{path}"),
        title=extra.pop("title", path),
        language="en",
        fetched_at=NOW,
        is_synthetic=True,
        source_url=extra.pop("source_url", "https://ingest.example/source"),
        **extra,
    )


def test_routes_checked_in_item_boundaries_without_evaluation_labels():
    from trade_agent.data.router import DocumentRouter

    router = DocumentRouter()
    b2b = router.load(source("b2b/products.json", FileType.JSON, SourceType.B2B))
    news = router.load(source("news/stories.json", FileType.JSON, SourceType.INDUSTRY_NEWS))
    social = router.load(source("social/posts.jsonl", FileType.JSONL, SourceType.SOCIAL))
    profiles = [router.load(source(f"customs_profiles/{p.name}", FileType.GENERATED_PROFILE, SourceType.CUSTOMS_PROFILE))[0]
                for p in sorted((ROOT / "customs_profiles").glob("*.json"))]
    assert len(b2b) == 18
    assert "reference_claim_id" not in b2b[0].attributes
    assert "reference_claim_id" not in b2b[0].attributes["source_payload"]
    assert "claim_id" not in b2b[0].units[0]["locator"]["raw"]
    assert "manifest_claim_ids" not in b2b[0].units[0]["locator"]["raw"]
    assert "CLAIM-B2B" not in json.dumps(b2b[0].model_dump(mode="json"))
    assert len(news) == 16
    assert news[12].attributes["canonical_story_id"] == "NEWS-001"
    assert len(social) == 12
    assert social[0].units[0]["locator"]["post_id"] == "POST-001"
    assert len(profiles) == 54
    assert profiles[0].attributes["aggregation_info"]["aggregation_grain"] == "company_country_hs_calendar_month"


def test_b2b_news_and_social_reference_claim_id_is_optional_and_never_indexed(tmp_path):
    from trade_agent.data.router import DocumentRouter

    fixtures = (
        (
            "products.json",
            FileType.JSON,
            SourceType.B2B,
            {"products": [{
                "product_id": "P-1", "product_name": "Pump", "sku": "PUMP-1",
                "supplier": "Maker", "hs_code": "841370", "url": "https://marketplace.example/p-1",
                "reference_answer": "evaluation-only", "extra_evidence": {"minimum_order": 5, "gold_label": "yes"},
            }]},
        ),
        (
            "stories.json",
            FileType.JSON,
            SourceType.INDUSTRY_NEWS,
            {"stories": [{
                "id": "N-1", "headline": "Expansion", "body": "A new line opened.",
                "entity": "Maker", "publisher": "News", "canonical_story_id": "N-1",
                "dedupe_cluster_id": "N-1", "url": "https://news.example/n-1",
                "published_at": NOW.isoformat(),
            }]},
        ),
    )
    for name, file_type, source_type, payload in fixtures:
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        inp = source("b2b/products.json", file_type, source_type).model_copy(update={"path": path})
        documents = DocumentRouter().load(inp)
        assert len(documents) == 1
        serialized = json.dumps(documents[0].model_dump(mode="json"))
        assert "reference_claim" not in serialized
        assert "evaluation-only" not in serialized
        assert "gold_label" not in serialized
        if source_type is SourceType.B2B:
            assert "minimum_order" in serialized

    social_path = tmp_path / "posts.jsonl"
    social_path.write_text(json.dumps({
        "post_id": "S-1", "text": "Factory line opened.", "company": "Maker",
        "url": "https://social.example/s-1", "published_at": NOW.isoformat(),
    }), encoding="utf-8")
    social_input = source("social/posts.jsonl", FileType.JSONL, SourceType.SOCIAL).model_copy(
        update={"path": social_path}
    )
    documents = DocumentRouter().load(social_input)
    assert len(documents) == 1
    assert "reference_claim" not in json.dumps(documents[0].model_dump(mode="json"))


def test_html_and_markdown_use_heading_sections():
    from trade_agent.data.router import DocumentRouter

    router = DocumentRouter()
    html = router.load(source("website/section-01.html", FileType.HTML, SourceType.OFFICIAL_WEBSITE))
    markdown = router.load(source("website/methodology.md", FileType.MARKDOWN, SourceType.OFFICIAL_WEBSITE))
    assert len(html) >= 1 and html[0].units[0]["locator"]["section"]
    assert len(markdown) >= 1 and markdown[0].units[0]["locator"]["section"]


def test_html_golden_hierarchy_groups_evidence_once_and_attaches_preamble(tmp_path):
    from trade_agent.data.router import DocumentRouter

    path = tmp_path / "nested.html"
    path.write_text(
        """<!doctype html><html><head><style>.bad{}</style><script>leak()</script></head><body>
        <aside>SYNTHETIC PREAMBLE</aside>
        <h1>Root <em>One</em></h1>
        <section>Direct section text.
          <p>First <strong>inline</strong> paragraph.</p>
          <p>Second paragraph.</p>
          <h2>Child</h2><div>Direct child text.</div>
          <ul><li>List <span>item</span>.</li></ul>
          <table><tr><th>Column</th><td>Value</td></tr></table>
          <h3>Grandchild</h3><p>Deep evidence.</p>
        </section>
        <h1>Sibling</h1><p>Sibling evidence.</p>
        </body></html>""",
        encoding="utf-8",
    )
    docs = DocumentRouter().load(
        source("website/section-01.html", FileType.HTML, SourceType.OFFICIAL_WEBSITE).model_copy(
            update={"path": path, "title": "File title"}
        )
    )

    assert [(doc.title, doc.content) for doc in docs] == [
        ("Root One", "SYNTHETIC PREAMBLE\nDirect section text.\nFirst inline paragraph.\nSecond paragraph."),
        ("Root One > Child", "Direct child text.\nList item.\nColumn\nValue"),
        ("Root One > Child > Grandchild", "Deep evidence."),
        ("Sibling", "Sibling evidence."),
    ]
    combined = "\n".join(doc.content for doc in docs)
    for evidence in ("SYNTHETIC PREAMBLE", "Direct section text.", "First inline paragraph.", "List item.", "Value"):
        assert combined.count(evidence) == 1
    assert "leak" not in combined and ".bad" not in combined


def test_markdown_golden_hierarchy_matches_html_without_file_title_prefix(tmp_path):
    from trade_agent.data.router import DocumentRouter

    path = tmp_path / "nested.md"
    path.write_text(
        "Preamble evidence.\n\n# Root\n\nFirst paragraph.\n\n## Child\n\n- List item\n\n"
        "| Column | Value |\n| --- | --- |\n\n### Grandchild\n\nDeep evidence.\n\n# Sibling\n\nSibling evidence.\n",
        encoding="utf-8",
    )
    docs = DocumentRouter().load(
        source("website/methodology.md", FileType.MARKDOWN, SourceType.OFFICIAL_WEBSITE).model_copy(
            update={"path": path, "title": "File title"}
        )
    )
    assert [doc.title for doc in docs] == ["Root", "Root > Child", "Root > Child > Grandchild", "Sibling"]
    assert docs[0].content == "Preamble evidence.\n\nFirst paragraph."
    assert "- List item" in docs[1].content and "| Column | Value |" in docs[1].content


@pytest.mark.parametrize(
    ("markup", "expected"),
    [
        ("<section>We sell <strong>industrial chargers</strong> globally.</section>", "We sell industrial chargers globally."),
        ("<div><span>Nested evidence only</span></div>", "Nested evidence only"),
    ],
)
def test_html_leaf_inline_text_is_preserved_exactly_once(tmp_path, markup, expected):
    from trade_agent.data.router import DocumentRouter

    path = tmp_path / "inline.html"
    path.write_text(f"<html><body><h1>Products</h1>{markup}</body></html>", encoding="utf-8")
    inp = source("website/section-01.html", FileType.HTML, SourceType.OFFICIAL_WEBSITE).model_copy(
        update={"path": path}
    )
    docs = DocumentRouter().load(inp)
    assert [(item.title, item.content) for item in docs] == [("Products", expected)]
    assert docs[0].content.count(expected) == 1


def test_markdown_fenced_code_hash_is_not_a_heading(tmp_path):
    from trade_agent.data.router import DocumentRouter

    path = tmp_path / "fenced.md"
    path.write_text(
        "# Root\n\nBefore.\n\n```markdown\n# not a heading\n## neither is this\n```\n\nAfter.\n\n## Child\nEvidence.\n",
        encoding="utf-8",
    )
    inp = source("website/methodology.md", FileType.MARKDOWN, SourceType.OFFICIAL_WEBSITE).model_copy(
        update={"path": path}
    )
    docs = DocumentRouter().load(inp)
    assert [item.title for item in docs] == ["Root", "Root > Child"]
    assert "# not a heading" in docs[0].content
    assert "## neither is this" in docs[0].content


def test_mismatched_or_malformed_input_quarantines_instead_of_text_fallback(tmp_path):
    from trade_agent.data.router import DocumentRouter

    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    router = DocumentRouter()
    inp = source("b2b/products.json", FileType.PDF, SourceType.B2B)
    assert router.load(inp) == []
    malformed = inp.model_copy(update={"path": bad, "file_type": FileType.JSON})
    assert router.load(malformed) == []
    assert [q.error_code for q in router.quarantines] == ["FILE_TYPE_MISMATCH", "MALFORMED_JSON"]


def test_rejects_unsupported_source_file_pair():
    from trade_agent.data.router import DocumentRouter

    router = DocumentRouter()
    assert router.load(source("social/posts.jsonl", FileType.JSONL, SourceType.B2B)) == []
    assert router.quarantines[-1].error_code == "UNSUPPORTED_SOURCE_FILE_PAIR"


def test_manifest_probe_is_deterministic_and_never_routes_ledger():
    from trade_agent.data.router import probe_manifest

    first = probe_manifest(ROOT, ROOT / "manifests" / "corpus_manifest.json")
    second = probe_manifest(ROOT, ROOT / "manifests" / "corpus_manifest.json")
    assert first == second
    assert first["documents"] == 116
    assert first["quarantines"] == [{"error_code": "SCANNED_PDF_OCR_UNAVAILABLE", "path": "pdf/scanned-regulator-notice.pdf"}]


def test_quarantine_diagnostics_redact_credentials_and_jsonl_limit(tmp_path):
    from trade_agent.data.quarantine import sanitize_diagnostic
    from trade_agent.data.router import DocumentRouter
    assert "secret-value" not in sanitize_diagnostic("token=secret-value https://u:p@example.test/?key=value")
    path = tmp_path / "posts.jsonl"; path.write_text('{"post_id":"a","text":"x"}\n{"post_id":"b","text":"x"}\n{"post_id":"c","text":"x"}\n')
    inp = source("social/posts.jsonl", FileType.JSONL, SourceType.SOCIAL).model_copy(update={"path": path})
    assert DocumentRouter(max_documents=2).load(inp) == []


def test_source_input_and_router_limits_are_strict_positive_and_json_safe():
    from trade_agent.data.router import DocumentRouter, SourceInput

    for kwargs in ({"max_bytes": 0}, {"max_bytes": True}, {"max_documents": 0}, {"max_documents": 1.5}):
        with pytest.raises((TypeError, ValueError)):
            DocumentRouter(**kwargs)
    with pytest.raises(ValidationError):
        source("b2b/products.json", FileType.JSON, SourceType.B2B).model_copy(
            update={"manifest_attributes": {"bad": Path("not-json")}}
        )
    with pytest.raises(ValidationError):
        source("b2b/products.json", FileType.JSON, SourceType.B2B).model_copy(
            update={"valid_from": NOW, "valid_to": NOW.replace(year=2025)}
        )
    with pytest.raises(ValidationError):
        SourceInput.model_validate({})
    with pytest.raises(ValidationError):
        source("b2b/products.json", FileType.JSON, SourceType.B2B).model_copy(
            update={"source_url": "https://not-reserved.invalid/feed"}
        )


def test_stat_and_bounded_read_enforce_size_even_if_stat_lies(tmp_path, monkeypatch):
    from trade_agent.data.router import DocumentRouter

    path = tmp_path / "products.json"
    path.write_bytes(b'{"products":[]}' + b"x" * 64)
    real_stat = Path.stat

    class Small:
        st_size = 1

    monkeypatch.setattr(Path, "stat", lambda self: Small() if self == path else real_stat(self))
    router = DocumentRouter(max_bytes=16)
    inp = source("b2b/products.json", FileType.JSON, SourceType.B2B).model_copy(update={"path": path})
    assert router.load(inp) == []
    assert router.quarantines[-1].error_code == "INPUT_TOO_LARGE"


def test_content_media_disagreement_is_rejected_even_when_extension_matches(tmp_path):
    from trade_agent.data.router import DocumentRouter

    path = tmp_path / "products.json"
    path.write_text("<!doctype html><html><body><p>Not a product feed</p></body></html>", encoding="utf-8")
    router = DocumentRouter()
    inp = source("b2b/products.json", FileType.JSON, SourceType.B2B).model_copy(update={"path": path})
    assert router.load(inp) == []
    assert router.quarantines[-1].error_code == "FILE_TYPE_MISMATCH"


def test_html_signature_allows_leading_comments_before_real_markup(tmp_path):
    from trade_agent.data.router import DocumentRouter

    path = tmp_path / "commented.html"
    path.write_text("<!-- generated -->\n<html><body><h1>Heading</h1><p>Evidence</p></body></html>", encoding="utf-8")
    inp = source("website/section-01.html", FileType.HTML, SourceType.OFFICIAL_WEBSITE).model_copy(update={"path": path})
    docs = DocumentRouter().load(inp)
    assert [(doc.title, doc.content) for doc in docs] == [("Heading", "Evidence")]


@pytest.mark.parametrize(
    ("payload", "expected_kind"),
    [
        (b"%PDF-1.7\n", "pdf"),
        (b"<!doctype html><html><body><p>Evidence</p></body></html>", "html"),
        (b'{"products": []}', "json"),
        (b'{"post_id":"P"}\n{"post_id":"Q"}\n', "jsonl"),
        (b'{"company":"Example","company_id":1,"country_code":"CN","hs_code":"010121","calendar_month":"2026-08","aggregation_grain":"month"}', "generated_profile"),
        (b"# Heading\n\nAngle-bracket prose: <not-a-tag> and `x < y`.\n", "text"),
        (b"<?xml version='1.0'?><svg xmlns='http://www.w3.org/2000/svg'><text>Not markdown</text></svg>", "xml_or_svg"),
        (b"<catalog><item>Not markdown</item></catalog>", "xml_or_svg"),
        (b"PK\x03\x04archive", "archive"),
        (b"\x89PNG\r\n\x1a\nimage", "image"),
        (b"BM" + (54).to_bytes(4, "little") + b"\x00" * 8 + (40).to_bytes(4, "little") + b"\x00" * 36, "image"),
        (b"\x00\x00\x01\x00\x01\x00" + b"\x00" * 16, "image"),
        (b"\x00\x01\x02\x03", "binary"),
    ],
)
def test_content_sniffer_classifies_bounded_physical_media(payload, expected_kind):
    """Fails if a new physical format can silently enter a text parser."""
    from trade_agent.data.router import sniff_media_kind

    assert sniff_media_kind(payload).value == expected_kind


@pytest.mark.parametrize(
    ("payload", "expected_title", "expected_content"),
    [
        (
            b'<?xml version="1.0"?><html><body><h1>Heading</h1><p>Evidence</p></body></html>',
            "Heading",
            "Evidence",
        ),
        (
            b'<?xml version="1.0"?>\n<!-- generated -->\n'
            b'<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>Namespaced</h1><p>Evidence</p></body></html>',
            "Namespaced",
            "Evidence",
        ),
        (
            b'<?XML version="1.0"?>\n<!-- generated -->\n'
            b'<HTML XMLNS="http://www.w3.org/1999/xhtml"><BODY><H1>Uppercase</H1><P>Evidence</P></BODY></HTML>',
            "Uppercase",
            "Evidence",
        ),
    ],
)
def test_xhtml_prolog_and_html_root_route_as_declared_html(tmp_path, payload, expected_title, expected_content):
    """Fails if an XML prolog hides the XHTML root from physical HTML classification."""
    from trade_agent.data.router import DocumentRouter, sniff_media_kind

    path = tmp_path / "document.html"
    path.write_bytes(payload)
    inp = source("website/section-01.html", FileType.HTML, SourceType.OFFICIAL_WEBSITE).model_copy(update={"path": path})

    assert sniff_media_kind(payload).value == "html"
    assert [(doc.title, doc.content) for doc in DocumentRouter().load(inp)] == [(expected_title, expected_content)]


@pytest.mark.parametrize(
    "markup",
    [
        "<em>Legitimate Markdown inline HTML</em>",
        "<strong>Strong Markdown evidence</strong>",
        '<a href="https://supplier.example">Linked Markdown evidence</a>',
        "<code>inline_code()</code>",
        '<span data-kind="evidence">Spanned Markdown evidence</span>',
        "<em>First inline element</em> and <strong>second inline element</strong>",
    ],
)
def test_known_inline_html_root_remains_valid_markdown(tmp_path, markup):
    """Fails if a complete Markdown-compatible inline element is treated as generic XML."""
    from trade_agent.data.router import DocumentRouter, sniff_media_kind

    payload = markup.encode()
    path = tmp_path / "inline.md"
    path.write_bytes(payload)
    inp = source("website/methodology.md", FileType.MARKDOWN, SourceType.OFFICIAL_WEBSITE).model_copy(update={"path": path})

    assert sniff_media_kind(payload).value == "text"
    assert [doc.content for doc in DocumentRouter().load(inp)] == [markup]


@pytest.mark.parametrize(
    "payload",
    [
        b"<invoice><id>42</id></invoice>",
        b"<invoice><id>42</id></invoice><!-- trailing comment --><?audit complete?>",
        b'<svg xmlns="http://www.w3.org/2000/svg"><text>Not Markdown</text></svg>',
    ],
)
def test_generic_xml_and_svg_roots_remain_rejected_for_markdown(tmp_path, payload):
    """Fails if the inline-HTML allowance broadens to arbitrary XML-like roots."""
    from trade_agent.data.router import DocumentRouter, sniff_media_kind

    path = tmp_path / "disguised.md"
    path.write_bytes(payload)
    router = DocumentRouter()
    inp = source("website/methodology.md", FileType.MARKDOWN, SourceType.OFFICIAL_WEBSITE).model_copy(update={"path": path})

    assert sniff_media_kind(payload).value == "xml_or_svg"
    assert router.load(inp) == []
    assert router.quarantines[-1].error_code == "FILE_TYPE_MISMATCH"


@pytest.mark.parametrize(
    ("name", "payload", "file_type", "source_type"),
    [
        ("asset.md", b"<?xml version='1.0'?><svg xmlns='http://www.w3.org/2000/svg'><text>Not markdown</text></svg>", FileType.MARKDOWN, SourceType.OFFICIAL_WEBSITE),
        ("asset.md", b"PK\x03\x04archive", FileType.MARKDOWN, SourceType.OFFICIAL_WEBSITE),
        ("asset.md", b"\x89PNG\r\n\x1a\nimage", FileType.MARKDOWN, SourceType.OFFICIAL_WEBSITE),
        ("asset.md", b"\x00\x01\x02\x03", FileType.MARKDOWN, SourceType.OFFICIAL_WEBSITE),
        ("asset.html", b"%PDF-1.7\n", FileType.HTML, SourceType.OFFICIAL_WEBSITE),
        ("asset.json", b"<!doctype html><html><body><p>Evidence</p></body></html>", FileType.JSON, SourceType.B2B),
        ("asset.html", b'{"products": []}', FileType.HTML, SourceType.OFFICIAL_WEBSITE),
        ("asset.md", b'{"products": []}', FileType.MARKDOWN, SourceType.OFFICIAL_WEBSITE),
        ("asset.json", b'{"company":"Example","company_id":1,"country_code":"CN","hs_code":"010121","calendar_month":"2026-08","aggregation_grain":"month"}', FileType.JSON, SourceType.B2B),
    ],
)
def test_content_media_mismatch_quarantines_before_the_declared_parser(tmp_path, name, payload, file_type, source_type):
    """Fails if a disguised physical type reaches a different declared parser."""
    from trade_agent.data.router import DocumentRouter

    path = tmp_path / name
    path.write_bytes(payload)
    router = DocumentRouter()
    inp = source("website/methodology.md", file_type, source_type).model_copy(update={"path": path})

    assert router.load(inp) == []
    assert router.quarantines[-1].error_code == "FILE_TYPE_MISMATCH"


def test_markdown_with_angle_bracket_prose_and_fenced_markup_is_still_routed(tmp_path):
    """Fails if XML/HTML sniffing treats ordinary Markdown punctuation as markup."""
    from trade_agent.data.router import DocumentRouter

    path = tmp_path / "legitimate.md"
    path.write_text(
        "# Notes\n\nUse <supplier@example.test> and write `x < y`.\n\n"
        "```html\n<svg><text>Example code, not a document.</text></svg>\n```\n",
        encoding="utf-8",
    )
    inp = source("website/methodology.md", FileType.MARKDOWN, SourceType.OFFICIAL_WEBSITE).model_copy(update={"path": path})

    docs = DocumentRouter().load(inp)
    assert len(docs) == 1
    assert "<supplier@example.test>" in docs[0].content
    assert "<svg><text>Example code, not a document.</text></svg>" in docs[0].content


def test_large_json_content_cannot_be_renamed_as_markdown(tmp_path):
    """Fails if a sniff prefix truncates a JSON document into ordinary text."""
    from trade_agent.data.router import DocumentRouter

    path = tmp_path / "disguised.md"
    path.write_text('{"products":[],"padding":"' + "x" * 70_000 + '"}', encoding="utf-8")
    inp = source("website/methodology.md", FileType.MARKDOWN, SourceType.OFFICIAL_WEBSITE).model_copy(update={"path": path})
    router = DocumentRouter(max_bytes=80_000)

    assert router.load(inp) == []
    assert router.quarantines[-1].error_code == "FILE_TYPE_MISMATCH"


def test_div_root_html_fragment_remains_a_supported_html_document(tmp_path):
    """Fails if content validation narrows established fragment-root HTML support."""
    from trade_agent.data.router import DocumentRouter

    path = tmp_path / "fragment.html"
    path.write_text("<div><h1>Catalog</h1><p>Evidence</p></div>", encoding="utf-8")
    inp = source("website/section-01.html", FileType.HTML, SourceType.OFFICIAL_WEBSITE).model_copy(update={"path": path})

    assert [(doc.title, doc.content) for doc in DocumentRouter().load(inp)] == [("Catalog", "Evidence")]


def test_markdown_starting_with_bm_prose_is_not_a_bmp_image(tmp_path):
    """Fails if a weak two-byte bitmap prefix rejects ordinary Markdown text."""
    from trade_agent.data.router import DocumentRouter

    path = tmp_path / "bm-prose.md"
    path.write_text("BM exporters discuss current shipment schedules.", encoding="utf-8")
    inp = source("website/methodology.md", FileType.MARKDOWN, SourceType.OFFICIAL_WEBSITE).model_copy(update={"path": path})

    docs = DocumentRouter().load(inp)
    assert [doc.content for doc in docs] == ["BM exporters discuss current shipment schedules."]


@pytest.mark.parametrize(
    ("name", "payload", "file_type", "source_type", "code"),
    [
        ("wrong.pdf", b"not a pdf", FileType.PDF, SourceType.REGULATOR, "FILE_TYPE_MISMATCH"),
        ("wrong.html", b"plain text only", FileType.HTML, SourceType.OFFICIAL_WEBSITE, "FILE_TYPE_MISMATCH"),
        ("wrong.json", b"<html><p>x</p></html>", FileType.JSON, SourceType.B2B, "FILE_TYPE_MISMATCH"),
        ("bad.json", b"not json", FileType.JSON, SourceType.B2B, "MALFORMED_JSON"),
        ("bad.jsonl", b'{"post_id":', FileType.JSONL, SourceType.SOCIAL, "MALFORMED_JSON"),
        ("bad.md", b"\xff\xfe", FileType.MARKDOWN, SourceType.OFFICIAL_WEBSITE, "FILE_TYPE_MISMATCH"),
    ],
)
def test_file_type_mismatch_and_parse_failures_have_specific_codes(tmp_path, name, payload, file_type, source_type, code):
    from trade_agent.data.router import DocumentRouter

    path = tmp_path / name
    path.write_bytes(payload)
    router = DocumentRouter()
    inp = source("website/methodology.md", file_type, source_type).model_copy(update={"path": path})
    assert router.load(inp) == []
    assert router.quarantines[-1].error_code == code
    assert router.quarantines[-1].error_code != "PARSE_FAILED"


@pytest.mark.parametrize(
    ("file_type", "source_type", "payload"),
    [
        (FileType.JSON, SourceType.B2B, {"products": [{"product_id": 1, "product_name": "x"}]}),
        (FileType.JSON, SourceType.INDUSTRY_NEWS, {"stories": [{"id": "N", "headline": "x", "body": 3}]}),
        (FileType.JSONL, SourceType.SOCIAL, {"post_id": "P", "text": ["bad"]}),
        (FileType.GENERATED_PROFILE, SourceType.CUSTOMS_PROFILE, {"company": "x", "company_id": True, "country_code": "CN", "hs_code": "010121", "calendar_month": "2026-01", "aggregation_grain": "g"}),
    ],
)
def test_business_payloads_require_complete_typed_shapes(tmp_path, file_type, source_type, payload):
    from trade_agent.data.router import DocumentRouter

    suffix = ".jsonl" if file_type is FileType.JSONL else ".json"
    path = tmp_path / f"bad{suffix}"
    text = json.dumps(payload) + ("\n" if file_type is FileType.JSONL else "")
    path.write_text(text, encoding="utf-8")
    router = DocumentRouter()
    inp = source("social/posts.jsonl" if file_type is FileType.JSONL else "b2b/products.json", file_type, source_type).model_copy(update={"path": path})
    assert router.load(inp) == []
    assert router.quarantines[-1].error_code == "INVALID_DOCUMENT_SHAPE"


@pytest.mark.parametrize(("file_type", "source_type", "text"), [
    (FileType.JSON, SourceType.B2B, '{"products":[]}'),
    (FileType.JSON, SourceType.INDUSTRY_NEWS, '{"stories":[]}'),
    (FileType.JSONL, SourceType.SOCIAL, "\n\n"),
])
def test_empty_structured_feeds_are_invalid_shapes(tmp_path, file_type, source_type, text):
    from trade_agent.data.router import DocumentRouter

    suffix = ".jsonl" if file_type is FileType.JSONL else ".json"
    path = tmp_path / f"empty{suffix}"
    path.write_text(text, encoding="utf-8")
    router = DocumentRouter()
    inp = source("social/posts.jsonl" if file_type is FileType.JSONL else "b2b/products.json", file_type, source_type).model_copy(update={"path": path})
    assert router.load(inp) == []
    assert router.quarantines[-1].error_code == "INVALID_DOCUMENT_SHAPE"


@pytest.mark.parametrize("file_type,source_type,body", [
    (FileType.HTML, SourceType.OFFICIAL_WEBSITE, "<h1>A</h1><p>x</p><h1>B</h1><p>y</p>"),
    (FileType.MARKDOWN, SourceType.OFFICIAL_WEBSITE, "# A\nx\n# B\ny\n"),
    (FileType.JSON, SourceType.B2B, None),
    (FileType.JSONL, SourceType.SOCIAL, None),
])
def test_document_limit_is_enforced_inside_every_multi_document_loop(tmp_path, file_type, source_type, body):
    from trade_agent.data.router import DocumentRouter

    if file_type is FileType.JSON:
        payload = {"products": [
            {"product_id": i, "product_name": f"p{i}", "sku": f"S{i}", "supplier": "s", "hs_code": "010121", "url": f"https://marketplace.example/{i}", "reference_claim_id": f"C{i}"}
            for i in range(2)
        ]}
        body = json.dumps(payload)
    elif file_type is FileType.JSONL:
        body = "\n".join(json.dumps({"post_id": f"P{i}", "text": "x", "company": "c", "published_at": NOW.isoformat(), "url": f"https://social.example/{i}", "reference_claim_id": f"C{i}"}) for i in range(2))
    suffix = {FileType.HTML: ".html", FileType.MARKDOWN: ".md", FileType.JSON: ".json", FileType.JSONL: ".jsonl"}[file_type]
    path = tmp_path / f"many{suffix}"
    path.write_text(body, encoding="utf-8")
    router = DocumentRouter(max_documents=1)
    inp = source("website/section-01.html", file_type, source_type).model_copy(update={"path": path})
    assert router.load(inp) == []
    assert router.quarantines[-1].error_code == "DOCUMENT_COUNT_LIMIT"


def test_news_count_limit_wins_before_later_item_validation(tmp_path):
    from trade_agent.data.router import DocumentRouter

    primary = json.loads((ROOT / "news" / "stories.json").read_text())["stories"][0]
    path = tmp_path / "stories.json"
    path.write_text(json.dumps({"stories": [primary, {"bad": "shape"}]}), encoding="utf-8")
    inp = source("news/stories.json", FileType.JSON, SourceType.INDUSTRY_NEWS).model_copy(update={"path": path})
    router = DocumentRouter(max_documents=1)
    assert router.load(inp) == []
    assert router.quarantines[-1].error_code == "DOCUMENT_COUNT_LIMIT"


def test_item_and_feed_urls_and_all_provenance_survive_documents_and_chunks():
    from trade_agent.data.chunkers import ChunkRouter
    from trade_agent.data.router import DocumentRouter

    router = DocumentRouter()
    feed = "https://feed.example/news-batch"
    docs = router.load(source("news/stories.json", FileType.JSON, SourceType.INDUSTRY_NEWS, source_url=feed))
    primary, mirror = docs[0], docs[12]
    assert str(primary.source_url) == "https://newsroom.example/story/001"
    assert str(mirror.source_url) == "https://syndication.example/story/013"
    assert str(mirror.canonical_url) == "https://newsroom.example/story/001"
    assert mirror.attributes["feed_source_url"] == feed
    assert mirror.attributes["item_source_url"] == "https://syndication.example/story/013"
    raw = mirror.units[0]["locator"]["raw"]
    assert raw["feed_source_url"] == feed
    assert raw["item_source_url"] == "https://syndication.example/story/013"
    assert raw["item_canonical_url"] == "https://newsroom.example/story/001"
    assert raw["story_id"] == "NEWS-013"
    assert raw["canonical_story_id"] == "NEWS-001"
    assert "claim_id" not in raw
    assert raw["dedupe_cluster_id"] == "NEWS-CLUSTER-001"
    chunk = ChunkRouter().chunk(mirror)[0]
    assert chunk.metadata.source_locator.raw == raw
    assert str(chunk.metadata.source_url) == str(mirror.source_url)
    assert str(chunk.metadata.canonical_url) == str(mirror.canonical_url)
    assert chunk.metadata.dedupe_cluster_id == "NEWS-CLUSTER-001"


def test_untrusted_item_metadata_cannot_override_catalog_provenance_or_trust(tmp_path):
    from trade_agent.data.chunkers import ChunkRouter
    from trade_agent.data.router import DocumentRouter

    trusted_item_url = "https://marketplace.example/products/P-1"
    item = {
        "product_id": "P-1",
        "product_name": "Industrial charger",
        "supplier": "Trusted Supplier",
        "sku": "CHG-100",
        "hs_code": "850440",
        "url": trusted_item_url,
        "reference_claim_id": "CLAIM-P-1",
        "description": "Export-ready industrial charger.",
        "feed_source_url": "https://evil.example/feed",
        "feed_canonical_url": "https://evil.example/canonical-feed",
        "item_source_url": "https://evil.example/item",
        "item_canonical_url": "https://evil.example/canonical-item",
        "source_weight": 1,
        "source_weight_version": "evil-v1",
        "fact_type": "risk_event",
        "license_scope": "evil-license",
        "publish_time": "1999-01-01T00:00:00Z",
        "published_at": "1999-01-01T00:00:00Z",
        "valid_from": "1999-01-01T00:00:00Z",
        "valid_to": "2099-01-01T00:00:00Z",
        "country_code": "US",
    }
    path = tmp_path / "products.json"
    path.write_text(json.dumps({"products": [item]}), encoding="utf-8")
    inp = source(
        "b2b/products.json",
        FileType.JSON,
        SourceType.B2B,
        source_url="https://trusted.example/feed",
        canonical_url="https://trusted.example/feed-canonical",
        publish_time=NOW,
        valid_from=NOW,
        valid_to=NOW,
        license_scope="catalog-license",
        manifest_attributes={
            "source_weight": 0.25,
            "source_weight_version": "catalog-v1",
            "fact_type": "product_offering",
            "country_code": "CN",
        },
    ).model_copy(update={"path": path})

    document = DocumentRouter().load(inp)[0]
    chunk = ChunkRouter().chunk(document)[0]
    assert str(document.source_url) == trusted_item_url
    assert str(document.canonical_url) == trusted_item_url
    assert document.attributes["feed_source_url"] == "https://trusted.example/feed"
    assert document.attributes["feed_canonical_url"] == "https://trusted.example/feed-canonical"
    assert document.attributes["item_source_url"] == trusted_item_url
    assert document.attributes["item_canonical_url"] == trusted_item_url
    assert document.attributes["source_weight"] == 0.25
    assert document.attributes["source_weight_version"] == "catalog-v1"
    assert document.attributes["fact_type"] == "product_offering"
    assert document.attributes["license_scope"] == "catalog-license"
    assert document.attributes["publish_time"] == NOW.isoformat()
    assert document.attributes["country_code"] == "CN"
    assert document.attributes["source_payload"]["feed_source_url"] == "https://evil.example/feed"
    assert document.attributes["source_payload"]["country_code"] == "US"
    assert chunk.metadata.source_weight == 0.25
    assert chunk.metadata.fact_type.value == "product_offering"
    assert chunk.metadata.license_scope == "catalog-license"
    assert chunk.metadata.publish_time == NOW
    assert str(chunk.metadata.source_url) == trusted_item_url
    assert chunk.metadata.source_locator.raw["feed_source_url"] == "https://trusted.example/feed"


@pytest.mark.parametrize("kind", ["news", "social", "profile"])
def test_reserved_metadata_ownership_is_uniform_across_structured_routes(tmp_path, kind):
    from trade_agent.data.chunkers import ChunkRouter
    from trade_agent.data.router import DocumentRouter

    if kind == "news":
        item = json.loads((ROOT / "news/stories.json").read_text(encoding="utf-8"))["stories"][0]
        payload = {"stories": [item]}
        file_type, source_type, fixture, suffix = FileType.JSON, SourceType.INDUSTRY_NEWS, "news/stories.json", ".json"
    elif kind == "social":
        item = json.loads((ROOT / "social/posts.jsonl").read_text(encoding="utf-8").splitlines()[0])
        payload = item
        file_type, source_type, fixture, suffix = FileType.JSONL, SourceType.SOCIAL, "social/posts.jsonl", ".jsonl"
    else:
        item = json.loads((ROOT / "customs_profiles/001-010121-2025-03.json").read_text(encoding="utf-8"))
        payload = item
        file_type, source_type = FileType.GENERATED_PROFILE, SourceType.CUSTOMS_PROFILE
        fixture, suffix = "customs_profiles/001-010121-2025-03.json", ".json"
    item.update({
        "feed_source_url": "https://evil.example/feed",
        "item_source_url": "https://evil.example/item",
        "item_canonical_url": "https://evil.example/canonical",
        "source_weight": 1,
        "fact_type": "risk_event",
        "license_scope": "evil-license",
        "publish_time": "1999-01-01T00:00:00Z",
        "valid_from": "1999-01-01T00:00:00Z",
        "ocr_confidence": 1,
    })
    path = tmp_path / f"record{suffix}"
    path.write_text(json.dumps(payload) + ("\n" if file_type is FileType.JSONL else ""), encoding="utf-8")
    inp = source(
        fixture, file_type, source_type,
        source_url="https://trusted.example/feed", publish_time=NOW, valid_from=NOW,
        license_scope="catalog-license",
        manifest_attributes={"source_weight": 0.4, "fact_type": "market_signal"},
    ).model_copy(update={"path": path})
    document = DocumentRouter().load(inp)[0]
    chunk = ChunkRouter().chunk(document)[0]
    assert document.attributes["feed_source_url"] == "https://trusted.example/feed"
    assert document.attributes["source_weight"] == 0.4
    assert document.attributes["fact_type"] == "market_signal"
    assert document.attributes["license_scope"] == "catalog-license"
    assert document.attributes["publish_time"] == NOW.isoformat()
    assert document.attributes["source_payload"]["item_source_url"] == "https://evil.example/item"
    assert document.attributes["source_payload"]["ocr_confidence"] == 1
    assert chunk.metadata.source_weight == 0.4
    assert chunk.metadata.fact_type.value == "market_signal"
    assert chunk.metadata.publish_time == NOW
    assert chunk.metadata.license_scope == "catalog-license"


def test_b2b_evidence_text_includes_canonical_identity_fields_without_duplicates(tmp_path):
    from trade_agent.data.router import DocumentRouter

    item = {
        "product_id": "P-2",
        "product_name": "Marine Pump",
        "category": "Pumps",
        "supplier": "Harbor Works",
        "sku": "PUMP-200",
        "hs_code": "841370",
        "url": "https://marketplace.example/products/P-2",
        "reference_claim_id": "CLAIM-P-2",
        "description": "Marine Pump PUMP-200 is export ready.",
    }
    path = tmp_path / "products.json"
    path.write_text(json.dumps({"products": [item]}), encoding="utf-8")
    inp = source("b2b/products.json", FileType.JSON, SourceType.B2B).model_copy(update={"path": path})
    document = DocumentRouter().load(inp)[0]
    assert "Supplier: Harbor Works" in document.content
    assert "Category: Pumps" in document.content
    assert "HS: 841370" in document.content
    assert document.content.count("Marine Pump") == 1
    assert document.content.count("PUMP-200") == 1


def test_b2b_short_identifiers_are_not_suppressed_by_narrative_substrings(tmp_path):
    from trade_agent.data.router import DocumentRouter

    item = {
        "product_id": "P-SHORT", "product_name": "Pump", "category": "Tools", "supplier": "US",
        "sku": "AI", "hs_code": "841370", "url": "https://marketplace.example/products/P-SHORT",
        "reference_claim_id": "CLAIM-P-SHORT",
        "description": "Business details for global buyers mention Pump.",
    }
    path = tmp_path / "products.json"
    path.write_text(json.dumps({"products": [item]}), encoding="utf-8")
    inp = source("b2b/products.json", FileType.JSON, SourceType.B2B).model_copy(update={"path": path})
    content = DocumentRouter().load(inp)[0].content
    assert "Supplier: US" in content
    assert "SKU: AI" in content
    assert content.count("Pump") == 1


def test_b2b_without_narrative_preserves_facts_but_removes_evaluation_labels(tmp_path):
    from trade_agent.data.router import DocumentRouter

    item = {
        "product_id": "P-3", "product_name": "Pump", "supplier": "Harbor Works",
        "sku": "PUMP-300", "hs_code": "841370", "url": "https://marketplace.example/products/P-3",
        "reference_claim_id": "CLAIM-P-3", "extra_evidence": {"minimum_order": 5},
    }
    path = tmp_path / "products.json"
    path.write_text(json.dumps({"products": [item]}), encoding="utf-8")
    inp = source("b2b/products.json", FileType.JSON, SourceType.B2B).model_copy(update={"path": path})
    document = DocumentRouter().load(inp)[0]
    expected = {key: value for key, value in item.items() if key != "reference_claim_id"}
    assert document.content == json.dumps(expected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize("kind", ["b2b", "news", "social"])
def test_duplicate_natural_ids_fail_closed_for_multi_record_routes(tmp_path, kind):
    from trade_agent.data.router import DocumentRouter

    if kind == "b2b":
        record = {
            "product_id": "P-DUP", "product_name": "Pump", "supplier": "Supplier", "sku": "SKU-DUP",
            "hs_code": "841370", "url": "https://marketplace.example/p", "reference_claim_id": "CLAIM-DUP",
        }
        body = json.dumps({"products": [record, {**record, "product_name": "Other"}]})
        file_type, source_type, fixture = FileType.JSON, SourceType.B2B, "b2b/products.json"
        suffix = ".json"
    elif kind == "news":
        record = {
            "id": "NEWS-DUP", "headline": "Headline", "body": "Evidence", "published_at": NOW.isoformat(),
            "url": "https://newsroom.example/dup", "reference_claim_id": "CLAIM-DUP",
        }
        body = json.dumps({"stories": [record, {**record, "headline": "Other"}]})
        file_type, source_type, fixture = FileType.JSON, SourceType.INDUSTRY_NEWS, "news/stories.json"
        suffix = ".json"
    else:
        record = {
            "post_id": "POST-DUP", "text": "Evidence", "company": "Supplier", "published_at": NOW.isoformat(),
            "url": "https://social.example/dup", "reference_claim_id": "CLAIM-DUP",
        }
        body = "\n".join((json.dumps(record), json.dumps({**record, "text": "Other"})))
        file_type, source_type, fixture = FileType.JSONL, SourceType.SOCIAL, "social/posts.jsonl"
        suffix = ".jsonl"
    path = tmp_path / f"duplicate{suffix}"
    path.write_text(body, encoding="utf-8")
    inp = source(fixture, file_type, source_type).model_copy(update={"path": path})
    router = DocumentRouter()
    assert router.load(inp) == []
    assert [item.error_code for item in router.quarantines] == ["INVALID_DOCUMENT_SHAPE"]


def test_source_truth_fields_and_profile_grain_reach_strict_chunk_metadata():
    from trade_agent.data.chunkers import ChunkRouter
    from trade_agent.data.router import DocumentRouter

    inp = source(
        "customs_profiles/001-010121-2025-03.json", FileType.GENERATED_PROFILE, SourceType.CUSTOMS_PROFILE,
        publish_time=NOW, valid_from=NOW, valid_to=NOW, license_scope="synthetic-demo",
        manifest_attributes={"fact_type": "trade_activity", "reference_claim_ids": ["CLAIM-CUSTOMS-001"]},
    )
    document = DocumentRouter().load(inp)[0]
    chunk = ChunkRouter().chunk(document)[0]
    assert chunk.metadata.publish_time == NOW
    assert chunk.metadata.valid_from == NOW and chunk.metadata.valid_to == NOW
    assert chunk.metadata.license_scope == "synthetic-demo"
    assert chunk.metadata.fact_type.value == "trade_activity"
    assert chunk.metadata.company_name == "Harbor CN Imports 01"
    assert chunk.metadata.country_code == "CN"
    assert chunk.metadata.hs_code == "010121"
    assert chunk.metadata.aggregation_info["aggregation_grain"] == "company_country_hs_calendar_month"
    raw = chunk.metadata.source_locator.raw
    assert raw["calendar_month"] == "2025-03" and raw["source_record_count"] == 2


def test_quarantine_record_validates_nonblank_and_sanitizer_removes_paths_and_all_url_values(tmp_path):
    from trade_agent.data.quarantine import QuarantineRecord, sanitize_diagnostic

    source_path = str((tmp_path / "private" / "source.json").absolute())
    diagnostic = f"failed {source_path} and {source_path} password: hunter2 api_key=secret https://user:pass@example.test/a?foo=bar&token=abc"
    sanitized = sanitize_diagnostic(diagnostic, source_path=source_path)
    assert source_path not in sanitized
    for secret in ("hunter2", "api_key=secret", "user:pass", "foo=bar", "token=abc"):
        assert secret not in sanitized
    for field in ("error_code", "source_path", "parser", "diagnostic"):
        values = {"error_code": "E", "source_path": source_path, "parser": "router", "diagnostic": "safe"}
        values[field] = "   "
        with pytest.raises((TypeError, ValueError, ValidationError)):
            QuarantineRecord(**values)
    quoted = sanitize_diagnostic('{"password":"hunter2","api_key":"secret"} Authorization: Bearer bearer-token')
    assert "hunter2" not in quoted and "secret" not in quoted and "bearer-token" not in quoted


def test_direct_router_quarantine_uses_deterministic_safe_absolute_fallback(tmp_path):
    from trade_agent.data.router import DocumentRouter

    path = tmp_path / "a" / "b" / "x.html"
    path.parent.mkdir(parents=True)
    path.write_text("not html", encoding="utf-8")
    inp = source("website/section-01.html", FileType.HTML, SourceType.OFFICIAL_WEBSITE).model_copy(update={"path": path})
    first_router = DocumentRouter()
    assert first_router.load(inp) == []
    router = DocumentRouter()
    router.load(inp)
    reported = router.quarantines[-1].source_path
    assert "/Users" not in reported and reported == first_router.quarantines[-1].source_path
    assert reported.endswith("/x.html")


def test_router_quarantine_prefers_explicit_safe_display_path(tmp_path):
    from trade_agent.data.router import DocumentRouter

    path = tmp_path / "x.html"
    path.write_text("not html", encoding="utf-8")
    inp = source("website/section-01.html", FileType.HTML, SourceType.OFFICIAL_WEBSITE).model_copy(update={"path": path, "display_path": "a/b/x.html"})
    DocumentRouter().load(inp)
    router = DocumentRouter()
    router.load(inp)
    assert router.quarantines[-1].source_path == "a/b/x.html"
