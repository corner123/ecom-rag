from datetime import datetime, timezone
from pathlib import Path

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


def test_routes_checked_in_item_boundaries_and_preserves_claims():
    from trade_agent.data.router import DocumentRouter

    router = DocumentRouter()
    b2b = router.load(source("b2b/products.json", FileType.JSON, SourceType.B2B))
    news = router.load(source("news/stories.json", FileType.JSON, SourceType.INDUSTRY_NEWS))
    social = router.load(source("social/posts.jsonl", FileType.JSONL, SourceType.SOCIAL))
    profiles = [router.load(source(f"customs_profiles/{p.name}", FileType.GENERATED_PROFILE, SourceType.CUSTOMS_PROFILE))[0]
                for p in sorted((ROOT / "customs_profiles").glob("*.json"))]
    assert len(b2b) == 18
    assert b2b[0].attributes["reference_claim_id"] == "CLAIM-B2B-001"
    assert len(news) == 16
    assert news[12].attributes["canonical_story_id"] == "NEWS-001"
    assert len(social) == 12
    assert social[0].units[0]["locator"]["post_id"] == "POST-001"
    assert len(profiles) == 54
    assert profiles[0].attributes["aggregation_info"]["aggregation_grain"] == "company_country_hs_calendar_month"


def test_html_and_markdown_use_heading_sections():
    from trade_agent.data.router import DocumentRouter

    router = DocumentRouter()
    html = router.load(source("website/section-01.html", FileType.HTML, SourceType.OFFICIAL_WEBSITE))
    markdown = router.load(source("website/methodology.md", FileType.MARKDOWN, SourceType.OFFICIAL_WEBSITE))
    assert len(html) >= 1 and html[0].units[0]["locator"]["section"]
    assert len(markdown) >= 1 and markdown[0].units[0]["locator"]["section"]


def test_mismatched_or_malformed_input_quarantines_instead_of_text_fallback(tmp_path):
    from trade_agent.data.router import DocumentRouter

    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    router = DocumentRouter()
    inp = source("b2b/products.json", FileType.PDF, SourceType.B2B)
    assert router.load(inp) == []
    malformed = inp.model_copy(update={"path": bad, "file_type": FileType.JSON})
    assert router.load(malformed) == []
    assert [q.error_code for q in router.quarantines] == ["FILE_TYPE_MISMATCH", "FILE_TYPE_MISMATCH"]


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
    assert first["documents"] == 117
    assert first["quarantines"] == [{"error_code": "SCANNED_PDF_OCR_UNAVAILABLE", "path": "pdf/scanned-regulator-notice.pdf"}]
