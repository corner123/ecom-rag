from datetime import datetime, timezone
from pathlib import Path

import pytest

from trade_agent.schemas.source import FileType, SourceType

ROOT = Path(__file__).parents[2] / "demo" / "trade_intel_seed"

def source(path, file_type, source_type):
    from trade_agent.data.router import SourceInput
    return SourceInput(path=ROOT / path, file_type=file_type, source_type=source_type,
        source_id=f"src-{path}", title=path, language="en", fetched_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
        is_synthetic=True, source_url="https://ingest.example/source")


def test_text_pdf_has_nonempty_page_units_and_scan_is_quarantined():
    from trade_agent.data.router import DocumentRouter

    router = DocumentRouter()
    text = router.load(source("pdf/regulator-notice.pdf", FileType.PDF, SourceType.REGULATOR))
    scan = router.load(source("pdf/scanned-regulator-notice.pdf", FileType.PDF, SourceType.REGULATOR))
    assert text and text[0].units and text[0].units[0]["locator"]["page"] == 1
    assert scan == []
    assert router.quarantines[-1].error_code == "SCANNED_PDF_OCR_UNAVAILABLE"


def test_mineru_structured_output_beats_markdown_and_never_uses_shell(tmp_path):
    from trade_agent.data.pdf import MinerUAdapter

    pdf = ROOT / "pdf" / "regulator-notice.pdf"
    calls = []
    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        out = Path(argv[argv.index("-o") + 1])
        (out / "x_content_list_v2.json").write_text('[{"type":"text","page_idx":0,"text":"structured","bbox":[0,0,1,1]}]')
        (out / "x.md").write_text("markdown")
        class Result: returncode = 0; stdout = ""; stderr = ""
        return Result()
    result = MinerUAdapter(runner=runner).extract(pdf)
    assert result.mode == "content_list" and result.units[0]["text"] == "structured"
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["timeout"] == 45


def test_malformed_v2_candidate_falls_through_to_valid_v1(tmp_path):
    from trade_agent.data.pdf import MinerUAdapter

    def runner(argv, **kwargs):
        out = Path(argv[argv.index("-o") + 1])
        (out / "a_content_list_v2.json").write_text('[{"page_idx":"bad","text":"must not escape"}]')
        (out / "b_content_list.json").write_text('[{"type":"text","page_idx":0,"text":"valid v1","bbox":[0,0,1000,1000]}]')
        class Result: returncode = 0; stdout = ""; stderr = ""
        return Result()

    adapter = MinerUAdapter(runner=runner)
    result = adapter.extract(ROOT / "pdf" / "regulator-notice.pdf")
    assert result is not None and result.units[0]["text"] == "valid v1"
    assert adapter.last_status == "structured_v1"


def test_content_list_parses_official_table_fields_and_rejects_invalid_geometry(tmp_path):
    from trade_agent.data.pdf import MinerUAdapter

    candidate = tmp_path / "content.json"
    candidate.write_text(
        """[
          {"type":"text","page_idx":0,"text":"Heading","text_level":1,"bbox":[0,0,1000,100],"score":0.98},
          {"type":"table","page_idx":1,"table_caption":["Monthly totals"],"table_body":"A | B\\n1 | 2","table_html":"<table></table>","bbox":[10,20,900,800],"confidence":0.87},
          {"type":"text","page_idx":-1,"text":"bad page","bbox":[0,0,1,1]},
          {"type":"text","page_idx":0,"text":"bad bbox","bbox":[0,0,1001,1]},
          {"type":"text","page_idx":0,"text":"bad confidence","bbox":[0,0,1,1],"score":2}
        ]""",
        encoding="utf-8",
    )
    units = MinerUAdapter._content_list(candidate)
    assert [unit["text"] for unit in units] == ["Heading", "Monthly totals\nA | B\n1 | 2"]
    assert units[0]["locator"] == {"page": 1, "block": 0, "raw": {"bbox": [0, 0, 1000, 100], "kind": "text", "text_level": 1}}
    assert units[1]["locator"]["page"] == 2
    assert units[1]["locator"]["table"] == "table-2-1"
    assert units[1]["confidence"] == 0.87


def test_middle_json_parses_nested_spans_and_tables(tmp_path):
    from trade_agent.data.pdf import MinerUAdapter

    candidate = tmp_path / "middle.json"
    candidate.write_text(
        """{"pdf_info":[{"page_idx":0,"para_blocks":[
          {"type":"text","bbox":[0,0,100,100],"lines":[{"spans":[{"content":"Nested"},{"text":" spans"}]}]},
          {"type":"table","bbox":[0,100,500,500],"table_caption":"Caption","table_body":"X | Y","score":0.75}
        ]}]}""",
        encoding="utf-8",
    )
    units = MinerUAdapter._middle(candidate)
    assert [unit["text"] for unit in units] == ["Nested spans", "Caption\nX | Y"]
    assert units[1]["locator"]["table"] == "table-1-1"
    assert all(unit["locator"]["page"] == 1 for unit in units)


@pytest.mark.parametrize(
    ("behavior", "expected_status"),
    [
        ("missing", "unavailable"),
        ("timeout", "timeout"),
        ("nonzero", "nonzero_exit"),
        ("empty", "no_output"),
        ("malformed", "malformed_output"),
    ],
)
def test_mineru_all_failure_statuses_are_truthful(behavior, expected_status):
    import subprocess
    from trade_agent.data.pdf import MinerUAdapter

    def runner(argv, **kwargs):
        out = Path(argv[argv.index("-o") + 1])
        if behavior == "missing":
            raise FileNotFoundError("mineru")
        if behavior == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if behavior == "malformed":
            (out / "x_content_list_v2.json").write_text("not-json")
        class Result:
            returncode = 7 if behavior == "nonzero" else 0
            stdout = ""
            stderr = "failed"
        return Result()

    adapter = MinerUAdapter(runner=runner)
    assert adapter.extract(ROOT / "pdf" / "regulator-notice.pdf") is None
    assert adapter.last_status == expected_status


def test_invalid_only_mineru_fields_are_reported_as_malformed_not_exception(tmp_path):
    from trade_agent.data.pdf import MinerUAdapter

    def runner(argv, **kwargs):
        out = Path(argv[argv.index("-o") + 1])
        (out / "x_content_list_v2.json").write_text('[{"page_idx":-1,"text":"bad","bbox":[0,0,1,1]}]')
        class Result: returncode = 0; stdout = ""; stderr = ""
        return Result()

    adapter = MinerUAdapter(runner=runner)
    assert adapter.extract(ROOT / "pdf" / "regulator-notice.pdf") is None
    assert adapter.last_status == "malformed_output"


def test_pdf_signature_with_invalid_body_has_typed_malformed_pdf_quarantine(tmp_path):
    from trade_agent.data.router import DocumentRouter

    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.7\nnot actually a PDF")
    router = DocumentRouter()
    inp = source("pdf/regulator-notice.pdf", FileType.PDF, SourceType.REGULATOR).model_copy(update={"path": path})
    assert router.load(inp) == []
    assert router.quarantines[-1].error_code == "MALFORMED_PDF"


def test_markdown_is_explicitly_degraded_only_after_all_structured_formats(tmp_path):
    from trade_agent.data.pdf import MinerUAdapter

    def runner(argv, **kwargs):
        out = Path(argv[argv.index("-o") + 1])
        (out / "x_content_list_v2.json").write_text("not-json")
        (out / "x_middle.json").write_text("{}")
        (out / "x.md").write_text("Markdown fallback evidence")
        class Result: returncode = 0; stdout = ""; stderr = ""
        return Result()

    adapter = MinerUAdapter(runner=runner)
    result = adapter.extract(ROOT / "pdf" / "regulator-notice.pdf")
    assert result is not None and result.mode == "markdown"
    assert result.degraded_components == ("structured_output_unavailable",)
    assert adapter.last_status == "markdown_degraded"


def test_mineru_failure_status_propagates_to_pymupdf_and_scan_quarantine():
    from trade_agent.data.pdf import MinerUAdapter
    from trade_agent.data.router import DocumentRouter

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("mineru")

    router = DocumentRouter(mineru=MinerUAdapter(runner=unavailable))
    text = router.load(source("pdf/regulator-notice.pdf", FileType.PDF, SourceType.REGULATOR))
    assert text[0].attributes["mineru_status"] == "unavailable"
    assert "mineru_unavailable" in text[0].attributes["degraded_components"]
    assert router.load(source("pdf/scanned-regulator-notice.pdf", FileType.PDF, SourceType.REGULATOR)) == []
    quarantine = router.quarantines[-1]
    assert quarantine.error_code == "SCANNED_PDF_OCR_UNAVAILABLE"
    assert "mineru_status=unavailable" in quarantine.diagnostic


def test_industry_news_pdf_chunks_keep_page_locator_and_title_budget():
    from trade_agent.data.chunkers import ChunkRouter
    from trade_agent.data.router import DocumentRouter

    document = DocumentRouter().load(source("pdf/market-bulletin.pdf", FileType.PDF, SourceType.INDUSTRY_NEWS))[0]
    chunks = ChunkRouter(max_tokens=100, overlap_tokens=10, token_counter=len).chunk(document)
    assert chunks
    assert all(chunk.metadata.source_locator.page >= 1 for chunk in chunks)
    assert all(chunk.content.startswith(document.title + "\n") for chunk in chunks)
    assert all(len(chunk.content) <= 100 for chunk in chunks)
