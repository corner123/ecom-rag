from datetime import datetime, timezone
from pathlib import Path

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
