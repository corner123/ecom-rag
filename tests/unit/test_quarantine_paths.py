from __future__ import annotations

import pytest
from pydantic import ValidationError

from trade_agent.data.quarantine import QuarantineRecord


@pytest.mark.parametrize("source_path", [r"C:\\corpus\\file.pdf", r"C:/corpus/file.pdf", r"\\\\server\\share\\file.pdf", "nested\\file.pdf"])
def test_quarantine_source_path_rejects_windows_or_backslash_locators(source_path: str) -> None:
    with pytest.raises(ValidationError):
        QuarantineRecord(error_code="parse_failed", source_path=source_path, parser="test", diagnostic="safe")


def test_quarantine_source_path_accepts_canonical_posix_relative_locator() -> None:
    record = QuarantineRecord(error_code="parse_failed", source_path="pdf/scanned.pdf", parser="test", diagnostic="safe")
    assert record.source_path == "pdf/scanned.pdf"
