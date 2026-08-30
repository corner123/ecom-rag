# Task 6 implementation report

## Delivered

- Added fail-closed `SourceInput` and `DocumentRouter` contracts with separate physical
  file routing and business source routing, source/file compatibility checks, byte/count
  limits, deterministic document IDs, and inspectable `QuarantineRecord` entries.
- Parses the checked-in HTML/Markdown, B2B JSON, news JSON, social JSONL, generated
  customs profiles, text PDFs and image-only scanned PDF fixtures.  The corpus probe
  deterministically produces 117 documents and exactly one
  `SCANNED_PDF_OCR_UNAVAILABLE` quarantine; no ledger source is routed.
- Added `MinerUAdapter`: argv-only `subprocess.run(..., shell=False)`, time-bounded
  isolated temporary output, structured content-list v2/content-list before middle JSON
  before Markdown, with 0-based MinerU page indices normalized to 1-based locators.
  Absent/failing MinerU is never represented as a successful MinerU run; text PDFs use
  PyMuPDF degradation and empty scans quarantine.
- Added source-aware chunkers, configurable 500/64 default sizing, deterministic fallback
  token counting that preserves alphanumeric identifiers, metadata propagation through the
  strict Task 4 models, title injection for news, table/page isolation for PDFs, and
  natural product/profile/post boundaries.

## TDD evidence

RED: `uv run pytest tests/unit/test_data_router.py tests/unit/test_chunkers.py tests/integration/test_pdf_pipeline.py -q`
failed with nine expected `ModuleNotFoundError` failures for missing Task 6 modules.

GREEN: the same command passed with `10 passed` after implementation and the manifest
probe regression.

## Verification

- Focused router/chunker/PDF suite: `10 passed`.
- Full host unit suite: `48 passed`.
- `git diff --check`: exit 0.

## Residual capability risk

MinerU is optional and was not installed on this machine.  The adapter is tested with an
injected runner and does not claim invocation when the executable is absent.  In that
state text PDFs use PyMuPDF and image-only PDFs quarantine instead of inventing OCR text.
