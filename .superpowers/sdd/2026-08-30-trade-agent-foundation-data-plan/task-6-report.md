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

## Review round 1 hardening

- Input byte limits are checked from `stat()` before any `read_bytes`; strict source
  timestamps/ranges and JSON-safe manifest attributes are validated at the boundary.
- Chunk splitting now uses the same injected counter for every boundary, overlap, and
  heading-inclusive maximum. It preserves original text slices rather than re-tokenizing
  into synthetic whitespace, and applies budgets to B2B/social/profile sources too.
- HTML section containers no longer produce duplicate child paragraph records; PDF sources
  select layout chunking irrespective of business source type. MinerU now records nonzero
  exit and no-usable-output status explicitly.

Review RED/GREEN:

```text
RED: injected len counter emitted one over-budget chunk; fallback chunks exceeded max.
GREEN: tests/unit/test_chunkers.py -> 5 passed; focused Task 6 suite -> 12 passed.
```

## Review round 2 hardening

- `parsers.py` now owns typed HTML/Markdown/JSON/JSONL parsing failures. JSONL stops
  before constructing more than the configured document limit and malformed/wrong-shape
  source payloads receive typed quarantine codes.
- Source timestamps and provenance remain validated at the edge and are serialized safely
  into document attributes for later strict chunk metadata validation.
- Quarantine diagnostics redact token/password/secret/API-key assignment values, URL
  credentials, and sensitive query values.

Round-2 exploit evidence: `uv run pytest tests/unit/test_data_router.py
tests/unit/test_chunkers.py tests/integration/test_pdf_pipeline.py -q` returned
`13 passed`; `uv run pytest tests/unit -q` returned `51 passed`.

## Residual capability risk

MinerU is optional and was not installed on this machine.  The adapter is tested with an
injected runner and does not claim invocation when the executable is absent.  In that
state text PDFs use PyMuPDF and image-only PDFs quarantine instead of inventing OCR text.
