# Task 6 implementation report

## Delivered contract

- `SourceInput` revalidates copies, requires nonblank strict fields, aware timestamps,
  ordered validity ranges, JSON-safe attributes, and reserved synthetic URLs.
  `DocumentRouter` requires positive integer byte/document limits, checks `stat()` before
  a bounded `max_bytes + 1` read, and verifies extension, `mimetypes`, PDF/HTML content
  signatures, and the explicit `source_type × file_type` matrix.
- Known failures have typed quarantine codes: file mismatch, malformed JSON, invalid
  business shape, UTF-8 decode failure, byte/count limits, malformed PDF, unreadable
  source, unsupported source/file pair, and scan OCR unavailability. Known bad inputs do
  not silently become text and do not fall through to generic `PARSE_FAILED`.
- HTML parsing is a one-pass evidence traversal. Inline descendants appear once; scripts,
  styles, templates, and noscript content are removed; direct section text, paragraphs,
  lists, tables, and asides are grouped under clean `h1 > h2 > h3` paths. A synthetic
  preamble/aside is attached to the first evidence section. Markdown uses the equivalent
  hierarchy without a file-title prefix. Both enforce document limits while grouping.
- B2B products, news stories, social posts, and customs profiles validate their required
  identifiers, text, URLs, times, HS/country/month/grain/count fields, and concrete types.
  Empty feeds are invalid. JSON and JSONL loops stop at the count limit; an oversized news
  feed rejects by count before inspecting later items.
- Batch feed URLs and per-item citation/canonical URLs are kept separately in document
  attributes and each locator's raw provenance. News mirrors resolve
  `canonical_story_id` to the original story URL. Product/story/post/profile IDs, claim
  IDs, canonical/syndication/dedupe relations, company/country/HS/month/grain/count, and
  manifest locators survive into every chunk. Source times, license, fact type, and OCR
  confidence populate strict `ChunkMetadata`.
- Chunking follows paragraph, sentence, then safe token boundaries while preserving the
  original slice punctuation and whitespace. The configured counter measures the full
  prefix plus slice for every maximum decision and measures overlap in the same units.
  Headings are inside the website/news budget. HS, SKU, mixed alphanumeric, hyphen,
  underscore, dot, slash, and URL identifiers are atomic in both chunks and overlap; an
  identifier that cannot fit raises an explicit error. Long B2B/social/profile records
  use the same maximum.
- Every PDF uses layout chunks without crossing a unit/page. Page, block, table, raw bbox,
  provenance, and confidence are retained. Industry-news PDF chunks inject the document
  heading inside the budget. Chunk IDs and content/parent hashes are deterministic.
- `MinerUAdapter` invokes argv with `shell=False`, a timeout, captured output, and an
  isolated temporary directory. Candidate precedence is content-list v2, content-list
  v1, middle JSON, then degraded Markdown. Each candidate is isolated, so malformed v2
  falls through to usable v1. Official 0-based `page_idx`, normalized bbox, text level,
  table caption/body/HTML, nested para-block/line/span/table fields, and confidence are
  validated. Invalid page/bbox/confidence cannot escape. Status distinguishes unavailable,
  timeout, execution error, nonzero exit, no output, malformed output, no usable output,
  structured success, and Markdown degradation.
- PyMuPDF always closes the document and records the exact MinerU failure component.
  Empty image-only scans quarantine with that exact status in the diagnostic.
- `QuarantineRecord` is frozen, strict, extra-forbidden, and nonblank. Diagnostics redact
  assignment/JSON/Bearer secrets, URL credentials, every URL query value, and repeated
  absolute source paths while retaining the explicit `source_path` field.

## Direct round-2 exploit reproduction

The inherited focused suite first returned `13 passed`, demonstrating that the old test
surface did not cover the reviewer attacks. Before repair, direct probes produced:

- HTML omitted direct section text; Markdown emitted paths such as
  `File title > Root > Child` instead of `Root > Child`.
- `beta_super-long-SKU-12345` was split into fragments including
  `beta_super-long-SKU-12` and `KU-12345`; overlap also emitted partial words.
- The injected nonadditive counter (`len(value) + 10` for nonempty input) produced a long
  sequence of two-character slices instead of applying the full prefix+slice budget.
- A malformed v2 MinerU `page_idx="oops"` raised `ValueError` out of `extract()` and
  prevented a valid v1 candidate from running.
- Sanitization retained the absolute source path twice and retained an arbitrary URL query
  value.

Test-first RED evidence for the expanded contracts:

```text
router regressions: 12 failed, 15 passed
chunk/PDF regressions: 11 failed, 15 passed
empty/MIME/MinerU/sanitizer edge regressions: 5 failed, 41 passed
final URL/count/comment strictness regressions: 3 failed, 30 passed
```

After repair, a dedicated 12-test exploit command passed every named HTML, Markdown,
dual-URL, sanitizer, protected-identifier, nonadditive-counter, exact-overlap, MinerU
fallthrough/malformed/status, corpus-wide, and repeat-determinism probe: `12 passed`.

## Corpus-wide evidence

- All `74` Task 5 manifest records were routed.
- Successful output is exactly `116` validated documents.
- The only quarantine is `pdf/scanned-regulator-notice.pdf` with
  `SCANNED_PDF_OCR_UNAVAILABLE` and `mineru_status=unavailable`.
- All successful documents produced `120` validated chunks across official website, B2B,
  industry news, social, regulator, and customs profile sources; no trade ledger document
  or chunk exists.
- Every chunk has a meaningful raw locator and strict JSON round-trip validation.
- Two complete routing/chunking runs produced identical document JSON, chunk IDs, chunk
  hashes/content, and quarantine records.

## Final verification evidence

Host Python `3.12.14`:

```text
focused Task 6 + corpus suite: 69 passed
all unit tests: 91 passed
named round-2 exploit suite: 12 passed
compileall: passed
uv lock --check: passed (49 packages resolved)
git diff --check: passed
high-confidence secret scan: passed
absolute /Users and /home path scan: passed
```

Rebuilt API image, Python `3.12.14`:

```text
focused Task 6 + corpus suite: 69 passed
all unit tests: 90 passed, 1 skipped
```

The skip is the existing macOS `/var` system-alias test and is expected on Linux.

Real Task 3 MySQL lifecycle used randomly generated process-local
`MYSQL__ROOT_PASSWORD`, `MYSQL__MIGRATION_PASSWORD`, `MYSQL__QUERY_PASSWORD`, and
`MINIO_ROOT_PASSWORD`. No `.env` or committed default was used. Cached API/MySQL images
were rebuilt through the explicit Docker Desktop Compose plugin/socket; MySQL became
healthy; migration and seed each ran twice. Only the one-shot migrate/seed/test containers
received the migration secret. The live DDL/ORM/index/FK/unique-key/query-grant/rollback
suite returned `12 passed`, then Compose teardown completed.

## MinerU capability statement

The `mineru` executable is not installed on this host, so no real MinerU OCR success is
claimed. Fake-runner tests exercise argv safety, timeout/nonzero/no-output/malformed
statuses and every output precedence branch. In the actual host/image environment, text
PDFs degrade explicitly to PyMuPDF and the image-only scanned PDF quarantines rather than
inventing OCR text.
