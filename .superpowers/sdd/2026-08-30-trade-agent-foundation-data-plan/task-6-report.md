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

## Controller exploit repair follow-up

The controller's post-repair probes exposed seven related boundary defects. Each was
reproduced directly before implementation:

- A B2B row containing `feed_source_url=https://evil.example`,
  `item_source_url=https://evil.example`, `source_weight=1`, and an invalid injected
  `fact_type` made the document cite the evil URL while its locator still named the
  trusted feed; chunk validation then failed on the attacker-selected fact type.
- Two B2B rows with the same `product_id` returned two documents with the same stable
  `document_id`; two social rows with the same `post_id` did the same. News alone already
  rejected duplicate story IDs.
- Punctuated and unpunctuated Chinese evidence raised `atomic token cannot fit within
  chunk budget`; there was no legal CJK boundary or codepoint fallback.
- B2B rows with a description normalized to the description alone, omitting supplier,
  category, SKU, and HS evidence. B2B HTML chunks omitted their section heading.
- `<section>We sell <strong>industrial chargers</strong> globally.</section>` produced
  `We sell\nglobally.` and dropped the inline evidence. A nested-only
  `<div><span>Nested evidence only</span></div>` quarantined as having no usable text.
- A fenced Markdown line `# not a heading` created fake section paths instead of staying
  inside the root section.
- A unique-token counter where both the prefix and the complete prefix-plus-body equaled
  the maximum raised `prefix consumes chunk budget` without evaluating the legal full
  candidate.

The test-first baseline for these probes was exactly `11 failed, 52 passed`. The repair
establishes explicit metadata ownership: the raw structured item remains inspectable as
`source_payload`, but catalog/system provenance, weight/version, fact type, time range,
license, aggregation, and confidence fields own their top-level values. The same contract
is tested across B2B, news, social, and generated profiles and through strict chunk
metadata. B2B/news/social batches prevalidate every row and natural ID before constructing
documents, so duplicates fail closed with `INVALID_DOCUMENT_SHAPE`.

HTML now consumes each block or coalesced inline flow exactly once in DOM order, and
Markdown tracks backtick/tilde fences. B2B narrative text composes supplier, product,
category, SKU, and HS fields, suppressing a label only when the exact value appears as a
standalone narrative value; `US` and `AI` regressions prove that incidental substrings in
`business`/`details` do not erase short identifiers. Rows without narrative retain the
full canonical JSON fallback. B2B HTML injects the section heading inside every chunk
budget.

CJK sentence/safe punctuation is recognized, and unpunctuated CJK prose uses a bounded
codepoint fallback plus same-unit overlap. Protected trade identifiers remain atomic in
both the cut and overlap paths. Prefix-only equality no longer rejects a candidate; every
decision evaluates the complete prefix-plus-slice counter result, while a genuinely new
over-budget token still fails explicitly.

## Final verification evidence (2026-08-31)

Fresh verification used the repository's existing `uv` virtual environment on host
Python `3.12.14`:

```text
controller/router/chunker regressions: 70 passed
handoff-only named contract slice: 28 passed, 42 deselected
focused Task 6 + corpus + PDF suite: 88 passed
all unit tests: 110 passed (5 third-party SWIG deprecation warnings)
corpus probe: 74 manifest records -> 116 documents -> 120 chunks; 1 explicit scanned-PDF quarantine; repeat deterministic
compileall: passed
uv lock --check: passed (49 packages resolved)
git diff --check: passed
high-confidence secret scan: passed
absolute /Users and /home path scan: passed
```

Docker Compose rendered successfully with four new process-local secrets and rebuilt the
current working-tree API image. Its combined unit, corpus-routing, and PDF suite returned
`127 passed, 1 skipped`; the skip is the expected macOS `/var` system-alias test on Linux.

A fresh real-MySQL lifecycle used randomly generated process-local
`MYSQL__ROOT_PASSWORD`, `MYSQL__MIGRATION_PASSWORD`, `MYSQL__QUERY_PASSWORD`, and
`MINIO_ROOT_PASSWORD`; no `.env` or committed default was used. MySQL became healthy;
migration and seed each succeeded twice in separate one-shot containers; and the combined
settings, seed, and live DDL/ORM/index/FK/unique-key/query-grant/rollback suite returned
`15 passed`. The dedicated Compose project was then removed with volumes. A post-teardown
audit found no project container, network, or volume and no listener on TCP 3306.

## MinerU capability statement

The `mineru` executable is not installed on this host, so no real MinerU OCR success is
claimed. Fake-runner tests exercise argv safety, timeout/nonzero/no-output/malformed
statuses and every output precedence branch. In the actual host/image environment, text
PDFs degrade explicitly to PyMuPDF and the image-only scanned PDF quarantines rather than
inventing OCR text.
