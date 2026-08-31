# Task 7 implementation report

## Delivered

- Added a deterministic `SourceCatalog` and `IngestionPipeline` that only
  orchestrate the reviewed `DocumentRouter` and `ChunkRouter`.
- Catalog expansion is sorted, rooted under the corpus root, rejects symlink
  sources, and records unsupported paths, traversal, missing globs, duplicate
  paths, unreadable sources, manifest failures, parser quarantines, and chunk
  failures explicitly.
- Each catalog source is joined to the frozen corpus manifest; source bytes are
  size-guarded and streamed into SHA-256 before the manifest hash is trusted.
- `BuildManifest` is strict and frozen. Source collections, counts, quarantine
  records, and backend state are immutable. Documents/chunks use canonical
  immutable snapshots with lossless `restore()` methods for later indexing.
- Build fingerprints include normalized catalog data, actual source hashes,
  router/chunker versions and settings, metadata schema version, and backend
  degradation; they exclude output location and volatile fetched/ingested times.
- Output uses a same-directory temporary file, `fsync`, and `os.replace`, and
  refuses unsafe symlink targets/ancestors. The CLI emits only a concise JSON
  summary. Cookie/session/signature assignment diagnostics are redacted.

## TDD evidence

RED command:

```text
$ uv run pytest tests/unit/test_manifest.py tests/integration/test_ingestion_pipeline.py -q
5 failed in 0.04s
```

## Review-fix round 3

The review regressions covered exact canonical snapshot payloads, bounded and
symlink-safe frozen-manifest loading, truthful scanned-PDF quarantine backend
state, deep relative quarantine locators, copied-root/catalog determinism,
output-ancestor safety, injected chunk failure persistence, and document/chunk
metadata and locator cross-invariants.

Test-first RED evidence:

```text
$ uv run pytest tests/unit/test_manifest.py tests/integration/test_ingestion_pipeline.py tests/unit/test_data_router.py -q -k 'snapshot or frozen_manifest or deep_relative or chunk_failure or direct_router_quarantine or explicit_safe_display or locator_not_in_document'
3 failed, 8 passed, 93 deselected in 0.13s
```

The three intended failures were acceptance of reordered/pretty snapshot JSON,
acceptance of a tampered chunk locator, and rejection of the new explicit
display-path field as an extra input. Initial fixture-only errors were corrected
before recording this RED result.

GREEN and final verification:

```text
$ uv run pytest tests/unit/test_manifest.py tests/integration/test_ingestion_pipeline.py tests/unit/test_data_router.py -q -k 'snapshot or frozen_manifest or deep_relative or chunk_failure or direct_router_quarantine or explicit_safe_display or locator_not_in_document'
11 passed, 93 deselected in 0.16s

$ uv run pytest tests/unit/test_manifest.py tests/integration/test_ingestion_pipeline.py -q
23 passed in 0.56s

$ uv run pytest tests/unit/test_manifest.py tests/integration/test_ingestion_pipeline.py tests/unit/test_data_router.py tests/unit/test_chunkers.py tests/integration/test_pdf_pipeline.py tests/integration/test_corpus_routing.py -q
146 passed, 5 third-party SWIG deprecation warnings in 0.80s

$ uv run pytest tests/unit -q
156 passed, 5 third-party SWIG deprecation warnings in 2.30s

$ uv run pytest tests/integration/test_corpus_routing.py tests/integration/test_pdf_pipeline.py tests/integration/test_ingestion_pipeline.py -q
35 passed, 5 third-party SWIG deprecation warnings in 0.66s

$ uv run python -m compileall -q trade_agent scripts tests
$ uv lock --check
Resolved 49 packages in 2ms
$ git diff --check
passed
```

The persisted demo regression remains exactly `74` sources, `116` documents,
and `120` chunks with one scanned-PDF quarantine. Its source records
`parser_backend=pymupdf` and `degraded=true`; parser backend state includes
`mineru_statuses=["unavailable"]` and `mineru_unavailable`, with no `/Users/`
or repository-root paths in the JSON.

The failures were missing `manifest`/`pipeline` modules and unredacted
`cookie`, `session_id`, `signature`, and `sig` values.

GREEN command:

```text
$ uv run pytest tests/unit/test_manifest.py tests/integration/test_ingestion_pipeline.py -q
7 passed in 0.38s
```

The focused cases cover deterministic output, unsupported type accounting,
hash mismatch, symlink output refusal, traversal/missing-glob quarantine,
snapshot restoration, full corpus CLI build, and sanitizer regression.

## Verification

```text
$ python -m scripts.ingest_trade_sources ... --output /tmp/task7-a.json
build_9fe6df4a869783894a1d721fa0c35e5c; 74 sources, 116 documents,
120 chunks, 1 explicit scanned-PDF quarantine, metadata_complete=true

$ cmp -s /tmp/task7-a.json /tmp/task7-b.json
deterministic-output-ok

$ docker compose run --rm --no-deps api pytest tests/unit/test_manifest.py tests/integration/test_ingestion_pipeline.py -q
7 passed in 0.73s

$ uv run pytest tests/unit tests/integration/test_ingestion_pipeline.py -q
155 passed, 5 expected third-party SWIG deprecation warnings

$ uv run pytest -q
173 passed, 5 failed: existing live-MySQL tests refuse absent non-placeholder
MYSQL migration/query credentials; Task 7 does not change that boundary.

$ python -m compileall -q trade_agent scripts tests
$ uv lock --check
Resolved 49 packages in 2ms
$ git diff --check
passed
```

Targeted secret/path scans found only the deliberate sanitizer fixture values;
no host paths or credential assignments appear in production Task 7 files.

## Degradation truth

MinerU is unavailable in the exercised environment. The pipeline records the
reviewed router's exact degradation and retains the scanned regulator PDF as
an explicit quarantine; it does not claim OCR succeeded.

## Controller follow-up hardening

The persisted-manifest boundary now verifies every snapshot against its
payload, derives the fingerprint from the restored records with only volatile
timestamps removed, and requires the build ID to equal that fingerprint hash.
It rejects changed IDs, content, payloads, counts, completeness flags, and
duplicate IDs on load. Catalog normalization sorts rules and set-like fields
and replaces every physical root with the stable `corpus-root` token, while
duplicate detection remains before normalization.

The source hash remains streaming and bounded; too-large sources now become
per-source `input_too_large` quarantines. Atomic writing rejects dangling
targets, preserves the prior output if replacement fails, removes the temp
file, and accepts only verified macOS platform aliases under `/private`.

Follow-up GREEN:

```text
$ uv run pytest tests/unit/test_manifest.py tests/integration/test_ingestion_pipeline.py -q
10 passed in 0.31s
$ docker compose run --rm --no-deps api pytest tests/unit/test_manifest.py tests/integration/test_ingestion_pipeline.py -q
10 passed in 0.83s
$ uv run pytest tests/unit tests/integration/test_ingestion_pipeline.py -q
158 passed, 5 third-party SWIG deprecation warnings
```

## Bounded hardening round

The `fingerprint` field is now only a lowercase 64-hex SHA-256 digest of the
semantic payload, rather than a second copy of the manifest. It includes
backend/version/degradation state and metadata schema version, and validation
recomputes it before deriving the build ID. Strict numeric/count, source-state,
hash, tuple ordering, and schema-version checks reject coercion and tampering.

RED controller probes changed parser backend/schema/count values and were
accepted by `1a61a1e`. GREEN verification:

```text
$ uv run pytest tests/unit/test_manifest.py tests/integration/test_ingestion_pipeline.py -q
10 passed in 0.37s
$ docker compose run --rm --no-deps api pytest tests/unit/test_manifest.py tests/integration/test_ingestion_pipeline.py -q
10 passed in 0.79s
$ uv run pytest tests/unit -q
150 passed, 5 third-party SWIG deprecation warnings
```

## Reload follow-up

Optional `parser_backend` validation is now null-safe, so quarantined source
records (including unsupported files) round-trip through `BuildManifest`
without an `AttributeError`. Source IDs, source/file types, hashes, states,
and count entry naming/order are validated strictly.

```text
$ uv run pytest tests/unit/test_manifest.py tests/integration/test_ingestion_pipeline.py -q
10 passed in 0.36s
$ docker compose run --rm --no-deps api pytest tests/unit/test_manifest.py tests/integration/test_ingestion_pipeline.py -q
10 passed in 0.73s
$ uv run pytest tests/unit -q
150 passed, 5 third-party SWIG deprecation warnings
```

## Review-fix round 1

Manifest model copies now revalidate, duplicate catalog paths collapse to one
fail-closed quarantine outcome, and restored documents/chunks are checked
against source IDs and types. Quarantine paths/diagnostics are schema-safe;
router reporting keeps a safe relative locator.

## Review-fix round 2 P1

Frozen record source/file types are checked before routing, chunk snapshots
inherit comparable document metadata, quarantine copies revalidate and require
canonical sanitizer output, and backend/schema invariants are strict.

```text
$ uv run pytest tests/unit/test_manifest.py tests/integration/test_ingestion_pipeline.py tests/unit/test_data_router.py -q
94 passed, 5 third-party SWIG deprecation warnings
```
