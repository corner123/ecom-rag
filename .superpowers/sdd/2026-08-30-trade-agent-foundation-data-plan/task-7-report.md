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
