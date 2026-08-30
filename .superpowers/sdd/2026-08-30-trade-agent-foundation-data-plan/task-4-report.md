# Task 4 SDD Report — Canonical Trade Document Contracts

## RED

Added `tests/unit/test_source_schemas.py` before production schema code. The
focused test initially failed during collection with
`ModuleNotFoundError: No module named 'trade_agent.schemas'`, confirming the
tests exercised the missing Task 4 contract rather than existing behavior.

## GREEN

Implemented strict Pydantic v2 contracts in `trade_agent/schemas/source.py`,
with locator-only re-exporting in `trade_agent/schemas/evidence.py` and public
exports in `trade_agent/schemas/__init__.py`. The focused suite passes:

```
9 passed
```

The full local suite reports 25 passing tests and 5 unrelated MySQL integration
failures because this environment has no non-placeholder migration/query
credentials. `git diff --check` and Python 3.12 `compileall` pass.

## Self-review

- Required enum strings and the justified `trade_ledger` source type are stable.
- All models forbid extras and require explicit `is_synthetic` truth fields.
- IDs/content are nonblank; content and parent hashes are exact 64-hex SHA-256;
  chunk/document hashes are checked against their content.
- `stable_id` uses length-prefixed UTF-8 canonical JSON, avoiding delimiter
  collisions and remaining deterministic for Unicode.
- HS codes use strict strings, preserve leading zeroes, and reject numerics;
  country codes normalize to ISO alpha-2 uppercase.
- Datetimes require timezone awareness and validity ranges are ordered.
- Synthetic source URLs are constrained to reserved `example.com`/
  `synthetic.example` hosts while non-synthetic public URLs remain allowed.
- `SourceLocator` supports page/block/table/section/post/row/profile/SQL/raw
  forms and serializes through Pydantic JSON mode.
- `DocumentRecord` exposes structured `units` and `attributes` without
  parser-specific top-level fields.

## Concerns / verification limits

- The requested Docker Compose cached/no-pull API build could not start because
  Compose interpolation requires local secrets that are intentionally absent;
  no credentials or `.env` were created.
- MySQL integration tests remain pending until a configured test database is
  available; no service was left running.

## Verification follow-up

Using process-local ephemeral Compose interpolation values (not persisted or
printed), the cached/no-pull API image rebuilt successfully. Inside the image:

```
Python 3.12.14
focused schema tests: 9 passed
full unit suite: 25 passed
python -m compileall -q trade_agent tests: passed
```

`docker compose ps -a` with the same redacted process-local setup returned no
containers. No product-code changes were needed after the initial commit.

## Review fix round 1

Addressed the contract review findings:

- `SourceLocator.post_id` is the stable Task 6 field; locators now require a
  meaningful concrete component and support structured JSON `raw` values.
- Structured units, attributes, aggregation info, and locator raw values use
  Pydantic's recursive `JsonValue` plus finite/JSON validation.
- Synthetic URL policy checks both primary and canonical URLs on all URL-bearing
  records, accepting normalized `.example` and `example.com`/`.org`/`.net`
  roots/subdomains while rejecting real or invalid hosts.
- Strict base `model_copy(update=...)` rebuilds through `model_validate`, so
  nested hashes, content, URL, truth, date-range, locator, and JSON invariants
  cannot be bypassed.
- Tests omit every required `ChunkMetadata` field individually and cover all
  direct negative probes.

Verification after the fix:

```
Host Python 3.12.14: focused 13 passed; full unit 29 passed
API image Python 3.12.14: focused 13 passed; full unit 29 passed
compileall, diff check, secret/absolute-path scan: passed
docker compose ps -a: no containers
```

## Review fix round 2

Closed the remaining strictness gaps:

- Numeric locator fields use strict integers with bounds; source weight and OCR
  confidence accept real Python ints/floats only, reject strings, booleans,
  non-finite values, and retain inclusive `[0, 1]` boundaries.
- Raw locator values are meaningful only when nonblank/nonempty recursively;
  valid structured raw locators remain supported.

Verification after the fix:

```
Host Python 3.12.14: focused 14 passed; full unit 30 passed
API image Python 3.12.14: focused 14 passed; full unit 30 passed
compileall, diff check, secret/absolute-path scan: passed
docker compose ps -a: no containers
```
