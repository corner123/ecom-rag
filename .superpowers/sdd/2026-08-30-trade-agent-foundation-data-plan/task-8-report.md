# Task 8 implementation report

## Delivered

- Added `scripts/smoke_foundation.py` with a public
  `run_foundation_smoke() -> dict` API and a fail-closed
  `python -m scripts.smoke_foundation` CLI.
- Smoke verification uses only `trade_query`, performs real read-only MySQL
  schema/seed checks, probes Redis, Milvus, etcd, and MinIO semantically, and
  never depends on the Docker socket.
- Added deterministic temporary corpus regeneration and two-run ingestion
  checks against the checked-in corpus hashes, manifest reload, source/file
  type counts, synthetic flags, metadata completeness, and the expected single
  scanned-PDF quarantine.
- Added named Compose data volumes so one-shot migration/seed containers do not
  lose the foundation database on dependency recreation. API persistence still
  receives only the query secret.
- `Settings.load()` honors `MYSQL__QUERY_PASSWORD` and defaults to
  `trade_query`, while preserving explicit legacy `MYSQL__PASSWORD`/
  `MYSQL_APP_PASSWORD` compatibility. Root/migration values are not Settings
  fields.
- `QuarantineRecord.source_path` now rejects Windows drive-letter/UNC paths and
  every backslash on POSIX; cross-platform unit coverage was added.
- Expanded README with safe local env setup, five-dependency startup,
  transient-secret migration/seed, rebuild, smoke, full tests, and teardown.

## TDD evidence

Initial RED command:

```text
$ uv run pytest tests/unit/test_smoke_foundation.py tests/unit/test_quarantine_paths.py tests/contract/test_compose_contract.py -q
9 failed, 3 passed
```

The intended failures were the absent smoke module and acceptance of Windows/
backslash quarantine locators.

Focused GREEN command:

```text
$ env -u MYSQL__QUERY_PASSWORD -u MYSQL__PASSWORD -u MYSQL_APP_PASSWORD \
  uv run pytest tests/unit/test_settings.py tests/unit/test_smoke_foundation.py \
  tests/unit/test_quarantine_paths.py tests/contract/test_compose_contract.py \
  -q -m 'not integration'
19 passed, 1 deselected
```

## Live verification

Process-local, non-placeholder credentials were used; no `.env` was created or
committed. Real Compose dependencies started healthy. Migration and seed were
run as transient commands with the migration password only on those commands.

```text
$ docker compose run --rm --no-deps -e MYSQL__MIGRATION_PASSWORD=… api \
  pytest tests/unit tests/contract/test_compose_contract.py \
  tests/integration/test_mysql_schema.py tests/integration/test_ingestion_pipeline.py \
  tests/integration/test_pdf_pipeline.py -q
215 passed, 1 skipped

$ docker compose run --rm --no-deps api python -m scripts.smoke_foundation
status=ok
mysql_tables=7
row_counts=8/60/12/30/2/60/825; months=18
corpus=74 sources / 116 documents / 120 chunks
quarantine_expected=1; quarantine_unexpected=0; metadata_required_completeness=1.0
```

The smoke summary also reported live MySQL 8.4.11, Milvus 2.6.22, etcd 3.5.25,
Redis 7.4.11, and healthy MinIO; schema and seed fingerprints matched the
deterministic contracts. The scanned regulator PDF was quarantined as
`scanned_pdf_ocr_unavailable` with MinerU status `unavailable`, truthfully
recorded as degraded.

## Release checks

```text
$ uv run python -m compileall -q trade_agent scripts tests
$ uv lock --check
Resolved 49 packages in 2ms
$ git diff --check
passed
```

No retrieval, SQL-agent, Milvus collection/index, answer generation, or
evaluation behavior was added; those remain later scopes. The live stack may
be stopped with `docker compose down`; local named volumes are disposable
development state and are not repository artifacts.
