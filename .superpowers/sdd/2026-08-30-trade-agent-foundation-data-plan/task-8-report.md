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
$ docker compose run --rm --no-deps -e MYSQL__MIGRATION_PASSWORD api \
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

## Controller review remediation

The review follow-up added a safe argparse boundary: unknown arguments are
converted to one sorted JSON error object (`stage=arguments`) without usage,
argv, paths, or secret text. The validator and ingestion gate now both require
exactly 74 sources, 116 documents, and 120 chunks, with regression coverage for
1/119/121 and related drift values.

All smoke resources resolve from the module-derived repository root, so corpus
checks work from an arbitrary current directory. Corpus runs use
`TemporaryDirectory`, and the MySQL engine is disposed in `finally` on success
and failure. Compose etcd now explicitly uses `--data-dir=/etcd-data`, bound
to its `etcd_data:/etcd-data` volume.

README secret setup no longer sources `.env` or expands a secret in argv. It
uses interactive `read -r -s` exports and passes only the variable name to
one-shot `-e` options, followed by explicit `unset`. The README states that
named-volume credentials apply only on first initialization: changing an env
value does not rotate existing users, and `down -v` deletes all local synthetic
data before reinitialization.

Remediation focused verification:

```text
$ uv run pytest tests/unit/test_smoke_foundation.py tests/contract/test_compose_contract.py -q -m 'not integration'
15 passed, 1 deselected
```

## Final controller-review evidence

The complete non-integration host suite passed after remediation:

```text
$ uv run pytest tests/unit tests/contract -q -m 'not integration'
183 passed, 1 deselected, 5 warnings
```

The live checks used the unique Compose project `task8review20260901` with
process-local synthetic credentials. The full live foundation suite passed
after rebuilding the API image:

```text
$ docker compose -p task8review20260901 run --rm --no-deps -e MYSQL__MIGRATION_PASSWORD api \
  pytest tests/unit tests/contract/test_compose_contract.py tests/integration/test_mysql_schema.py \
  tests/integration/test_ingestion_pipeline.py tests/integration/test_pdf_pipeline.py -q
227 passed, 1 skipped in 20.07s
```

The final smoke after rebuild and after a dependency down/up persistence cycle
returned `status=ok`, `synthetic_only=true`, `mysql_tables=7`, row counts
`8/60/12/30/2/60/825`, `months=18`, corpus `74/116/120`, metadata completeness
`1.0`, quarantine `expected=1` and `unexpected=0`, with all five service
readiness values true. Live versions were MySQL 8.4.11, Milvus 2.6.22, etcd
3.5.25, Redis 7.4.11, and healthy MinIO. The smoke reported build ID
`build_95a0a4d094fb5c3ec2f67fe8092862d9`, schema fingerprint
`58c6fcf620e71d3f4ef9af35cf028c859fb350278b656081341728e7e343b799`, and seed
hash `05c9095448085f074338e3dec227aa9daa8c7423ff79aa7f75e3a542fbb7bbb5`.

The etcd service command now includes the real `--data-dir=/etcd-data` and its
`etcd_data:/etcd-data` volume is covered by the static contract. The unique
project was cleaned with `docker compose -p task8review20260901 down -v`; the
pre-existing `foreign-trade-agent_*` volumes were left untouched.

The follow-up unit test also proves that a MySQL connection failure disposes
the created engine, and the obsolete combined service probe was removed so
each service remains independently staged and redacted.

## Final P2 remediation evidence

The CLI help boundary was tested RED first: both `-h` and `--help` previously
returned argparse's help text with exit 0. The parser now disables argparse's
built-in help option, so help and unknown arguments produce exactly one sorted
JSON object on stderr (`status=error`, `stage=arguments`, `error=ArgumentError`)
with empty stdout and exit 2. POSIX, Windows drive-letter, and UNC/secret-like
unknown argument regression cases remain covered.

README secret setup now works in both Bash and zsh. Each secret is obtained by
`python3 -c` calling `getpass.getpass`; no read-based prompt flags, `.env` sourcing, secret
file write, or secret-valued argv expansion is used. Four non-empty parameter
assertions run before Compose commands, with `set -eu` making failures stop;
an EXIT trap clears the process-local values if Compose or any later command
fails. The optional editor invocation is guarded for shells with no `EDITOR`.
The contract test executes the setup in both shells with a mocked prompt and
checks that each exported value is the simulated value rather than the
`.env.example` placeholder, without printing that value.

```text
$ uv run pytest tests/unit tests/contract -q -m 'not integration'
188 passed, 1 deselected, 5 warnings

$ uv run python -m compileall -q trade_agent scripts tests
$ uv lock --check
Resolved 49 packages in 2ms
$ git diff --check
passed
```

Compose precedence was verified using `.env.example` as the `--env-file` and
process-local non-secret proof values in the shell: `docker compose config`
resolved the shell values for MySQL root/migration/query and API query fields,
and the check emitted only `precedence=ok`. This confirms exported secrets
override the field-list placeholders while remaining out of files and argv.
