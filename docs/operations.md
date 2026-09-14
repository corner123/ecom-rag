# Operations

## Safe local checks

Repository and report verification do not require services:

```sh
.venv/bin/python -m pytest tests/security/test_repository_hygiene.py tests/contract/test_api.py -q
.venv/bin/python -m scripts.verify_repository
.venv/bin/python -m scripts.verify_trade_report --latest data/eval/trade_intel/reports_public --kind development
.venv/bin/python -m scripts.verify_trade_report --latest data/eval/trade_intel/reports_public --kind holdout
```

`scripts.verify_repository` reports repository/artifact state. It deliberately does not infer that MySQL, Redis, Milvus, BGE, or the API are running.
The local uv-managed `.venv` may omit the `pip` module. `python -m pip check` below is a controller gate inside the built API image, where pip is part of that environment.

## Secrets and service startup

Copy `.env.example` to an untracked `.env` and replace every placeholder locally. Use separate MySQL root, migration, and query passwords. Never place credentials on a command line, in a report, or in Git. The API receives only the query-role credential. Model artifacts live in the external model cache and are verified against pinned revisions and checksums.

The controller-final live gate starts the six Compose services and waits for health:

```sh
docker compose --env-file .env up -d --wait
docker compose --env-file .env exec api python -m pip check
docker compose --env-file .env exec api pytest -q
docker compose --env-file .env exec api python -m scripts.smoke_foundation
docker compose --env-file .env exec api python -m scripts.smoke_embeddings
docker compose --env-file .env exec api python -m scripts.smoke_milvus_roundtrip
docker compose --env-file .env exec api python -m scripts.verify_repository
```

These commands are acceptance gates to be executed and recorded by the controller. Their presence here is an operating procedure, not evidence that this Task 10 implementation turn ran them.

## Database lifecycle

Run migration and deterministic synthetic seed with the migration credential supplied only to the one-shot container. Rebuild or start the API afterward with the query role. Verify seven tables, foreign keys, unique constraints, indexes, seed cardinalities, schema fingerprint, and that the query account can select but cannot mutate.

Changing values in `.env` does not rotate accounts already stored in an existing MySQL volume. Deleting named volumes destroys local synthetic service state; do it only when deliberately rebuilding the environment.

## Index lifecycle

Ingest a reviewed catalog into a new manifest, then build a new immutable index ID. A valid publication requires:

- complete chunk metadata and stable chunk hashes;
- verified pinned BGE-M3 artifacts and a live production embedding contract;
- exact BM25 membership;
- Milvus create/insert/flush/load, unfiltered and filtered ANN, process reconnect, build/collection contract checks, and owned test-collection cleanup.

Do not substitute the local evaluation adapter for this gate. If reranking is unavailable, return the recorded degraded/error state instead of labeling the result as full reranking.

## API checks

`GET /health` is process liveness. `GET /ready` is the dependency/build contract and must fail when MySQL schema, Milvus collection/build, or Redis is unavailable. Exercise SQL-only, RAG-only, mixed SQL+RAG, conflict, insufficient-evidence, idempotent resume, retry-limit, and step-limit workflows. Inspect evidence IDs and safe locators rather than logging source bodies or SQL rows.

CLI evaluation delegates to the local synthetic runner:

```sh
trade-intel eval --dataset data/eval/trade_intel/dev_public.jsonl --output data/eval/private/dev-next
trade-intel verify-report --latest data/eval/trade_intel/reports_public --kind development
```

Outputs belong in ignored directories. The first private holdout is already consumed; do not issue another consume command against it.

## Shutdown and incident handling

Stop the stack without deleting volumes:

```sh
docker compose --env-file .env down
```

On a failure, retain machine-readable error codes, build/profile IDs, safe hashes, and stage status. Do not log passwords, tokens, private holdout labels, full SQL results, or unredacted source text. Deterministic policy/schema/evidence failures require input or configuration correction; retry only transient timeout, rate-limit, or connection failures within the configured budget.
