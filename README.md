# Trade Intelligence Agent

An evidence-first foreign-trade intelligence agent. The foundation is under
construction and currently covers only the local data/services contract.

Every company, trade record, website, B2B listing, news item, social post, PDF,
customs profile, label, and metric in this repository is fictional synthetic
data. It is not customer, company, customs, or production data.

## Safe local foundation run

The example file is a field checklist only; do not source it as shell code.
Copy it for Compose variable names, edit non-secret host/model fields, and do
not commit `.env`; secrets below are still entered interactively:

```bash
cp .env.example .env
$EDITOR .env
read -r -s -p 'MySQL root password: ' MYSQL__ROOT_PASSWORD; printf '\n' >&2
read -r -s -p 'MySQL migration password: ' MYSQL__MIGRATION_PASSWORD; printf '\n' >&2
read -r -s -p 'MySQL query password: ' MYSQL__QUERY_PASSWORD; printf '\n' >&2
read -r -s -p 'MinIO root password: ' MINIO_ROOT_PASSWORD; printf '\n' >&2
export MYSQL__ROOT_PASSWORD MYSQL__MIGRATION_PASSWORD MYSQL__QUERY_PASSWORD MINIO_ROOT_PASSWORD
```

Start exactly the five foundation dependencies (ports are loopback-only):

```bash
docker compose --env-file .env up -d mysql etcd minio milvus redis --wait
```

Run migration and seed as one-shot commands. The migration secret is injected
only into these transient containers; it is not part of the persistent API
environment, which receives only the query role password:

```bash
docker compose --env-file .env run --rm --no-deps \
  -e MYSQL__MIGRATION_PASSWORD api \
  python -m trade_agent.db.migrate
docker compose --env-file .env run --rm --no-deps \
  -e MYSQL__MIGRATION_PASSWORD api \
  python -m trade_agent.db.seed --seed 20260830
```

Rebuild the API image, then run the read-only foundation smoke command:

```bash
docker compose --env-file .env build api
docker compose --env-file .env run --rm --no-deps api python -m scripts.smoke_foundation
```

Run the complete foundation test command:

```bash
docker compose --env-file .env run --rm --no-deps \
  -e MYSQL__MIGRATION_PASSWORD api \
  pytest tests/unit tests/contract/test_compose_contract.py \
  tests/integration/test_mysql_schema.py tests/integration/test_ingestion_pipeline.py \
  tests/integration/test_pdf_pipeline.py -q
```

Tear down services when finished (named volumes are retained unless removed
explicitly):

```bash
docker compose --env-file .env down

unset MYSQL__ROOT_PASSWORD MYSQL__MIGRATION_PASSWORD MYSQL__QUERY_PASSWORD MINIO_ROOT_PASSWORD
```

The MySQL named-volume users and passwords are applied only during the
volume's first initialization; changing environment variables does not rotate
credentials for an existing database. To replace local credentials, first
understand that `docker compose down -v` deletes all local synthetic data and
named volumes, then run the startup, migration, and seed commands again.

The scanned regulator PDF is intentionally expected to be quarantined when
MinerU is unavailable; the smoke output records this degradation truthfully.
This task verifies only foundation services, schema/seed, synthetic corpus,
ingestion, and manifests. Retrieval, SQL-agent orchestration, answer
generation, and evaluation are later scopes.
