# Trade Intelligence Agent

An evidence-first foreign-trade intelligence agent. The foundation is under
construction and currently covers only the local data/services contract.

Every company, trade record, website, B2B listing, news item, social post, PDF,
customs profile, label, and metric in this repository is fictional synthetic
data. It is not customer, company, customs, or production data.

## Safe local foundation run

Copy the example environment and replace every `replace-with-*` value with a
process-local secret. Do not commit `.env`:

```bash
cp .env.example .env
$EDITOR .env
# Export only for this shell session so one-shot -e arguments can read it.
set -a; . ./.env; set +a
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
  -e MYSQL__MIGRATION_PASSWORD="$MYSQL__MIGRATION_PASSWORD" api \
  python -m trade_agent.db.migrate
docker compose --env-file .env run --rm --no-deps \
  -e MYSQL__MIGRATION_PASSWORD="$MYSQL__MIGRATION_PASSWORD" api \
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
  -e MYSQL__MIGRATION_PASSWORD="$MYSQL__MIGRATION_PASSWORD" api \
  pytest tests/unit tests/contract/test_compose_contract.py \
  tests/integration/test_mysql_schema.py tests/integration/test_ingestion_pipeline.py \
  tests/integration/test_pdf_pipeline.py -q
```

Tear down services when finished (named volumes are retained unless removed
explicitly):

```bash
docker compose --env-file .env down
```

The scanned regulator PDF is intentionally expected to be quarantined when
MinerU is unavailable; the smoke output records this degradation truthfully.
This task verifies only foundation services, schema/seed, synthetic corpus,
ingestion, and manifests. Retrieval, SQL-agent orchestration, answer
generation, and evaluation are later scopes.
