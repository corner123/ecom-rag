# Trade Agent Foundation and Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy application tree with a Python 3.12 foreign-trade project, a real local service stack, seven-table MySQL schema, deterministic synthetic corpus, source-aware parsing, chunking, metadata, and manifests.

**Architecture:** Docker Compose owns MySQL, Milvus dependencies, Redis, and the API runtime. Deterministic generators create explicitly synthetic relational and document data; a two-stage router resolves physical `file_type` and business `source_type`, then emits validated canonical records and source-specific chunks.

**Tech Stack:** Python 3.12, Pydantic v2, SQLAlchemy 2, PyMySQL, MySQL 8.4, Docker Compose, PyYAML, BeautifulSoup4, PyMuPDF, pypdf, reportlab, Pillow, MinerU CLI adapter, pytest.

**Spec:** `docs/superpowers/specs/2026-08-29-foreign-trade-agent-overhaul-design.md`

## Global Constraints

- Work only on `codex/foreign-trade-agent-overhaul`; never commit to `main`.
- All generated business data must have `is_synthetic=true` and use reserved/example domains.
- Runtime is Python 3.12; services bind development ports to `127.0.0.1`.
- No `.env`, credentials, database volumes, raw private corpora, indexes, model weights, or private holdout files enter Git.
- Unsupported or failed documents are quarantined with a typed error; they are never silently indexed as plain text.
- Every task follows red-green-refactor and ends with the named focused tests passing.

---

### Task 1: Replace the Legacy Product Tree with a Minimal Package

**Files:**
- Delete: legacy runtime under `rag_core/`, `tools/`, `utils/`, `web/`, `demo/ecommerce_seed/`, old `scripts/`, old `tests/`, `app.py`, `engineering_api.py`, `main.py`, `config.py`, `environment.yml`, `requirements.txt`, `introduce.md`
- Preserve: `.git/`, `AGENTS.md`, design/plan docs
- Create: `trade_agent/__init__.py`
- Create: `trade_agent/cli.py`
- Create: `tests/unit/test_package.py`
- Create: `pyproject.toml`
- Replace: `.gitignore`

**Interfaces:**
- Produces: importable `trade_agent` package and `trade-intel` console command mapped to `trade_agent.cli:main`.

- [ ] **Step 1: Write the package smoke test**

```python
def test_package_exposes_version():
    import trade_agent
    assert trade_agent.__version__ == "0.1.0"
```

- [ ] **Step 2: Remove only tracked legacy product paths and verify the branch**

Run: `test "$(git branch --show-current)" = codex/foreign-trade-agent-overhaul`

Run: `git rm -r rag_core tools utils web demo/ecommerce_seed scripts tests app.py engineering_api.py main.py config.py environment.yml requirements.txt introduce.md`

Expected: only paths inside the cloned repository are staged for deletion; design and plans remain.

- [ ] **Step 3: Create the package and dependency metadata**

```python
# trade_agent/__init__.py
__version__ = "0.1.0"
```

`pyproject.toml` must define Python `>=3.12,<3.13`, the `trade-intel` console script, runtime dependency groups, and pytest configuration with markers `integration`, `milvus`, and `e2e`.

- [ ] **Step 4: Run the focused test**

Run: `python3.12 -m pytest tests/unit/test_package.py -q`

Expected: PASS inside the eventual API image; if host Python 3.12 is absent, run `docker compose run --rm api ...` after Task 2.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: reset repository for trade intelligence"
```

### Task 2: Add Typed Configuration and the Local Service Stack

**Files:**
- Create: `trade_agent/config/settings.py`
- Create: `trade_agent/config/__init__.py`
- Create: `tests/unit/test_settings.py`
- Create: `.env.example`
- Create: `docker-compose.yml`
- Create: `docker/Dockerfile`
- Create: `docker/healthcheck.py`

**Interfaces:**
- Produces: `Settings.load() -> Settings` with nested `MysqlSettings`, `MilvusSettings`, `RedisSettings`, `ModelSettings`, and `RuntimeLimits`.
- Produces: Compose services named `mysql`, `etcd`, `minio`, `milvus`, `redis`, and `api`.

- [ ] **Step 1: Write configuration validation tests**

```python
def test_settings_reject_placeholder_password(monkeypatch):
    monkeypatch.setenv("MYSQL_APP_PASSWORD", "change-me")
    with pytest.raises(ValueError, match="placeholder"):
        Settings.load(runtime="compose")

def test_limits_are_positive():
    limits = RuntimeLimits(max_graph_steps=12, max_retries=1, max_llm_calls=5)
    assert limits.max_graph_steps == 12
```

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/unit/test_settings.py -q`

Expected: FAIL because settings models do not exist.

- [ ] **Step 3: Implement strict settings**

```python
class RuntimeLimits(BaseModel):
    max_graph_steps: int = Field(default=12, gt=0, le=50)
    max_retries: int = Field(default=1, ge=0, le=3)
    max_llm_calls: int = Field(default=5, ge=0, le=10)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    environment: Literal["test", "development", "compose"] = "development"
    mysql: MysqlSettings
    milvus: MilvusSettings
    redis: RedisSettings
    models: ModelSettings
    limits: RuntimeLimits = RuntimeLimits()
```

Compose must pin `milvusdb/milvus:v2.6.22`, matching official etcd/MinIO dependencies, MySQL 8.4, Redis 7.4-alpine, and Python 3.12-slim. Ports bind to loopback and health checks gate dependencies.

- [ ] **Step 4: Validate Compose and settings**

Run: `docker compose config --quiet`

Run: `docker compose build api`

Run: `docker compose run --rm --no-deps api pytest tests/unit/test_settings.py -q`

Expected: all commands succeed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .env.example docker-compose.yml docker trade_agent/config tests/unit/test_settings.py
git commit -m "build: add typed config and service stack"
```

### Task 3: Create the Seven-Table MySQL Contract and Synthetic Seed

**Files:**
- Create: `db/migrations/001_schema.sql`
- Create: `db/init/010_users.sh`
- Create: `trade_agent/db/models.py`
- Create: `trade_agent/db/session.py`
- Create: `trade_agent/db/migrate.py`
- Create: `trade_agent/db/seed.py`
- Create: `trade_agent/db/__init__.py`
- Create: `tests/unit/test_trade_seed.py`
- Create: `tests/integration/test_mysql_schema.py`

**Interfaces:**
- Produces: `generate_trade_seed(seed: int = 20260830) -> TradeSeedBundle`.
- Produces: `seed_database(engine: Engine, bundle: TradeSeedBundle) -> SeedSummary`.
- Produces: SQLAlchemy models `Country`, `Company`, `HsCode`, `Product`, `DataSource`, `CompanyProduct`, `TradeRecord`.

- [ ] **Step 1: Write deterministic seed and invariant tests**

```python
def test_seed_is_deterministic_and_synthetic():
    left = generate_trade_seed(20260830)
    right = generate_trade_seed(20260830)
    assert left.model_dump() == right.model_dump()
    assert len(left.companies) >= 60
    assert len(left.trade_records) >= 800
    assert all(company.is_synthetic for company in left.companies)
    assert {"growing", "declining", "dormant", "active"} <= set(left.lead_labels.values())
```

- [ ] **Step 2: Run tests to verify failure**

Run: `docker compose run --rm --no-deps api pytest tests/unit/test_trade_seed.py -q`

Expected: FAIL because generator and models are absent.

- [ ] **Step 3: Implement DDL, models, generator, and read-only user setup**

DDL must implement all seven tables and the unique/foreign/composite indexes in the spec. The generator must use fixed Faker-free vocabularies, Decimal amounts, 18 monthly windows, controlled lead patterns, unique `source_id + raw_record_id`, and `.example` websites.

```python
def generate_trade_seed(seed: int = 20260830) -> TradeSeedBundle:
    rng = random.Random(seed)
    countries = build_countries()
    companies = build_companies(rng, countries, count=60)
    records, labels = build_trade_records(rng, companies, months=18, minimum=800)
    return TradeSeedBundle(..., trade_records=records, lead_labels=labels)
```

`010_users.sh` must create a migration account and a query-only account; the query account receives only `SELECT` on `foreign_trade_db.*`.

- [ ] **Step 4: Run unit and real MySQL contract tests**

Run: `docker compose up -d mysql --wait`

Run: `docker compose run --rm api python -m trade_agent.db.migrate`

Run: `docker compose run --rm api python -m trade_agent.db.seed --seed 20260830`

Run: `docker compose run --rm api pytest tests/unit/test_trade_seed.py tests/integration/test_mysql_schema.py -q`

Expected: seven tables exist; row counts meet targets; duplicate raw record fails; query user cannot INSERT/UPDATE/DELETE.

- [ ] **Step 5: Commit**

```bash
git add db trade_agent/db tests/unit/test_trade_seed.py tests/integration/test_mysql_schema.py
git commit -m "feat: add synthetic foreign trade database"
```

### Task 4: Define Canonical Source, Document, Chunk, and Metadata Schemas

**Files:**
- Create: `trade_agent/schemas/source.py`
- Create: `trade_agent/schemas/evidence.py`
- Create: `trade_agent/schemas/__init__.py`
- Create: `tests/unit/test_source_schemas.py`

**Interfaces:**
- Produces: enums `FileType`, `SourceType`, `FactType`.
- Produces: `SourceRecord`, `DocumentRecord`, `ChunkMetadata`, `ChunkRecord`, and `SourceLocator`.
- Produces: `stable_id(prefix: str, *parts: str) -> str` and `content_sha256(value: str | bytes) -> str`.

- [ ] **Step 1: Write schema invariants**

```python
def test_chunk_metadata_requires_truth_boundary():
    with pytest.raises(ValidationError):
        ChunkMetadata(file_type="html", source_type="official_website")

def test_hs_code_preserves_leading_zero():
    value = ChunkMetadata.validated_fixture(hs_code="010121", is_synthetic=True)
    assert value.hs_code == "010121"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `docker compose run --rm --no-deps api pytest tests/unit/test_source_schemas.py -q`

Expected: FAIL because schema types are absent.

- [ ] **Step 3: Implement strict Pydantic models**

```python
class ChunkMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    document_id: str
    entity_id: str | None = None
    company_name: str | None = None
    normalized_name: str | None = None
    country_code: str | None = None
    region: str | None = None
    hs_code: str | None = None
    product_name: str | None = None
    sku: str | None = None
    file_type: FileType
    source_type: SourceType
    source_weight: float = Field(ge=0, le=1)
    fact_type: FactType | None = None
    publish_time: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    ingested_at: datetime
    source_url: AnyUrl | None = None
    canonical_url: AnyUrl | None = None
    source_locator: SourceLocator
    raw_record_id: str | None = None
    aggregation_info: dict[str, JsonValue] | None = None
    content_hash: str
    parent_document_hash: str
    language: str
    ocr_confidence: float | None = Field(default=None, ge=0, le=1)
    is_synthetic: bool
    license_scope: str | None = None
    dedupe_cluster_id: str | None = None
```

- [ ] **Step 4: Run focused tests**

Run: `docker compose run --rm --no-deps api pytest tests/unit/test_source_schemas.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/schemas tests/unit/test_source_schemas.py
git commit -m "feat: define trade document contracts"
```

### Task 5: Generate Every Required Synthetic Document Type

**Files:**
- Create: `trade_agent/data/demo_generator.py`
- Create: `scripts/bootstrap_trade_intel_demo.py`
- Create: `data/sources/trade_intel_demo.yaml`
- Create: `demo/trade_intel_seed/README.md`
- Create: `tests/unit/test_demo_corpus.py`

**Interfaces:**
- Produces: `generate_demo_corpus(output: Path, seed: int, clean: bool = False) -> CorpusSummary`.
- Consumes: `TradeSeedBundle` from Task 3.

- [ ] **Step 1: Write corpus coverage tests**

```python
def test_demo_has_every_required_source_and_file_type(tmp_path):
    summary = generate_demo_corpus(tmp_path, seed=20260830)
    assert summary.source_types == {
        "official_website", "b2b", "industry_news", "social",
        "regulator", "customs_profile",
    }
    assert {"html", "json", "jsonl", "pdf", "generated_profile"} <= summary.file_types
    assert summary.text_pdf_count >= 1
    assert summary.scanned_pdf_count >= 1
```

- [ ] **Step 2: Run test to verify failure**

Run: `docker compose run --rm --no-deps api pytest tests/unit/test_demo_corpus.py -q`

Expected: FAIL because the generator is absent.

- [ ] **Step 3: Implement deterministic corpus generation**

Create at least 12 website sections, 18 B2B products, 16 news stories including controlled syndication, 12 social posts, two text PDFs, one scanned-image PDF, and one monthly customs profile per selected company/HS/month. Every fixture includes a synthetic banner and manifest entry with content hash, expected entity, fact type, timestamps, and reference claim IDs.

```python
def generate_demo_corpus(output: Path, seed: int, clean: bool = False) -> CorpusSummary:
    ensure_safe_output(output, clean=clean)
    records = write_websites(...) + write_b2b(...) + write_news(...)
    records += write_social(...) + write_pdf_reports(...) + write_customs_profiles(...)
    return write_manifest(output, records, synthetic=True, seed=seed)
```

- [ ] **Step 4: Generate and validate the checked-in demo**

Run: `docker compose run --rm --no-deps api python -m scripts.bootstrap_trade_intel_demo --output demo/trade_intel_seed --clean`

Run: `docker compose run --rm --no-deps api pytest tests/unit/test_demo_corpus.py -q`

Expected: generator is idempotent and all manifest hashes match files.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/data/demo_generator.py scripts/bootstrap_trade_intel_demo.py data/sources/trade_intel_demo.yaml demo/trade_intel_seed tests/unit/test_demo_corpus.py
git commit -m "feat: generate multi-source trade intelligence corpus"
```

### Task 6: Implement File-Type Routing and Source-Specific Chunking

**Files:**
- Create: `trade_agent/data/router.py`
- Create: `trade_agent/data/parsers.py`
- Create: `trade_agent/data/pdf.py`
- Create: `trade_agent/data/chunkers.py`
- Create: `trade_agent/data/quarantine.py`
- Create: `tests/unit/test_data_router.py`
- Create: `tests/unit/test_chunkers.py`
- Create: `tests/integration/test_pdf_pipeline.py`

**Interfaces:**
- Produces: `DocumentRouter.load(SourceInput) -> list[DocumentRecord]`.
- Produces: `ChunkRouter.chunk(DocumentRecord) -> list[ChunkRecord]`.
- Produces: `QuarantineRecord(error_code, source_path, parser, diagnostic)`.

- [ ] **Step 1: Write routing and golden-boundary tests**

```python
def test_social_post_is_one_chunk(router, social_fixture):
    docs = router.load(social_fixture)
    chunks = ChunkRouter().chunk(docs[0])
    assert len(chunks) == 1
    assert chunks[0].metadata.source_locator.post_id == "POST-001"

def test_news_title_is_injected_into_each_chunk(news_document):
    chunks = ChunkRouter(max_tokens=80, overlap_tokens=10).chunk(news_document)
    assert len(chunks) > 1
    assert all("Factory expansion" in chunk.content for chunk in chunks)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `docker compose run --rm --no-deps api pytest tests/unit/test_data_router.py tests/unit/test_chunkers.py -q`

Expected: FAIL because router/chunkers do not exist.

- [ ] **Step 3: Implement parsers and source-aware chunkers**

Implement physical parsers for HTML/Markdown, JSON/JSONL, text PDF, scanned PDF, and generated profile. Implement `WebsiteSectionChunker`, `B2BProductChunker`, `NewsParagraphChunker`, `SocialPostChunker`, `PdfLayoutChunker`, and `CustomsProfileChunker`. Token counting uses the configured embedding tokenizer when available and a deterministic character fallback in unit tests.

```python
class ChunkRouter:
    def chunk(self, document: DocumentRecord) -> list[ChunkRecord]:
        chunker = self._by_source_type[document.source_type]
        return assign_stable_chunk_metadata(document, chunker.split(document))
```

The PDF adapter executes MinerU without a shell in an isolated temporary directory, parses content-list/middle JSON first, Markdown second, and records degraded fallback. Scanned PDFs without usable OCR are quarantined.

- [ ] **Step 4: Run all parser and chunker tests**

Run: `docker compose run --rm --no-deps api pytest tests/unit/test_data_router.py tests/unit/test_chunkers.py tests/integration/test_pdf_pipeline.py -q`

Expected: text PDF preserves page locators; scanned fixture either uses MinerU or produces the explicit tested degraded/quarantine state; each source type matches its golden boundaries.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/data tests/unit/test_data_router.py tests/unit/test_chunkers.py tests/integration/test_pdf_pipeline.py
git commit -m "feat: route and chunk trade intelligence sources"
```

### Task 7: Build Deterministic Manifests and Quarantine Reports

**Files:**
- Create: `trade_agent/data/manifest.py`
- Create: `trade_agent/data/pipeline.py`
- Create: `scripts/ingest_trade_sources.py`
- Create: `tests/unit/test_manifest.py`
- Create: `tests/integration/test_ingestion_pipeline.py`

**Interfaces:**
- Produces: `IngestionPipeline.run(catalog: SourceCatalog, output: Path) -> BuildManifest`.
- Produces: immutable `BuildManifest(build_id, sources, documents, chunks, quarantined, config_hash)`.

- [ ] **Step 1: Write determinism and failure-accounting tests**

```python
def test_same_inputs_produce_same_build_id(pipeline, catalog, tmp_path):
    first = pipeline.run(catalog, tmp_path / "one")
    second = pipeline.run(catalog, tmp_path / "two")
    assert first.build_id == second.build_id
    assert [c.chunk_id for c in first.chunks] == [c.chunk_id for c in second.chunks]

def test_unsupported_file_is_counted_not_dropped(pipeline, catalog_with_binary, tmp_path):
    result = pipeline.run(catalog_with_binary, tmp_path)
    assert result.quarantined[0].error_code == "unsupported_file_type"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `docker compose run --rm --no-deps api pytest tests/unit/test_manifest.py tests/integration/test_ingestion_pipeline.py -q`

Expected: FAIL because pipeline and manifest do not exist.

- [ ] **Step 3: Implement canonical hashing and atomic manifest writes**

```python
def canonical_hash(value: JsonValue) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()

class IngestionPipeline:
    def run(self, catalog: SourceCatalog, output: Path) -> BuildManifest:
        documents, quarantined = self.collect_and_parse(catalog)
        chunks = [chunk for document in documents for chunk in self.chunk_router.chunk(document)]
        return BuildManifest.freeze(documents, chunks, quarantined, self.config)
```

Manifest writes use temporary files plus `os.replace`; build IDs exclude volatile ingestion timestamps but include source content hashes, parser/chunker versions, metadata schema, and config.

- [ ] **Step 4: Run ingestion and tests**

Run: `docker compose run --rm --no-deps api python -m scripts.ingest_trade_sources --catalog data/sources/trade_intel_demo.yaml --output /tmp/trade-build.json`

Run: `docker compose run --rm --no-deps api pytest tests/unit/test_manifest.py tests/integration/test_ingestion_pipeline.py -q`

Expected: manifest contains all source categories, stable IDs, 100% valid required metadata, and explicit quarantine totals.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/data/manifest.py trade_agent/data/pipeline.py scripts/ingest_trade_sources.py tests/unit/test_manifest.py tests/integration/test_ingestion_pipeline.py
git commit -m "feat: build deterministic trade ingestion manifests"
```

### Task 8: Verify Foundation Services and Produce a Reusable Smoke Command

**Files:**
- Create: `scripts/smoke_foundation.py`
- Create: `tests/contract/test_compose_contract.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `python -m scripts.smoke_foundation` returning exit 0 only when config, MySQL schema/seed, corpus generation, ingestion, and manifest checks pass.

- [ ] **Step 1: Write the smoke contract test**

```python
def test_foundation_smoke_returns_machine_readable_summary(compose_stack):
    summary = run_foundation_smoke()
    assert summary["mysql_tables"] == 7
    assert summary["trade_records"] >= 800
    assert summary["quarantine_unexpected"] == 0
    assert summary["metadata_required_completeness"] == 1.0
```

- [ ] **Step 2: Run test to verify failure**

Run: `docker compose run --rm api pytest tests/contract/test_compose_contract.py -q`

Expected: FAIL because smoke runner is absent.

- [ ] **Step 3: Implement smoke runner and foundation README**

The runner emits JSON with service versions, schema fingerprint, seed hash, corpus counts, build ID, parser backends, chunk counts by source/file type, quarantine counts, and metadata completeness. README documents exact commands and synthetic limitations.

- [ ] **Step 4: Run the full foundation verification**

Run: `docker compose up -d mysql etcd minio milvus redis --wait`

Run: `docker compose run --rm api python -m scripts.smoke_foundation`

Run: `docker compose run --rm api pytest tests/unit tests/contract/test_compose_contract.py tests/integration/test_mysql_schema.py tests/integration/test_ingestion_pipeline.py tests/integration/test_pdf_pipeline.py -q`

Expected: all pass; JSON summary proves seven tables, every source type, deterministic build, and truthful PDF degradation state.

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke_foundation.py tests/contract/test_compose_contract.py README.md
git commit -m "test: verify trade data foundation"
```
