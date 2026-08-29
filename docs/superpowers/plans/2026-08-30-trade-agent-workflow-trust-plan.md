# Trade Agent SQL, Workflow, and Trust Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe Text-to-SQL, unified Evidence, pre-generation sufficiency validation, atomic Claim grounding, a bounded LangGraph SQL+RAG workflow with Redis checkpointing, and one strict API/CLI product surface.

**Architecture:** Natural language first becomes a typed route and constraint plan. SQL and RAG nodes run concurrently when independent, normalize into one Evidence contract, resolve entities and conflicts, validate sufficiency, generate a structured answer, and enforce claim-to-evidence grounding before final output. Deterministic policies control tools, retries, budgets, and fail-closed behavior; Redis persists resumable state IDs and hashes.

**Tech Stack:** Python 3.12, Pydantic v2, SQLAlchemy/PyMySQL, SQLGlot, LangGraph, langgraph-checkpoint-redis, Redis 7.4, FastAPI, httpx, pytest.

**Spec:** `docs/superpowers/specs/2026-08-29-foreign-trade-agent-overhaul-design.md`

## Global Constraints

- The database query user is SELECT-only; prompt rules never substitute for database permissions or AST validation.
- SQL and RAG both emit the same Evidence model; downstream code does not consume raw rows or raw Milvus hits.
- Evidence Validator runs before generation; Claim Guard runs after generation and before any answer leaves the API.
- A core fact that is missing, stale, conflicted, or outside scope causes refusal or explicit degradation.
- Only retryable transport/rate-limit failures retry; policy/schema/validation errors do not.
- Every external node obeys `MAX_GRAPH_STEPS`, `MAX_RETRIES`, `MAX_LLM_CALLS`, timeout, candidate, and token budgets.

---

### Task 1: Build a Live Schema Registry

**Files:**
- Create: `trade_agent/db/registry.py`
- Create: `trade_agent/db/contracts.py`
- Create: `trade_agent/config/schema_registry.yaml`
- Create: `tests/unit/test_schema_registry.py`
- Create: `tests/integration/test_live_schema_registry.py`

**Interfaces:**
- Produces: `SchemaRegistry.refresh(connection: Connection) -> RegistrySnapshot`.
- Produces: `SchemaRegistry.link(constraints: QueryConstraints) -> SchemaLinkResult`.
- Produces: `RegistrySnapshot(fingerprint, tables, columns, joins, aliases, aggregations)`.

- [ ] **Step 1: Write registry fingerprint and unknown-field tests**

```python
def test_registry_rejects_unknown_business_field(registry):
    result = registry.link(QueryConstraints(metrics=["profit_margin"]))
    assert result.ok is False
    assert result.error_code == "schema_not_registered"

def test_live_schema_change_changes_fingerprint(mysql_registry):
    first = mysql_registry.refresh()
    mysql_registry.test_only_add_column("companies", "unexpected_col")
    second = mysql_registry.refresh()
    assert first.fingerprint != second.fingerprint
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/unit/test_schema_registry.py -q`

Expected: FAIL because registry does not exist.

- [ ] **Step 3: Implement INFORMATION_SCHEMA discovery plus reviewed semantics**

```python
class SchemaRegistry:
    def refresh(self, connection: Connection) -> RegistrySnapshot:
        physical = inspect_information_schema(connection, database="foreign_trade_db")
        semantics = RegistrySemantics.load(self.semantic_path)
        return RegistrySnapshot.from_physical_and_semantic(physical, semantics)
```

The YAML registers table/column descriptions, aliases, permitted joins, groupable dimensions, aggregations, sensitive fields, and maximum result rows. Refresh fails if physical and semantic contracts diverge.

- [ ] **Step 4: Run unit and real MySQL tests**

Run: `docker compose run --rm api pytest tests/unit/test_schema_registry.py tests/integration/test_live_schema_registry.py -q`

Expected: PASS against seven live tables and fail-closed mismatch fixtures.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/db/registry.py trade_agent/db/contracts.py trade_agent/config/schema_registry.yaml tests/unit/test_schema_registry.py tests/integration/test_live_schema_registry.py
git commit -m "feat: register the live trade schema"
```

### Task 2: Parse Intent and Produce a Restricted SQL Query Plan

**Files:**
- Create: `trade_agent/agents/intent.py`
- Create: `trade_agent/db/sql_planner.py`
- Create: `tests/unit/test_intent_parser.py`
- Create: `tests/unit/test_sql_planner.py`

**Interfaces:**
- Produces: `IntentParser.parse(question: str, explicit_filters: RetrievalFilter | None) -> QueryIntent`.
- Produces: `SqlPlanner.plan(intent: QueryIntent, registry: RegistrySnapshot) -> SqlQueryPlan`.
- `SqlQueryPlan` contains selected tables/columns, joins, predicates, group_by, aggregations, order_by, and limit; it is not raw executable SQL.

- [ ] **Step 1: Write representative intent/plan tests**

```python
def test_top_importers_plan(parser, planner, registry):
    intent = parser.parse("最近半年美国采购 HS850440 金额最高的 10 家公司")
    plan = planner.plan(intent, registry)
    assert plan.aggregations == [Aggregation("trade_amount", "sum", "purchase_amount_usd")]
    assert plan.limit == 10
    assert plan.filters.hs_codes == ["850440"]
    assert plan.time_range.months == 6
    assert plan.need_trade_data is True
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/unit/test_intent_parser.py tests/unit/test_sql_planner.py -q`

Expected: FAIL because intent/planner are absent.

- [ ] **Step 3: Implement deterministic constraint extraction and optional structured LLM provider**

The deterministic path supports all frozen evaluation intents and preserves identifiers. The optional provider must return the same Pydantic `QueryIntent`/`SqlQueryPlan` JSON schema and is rejected if it introduces unknown schema elements.

```python
class SqlPlanner:
    def plan(self, intent: QueryIntent, registry: RegistrySnapshot) -> SqlQueryPlan:
        linked = registry.link(intent.constraints)
        if not linked.ok:
            raise SchemaNotRegistered(linked.missing)
        return compile_business_intent(intent, linked, enforced_limit=50)
```

- [ ] **Step 4: Run tests**

Run: `docker compose run --rm api pytest tests/unit/test_intent_parser.py tests/unit/test_sql_planner.py -q`

Expected: PASS for top-N, company trend, country/HS activity, mixed SQL+RAG, and out-of-scope questions.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/agents/intent.py trade_agent/db/sql_planner.py tests/unit/test_intent_parser.py tests/unit/test_sql_planner.py
git commit -m "feat: plan constrained trade SQL queries"
```

### Task 3: Validate and Execute Read-Only SQL with Structured Provenance

**Files:**
- Create: `trade_agent/db/sql_renderer.py`
- Create: `trade_agent/db/sql_validator.py`
- Create: `trade_agent/db/sql_executor.py`
- Create: `trade_agent/evidence/models.py`
- Create: `trade_agent/evidence/sql.py`
- Create: `tests/unit/test_sql_validator.py`
- Create: `tests/integration/test_sql_execution.py`

**Interfaces:**
- Produces: `SqlRenderer.render(plan: SqlQueryPlan) -> RenderedSql` with bound parameters.
- Produces: `SqlValidator.validate(rendered: RenderedSql, registry: RegistrySnapshot) -> ValidatedSql`.
- Produces: `ReadOnlySqlExecutor.execute(validated: ValidatedSql) -> SqlExecutionResult`.
- Produces: `build_sql_evidence(result: SqlExecutionResult) -> list[Evidence]`.

- [ ] **Step 1: Write adversarial AST and permission tests**

```python
@pytest.mark.parametrize("sql", [
    "DROP TABLE companies", "SELECT 1; DELETE FROM trade_records",
    "SELECT * FROM mysql.user", "SELECT * FROM companies",
])
def test_unsafe_sql_is_rejected(validator, registry, sql):
    with pytest.raises(SqlAstRejected):
        validator.validate(RenderedSql(sql=sql, params={}), registry)

def test_query_user_cannot_write(query_connection):
    with pytest.raises(DatabaseError):
        query_connection.execute(text("UPDATE companies SET industry='x'"))
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/unit/test_sql_validator.py -q`

Expected: FAIL because validator/executor are absent.

- [ ] **Step 3: Implement SQLGlot validation, bound rendering, budgets, and Evidence**

```python
class SqlValidator:
    def validate(self, rendered: RenderedSql, registry: RegistrySnapshot) -> ValidatedSql:
        tree = sqlglot.parse_one(rendered.sql, read="mysql")
        require_single_select_or_cte(tree)
        require_registered_tables_columns_joins(tree, registry)
        require_limit(tree, maximum=registry.max_result_rows)
        reject_star_system_tables_and_side_effects(tree)
        return ValidatedSql(tree.sql(dialect="mysql"), rendered.params, registry.fingerprint)
```

Executor uses `SET SESSION TRANSACTION READ ONLY`, statement timeout controls, maximum rows, monotonic timing, and result hashing. Evidence records normalized SQL, bound-filter names without secrets, schema fingerprint, aggregation grain, row count, result hash, and raw record locators.

- [ ] **Step 4: Run unit and live MySQL tests**

Run: `docker compose run --rm api pytest tests/unit/test_sql_validator.py tests/integration/test_sql_execution.py -q`

Expected: all attacks fail; approved aggregate queries return correct Decimal totals and replayable evidence.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/db/sql_renderer.py trade_agent/db/sql_validator.py trade_agent/db/sql_executor.py trade_agent/evidence tests/unit/test_sql_validator.py tests/integration/test_sql_execution.py
git commit -m "feat: execute safe SQL as evidence"
```

### Task 4: Normalize Retrieved Chunks and Define Claims, Conflicts, and Answers

**Files:**
- Modify: `trade_agent/evidence/models.py`
- Create: `trade_agent/evidence/normalize.py`
- Create: `tests/unit/test_evidence_models.py`

**Interfaces:**
- Produces: `normalize_retrieval(outcome: RetrievalOutcome) -> list[Evidence]`.
- Produces: Pydantic models `Evidence`, `Claim`, `Conflict`, `IntelligenceAnswer`, `RetrievalProvenance`, `SqlProvenance`.

- [ ] **Step 1: Write cross-source and citation invariants**

```python
def test_sql_and_rag_evidence_share_contract(sql_result, retrieval_outcome):
    sql_evidence = build_sql_evidence(sql_result)[0]
    rag_evidence = normalize_retrieval(retrieval_outcome)[0]
    assert type(sql_evidence) is type(rag_evidence) is Evidence
    assert sql_evidence.locator.raw_record_ids
    assert rag_evidence.locator.chunk_id
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/unit/test_evidence_models.py -q`

Expected: FAIL because complete models/normalizer are absent.

- [ ] **Step 3: Implement extra-forbid models and stable evidence IDs**

Evidence IDs derive from source/build/locator/content hash, not retrieval rank. Claims use statuses `supported`, `conflicted`, `stale`, `insufficient`, or `analysis`; final answers include evidence, conflicts, refusal reason, degraded components, and public trace.

- [ ] **Step 4: Run tests**

Run: `docker compose run --rm api pytest tests/unit/test_evidence_models.py -q`

Expected: PASS; invalid citation IDs, impossible validity ranges, and unbounded confidence fail validation.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/evidence/models.py trade_agent/evidence/normalize.py tests/unit/test_evidence_models.py
git commit -m "feat: unify SQL and retrieval evidence"
```

### Task 5: Implement Evidence Sufficiency Validation

**Files:**
- Create: `trade_agent/evidence/requirements.py`
- Create: `trade_agent/evidence/validator.py`
- Create: `tests/unit/test_evidence_validator.py`

**Interfaces:**
- Produces: `EvidenceRequirements.for_intent(intent: QueryIntent) -> EvidenceRequirements`.
- Produces: `EvidenceValidator.validate(intent, evidence, conflicts, as_of) -> ValidationOutcome`.

- [ ] **Step 1: Write sufficiency, staleness, and conflict tests**

```python
def test_lead_decision_requires_trade_and_current_status(validator):
    outcome = validator.validate(lead_intent(), only_social_evidence(), [], AS_OF)
    assert outcome.can_answer is False
    assert outcome.error_code == "evidence_insufficient"

def test_conflicted_core_fact_blocks_generation(validator):
    outcome = validator.validate(status_intent(), conflicting_evidence(), conflicts(), AS_OF)
    assert outcome.can_answer is False
    assert outcome.error_code == "evidence_conflict"
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/unit/test_evidence_validator.py -q`

Expected: FAIL because requirements/validator are absent.

- [ ] **Step 3: Implement deterministic evidence contracts per intent**

Requirements specify required fact types, authority/directness, maximum age, independent source minimum, temporal scope, and whether SQL evidence is mandatory. Outcome lists satisfied and missing requirements and selects `answer`, `rewrite_once`, or `refuse`.

- [ ] **Step 4: Run tests**

Run: `docker compose run --rm api pytest tests/unit/test_evidence_validator.py -q`

Expected: PASS for answerable, stale, missing SQL, low-authority, unresolved entity, and conflict cases.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/evidence/requirements.py trade_agent/evidence/validator.py tests/unit/test_evidence_validator.py
git commit -m "feat: validate trade evidence sufficiency"
```

### Task 6: Generate Structured Answers and Guard Every Atomic Claim

**Files:**
- Create: `trade_agent/generation/base.py`
- Create: `trade_agent/generation/deterministic.py`
- Create: `trade_agent/generation/deepseek.py`
- Create: `trade_agent/evidence/claim_guard.py`
- Create: `tests/unit/test_generation.py`
- Create: `tests/unit/test_claim_guard.py`

**Interfaces:**
- Produces: `AnswerGenerator.generate(intent, evidence, validation) -> DraftAnswer`.
- Produces: `ClaimHallucinationGuard.guard(draft: DraftAnswer, evidence: Sequence[Evidence]) -> GuardOutcome`.

- [ ] **Step 1: Write adversarial claim mutation tests**

```python
@pytest.mark.parametrize("mutation", ["amount", "currency", "date", "entity", "evidence_id"])
def test_guard_rejects_mutated_claim(mutation, guard, supported_draft):
    draft = mutate_claim(supported_draft, mutation)
    outcome = guard.guard(draft, draft.fixture_evidence)
    assert outcome.accepted is False
    assert "claim_unsupported" in outcome.error_codes
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/unit/test_generation.py tests/unit/test_claim_guard.py -q`

Expected: FAIL because generators/guard are absent.

- [ ] **Step 3: Implement evidence-only deterministic generation and optional DeepSeek JSON generation**

Both providers return `DraftAnswer(answer, claims)` where every factual claim contains explicit evidence IDs. The deterministic provider serializes validated facts for reproducible evaluation. The DeepSeek provider uses a strict JSON schema and never receives evidence excluded by license/policy.

Guard checks citation existence, numeric/unit/period/grain equality, entity/time/jurisdiction match, and lexical/structured support. Unsupported non-core claims are removed or limited; unsupported core claims refuse the answer.

- [ ] **Step 4: Run tests**

Run: `docker compose run --rm api pytest tests/unit/test_generation.py tests/unit/test_claim_guard.py -q`

Expected: PASS for supported answers and all deliberate hallucination mutations.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/generation trade_agent/evidence/claim_guard.py tests/unit/test_generation.py tests/unit/test_claim_guard.py
git commit -m "feat: ground every generated trade claim"
```

### Task 7: Implement the Bounded LangGraph SQL+RAG Workflow

**Files:**
- Create: `trade_agent/agents/state.py`
- Create: `trade_agent/agents/router.py`
- Create: `trade_agent/agents/nodes.py`
- Create: `trade_agent/agents/graph.py`
- Create: `trade_agent/errors.py`
- Create: `tests/unit/test_router.py`
- Create: `tests/integration/test_graph.py`

**Interfaces:**
- Produces: `build_trade_graph(deps: GraphDependencies, checkpointer=None) -> CompiledStateGraph`.
- Produces: `TradeIntelState` and `RoutePlan`.

- [ ] **Step 1: Write route, parallel, retry, and limit tests**

```python
@pytest.mark.parametrize(("question", "sql", "rag"), [
    ("ABC 最近半年采购额", True, False),
    ("ABC 官网最近是否扩产", False, True),
    ("ABC 是否值得跟进", True, True),
])
def test_router_selects_required_channels(router, question, sql, rag):
    plan = router.route(question)
    assert (plan.need_trade_data, plan.need_external_intel) == (sql, rag)

def test_loop_stops_at_step_limit(graph_with_rewrite_loop):
    result = graph_with_rewrite_loop.invoke(request_state(), config_with_limit(4))
    assert result["errors"][-1]["code"] == "step_limit_exceeded"
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/unit/test_router.py tests/integration/test_graph.py -q`

Expected: FAIL because graph modules are absent.

- [ ] **Step 3: Implement explicit nodes and conditional edges**

Implement `policy_gate`, `router`, parallel `sql_node`/`rag_node`, `normalize`, `entity_dedup_conflict`, `evidence_validator`, one `query_rewrite`, `answer_draft`, `claim_guard`, and `finalizer`. Nodes return typed status/error objects; broad exceptions become classified errors and never silently continue.

- [ ] **Step 4: Run graph tests**

Run: `docker compose run --rm api pytest tests/unit/test_router.py tests/integration/test_graph.py -q`

Expected: SQL-only, RAG-only, parallel, one-branch timeout, required-branch timeout, rewrite-once, refusal, and loop limit tests pass.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/agents trade_agent/errors.py tests/unit/test_router.py tests/integration/test_graph.py
git commit -m "feat: orchestrate bounded SQL and RAG"
```

### Task 8: Add Redis Checkpoint Persistence and Resume

**Files:**
- Create: `trade_agent/agents/checkpoint.py`
- Create: `tests/integration/test_redis_checkpoint.py`

**Interfaces:**
- Produces: `RedisCheckpointFactory.create(settings) -> AsyncRedisSaver`.
- Produces: `resume_run(thread_id: str, run_id: str) -> TradeIntelState`.

- [ ] **Step 1: Write persistence, TTL, and sensitive-state tests**

```python
async def test_graph_resumes_after_generation_failure(redis_graph):
    failed = await redis_graph.first_run(fail_at="answer_draft")
    resumed = await redis_graph.resume(failed.thread_id, failed.run_id)
    assert resumed.completed_nodes.count("sql_node") == 1
    assert resumed.completed_nodes.count("rag_node") == 1

async def test_checkpoint_does_not_store_raw_sql_rows(redis_client, failed_run):
    payload = await redis_client.dump_checkpoint(failed_run.thread_id)
    assert "raw_rows" not in payload
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/integration/test_redis_checkpoint.py -q`

Expected: FAIL because checkpoint integration is absent.

- [ ] **Step 3: Implement AsyncRedisSaver integration and state minimization**

Checkpoint config uses `thread_id` and `run_id`, TTL, evidence IDs/hashes, typed errors, and idempotency keys. Connection failure is classified `checkpoint_unavailable`; strict resume requests fail closed instead of silently starting a new run.

- [ ] **Step 4: Run real Redis tests**

Run: `docker compose up -d redis --wait`

Run: `docker compose run --rm api pytest tests/integration/test_redis_checkpoint.py -q`

Expected: resume skips completed retrieval/SQL nodes, TTL removes test state, and sensitive values are absent.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/agents/checkpoint.py tests/integration/test_redis_checkpoint.py
git commit -m "feat: resume trade graphs from Redis"
```

### Task 9: Replace Product Entrypoints with Strict FastAPI and CLI

**Files:**
- Create: `trade_agent/api/models.py`
- Create: `trade_agent/api/dependencies.py`
- Create: `trade_agent/api/app.py`
- Modify: `trade_agent/cli.py`
- Create: `tests/contract/test_api.py`
- Create: `tests/e2e/test_query_workflow.py`

**Interfaces:**
- Produces endpoints `/health`, `/ready`, `/v1/query`, `/v1/retrieve`, `/v1/runs/{run_id}`, `/v1/runs/{run_id}/resume`, `/v1/evidence/{evidence_id}`.
- Produces CLI commands `bootstrap-demo`, `db`, `ingest`, `index`, `query`, `eval`, `smoke`, `verify-report`.

- [ ] **Step 1: Write API contract and E2E tests**

```python
def test_query_response_separates_claims_and_evidence(client):
    response = client.post("/v1/query", json={"question": "ABC 是否值得跟进", "top_k": 10})
    assert response.status_code == 200
    body = response.json()
    assert all(claim["evidence_ids"] for claim in body["claims"] if claim["status"] == "supported")
    assert body["build_id"]

def test_ready_is_false_when_milvus_contract_missing(client_without_index):
    response = client_without_index.get("/ready")
    assert response.status_code == 503
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/contract/test_api.py tests/e2e/test_query_workflow.py -q`

Expected: FAIL because API does not exist.

- [ ] **Step 3: Implement dependency lifespan, strict request models, and endpoints**

Dependencies initialize lazily during lifespan, verify schema/index contracts, and expose health separately from readiness. Request models forbid extra fields and bound text/top-k. Streaming is excluded from v1 so unvalidated drafts cannot leak.

- [ ] **Step 4: Run API/E2E tests and smoke requests**

Run: `docker compose up -d --wait`

Run: `docker compose exec api pytest tests/contract/test_api.py tests/e2e/test_query_workflow.py -q`

Run: `curl -fsS http://127.0.0.1:8000/ready`

Expected: all pass; mixed question includes SQL and RAG Evidence, conflicts/refusals are machine-readable, and no unsupported claim passes.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/api trade_agent/cli.py tests/contract/test_api.py tests/e2e/test_query_workflow.py
git commit -m "feat: expose the grounded trade agent"
```
