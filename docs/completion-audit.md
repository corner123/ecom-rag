# Completion audit

## Status vocabulary

- **Repository evidence present:** code, tests, configuration, and deterministic artifacts exist and are checked by `scripts.verify_repository`.
- **Committed synthetic evidence:** immutable local CPU evaluation reports pass their checksum/recomputation verifier.
- **Controller-final live gate:** the implementation exists, but acceptance requires a fresh service-backed command and its output from the controller. Synthetic reports do not satisfy this status.

## Requirement matrix

| Requirement | Repository evidence | Current status | Final gate |
| --- | --- | --- | --- |
| Synthetic-only product boundary | corpus generator, source schemas, manifests, README/docs | Repository evidence present | Inspect generated corpus/manifest in smoke |
| Docker topology for API, MySQL, etcd, MinIO, Milvus, Redis | `docker-compose.yml`, `.env.example`, compose contract tests | Repository evidence present | Controller: `docker compose up -d --wait` |
| Seven MySQL tables, constraints, seed scale, read-only role | `db/migrations/001_schema.sql`, `db/init/010_users.sh`, DB models/seed, integration tests | Repository evidence present | Controller: migration/seed and live permission tests |
| Website/B2B/news/social/PDF/scanned-PDF/customs routing | `trade_agent/data`, catalog, corpus fixtures, routing/PDF integration tests | Repository evidence present | Controller: foundation ingestion smoke; scanned fixture may explicitly quarantine when OCR is unavailable |
| Canonical metadata completeness | source schemas, chunkers, manifest/build validation, unit/integration tests | Repository evidence present | Controller: service-backed build and metadata check |
| Real BGE-M3 artifact and embedding contract | pinned revisions, trusted artifact manifest, embedding tests/smoke | Repository evidence present | Controller: `scripts.smoke_embeddings` with verified model bytes |
| Milvus scalar metadata and filter-first ANN | collection contract/store, typed filter compiler, index builder, Milvus tests | Repository evidence present | Controller: real insert/search/filter/reconnect/cleanup; fake/skip is not acceptance |
| BM25, dense, weighted RRF, filters, reranking, traces | retrieval package, profile config, unit/integration tests | Repository evidence present | Controller: real BGE/Milvus/BGE-reranker trace |
| Entity resolution, deduplication, conflict arbitration | `trade_agent/entities`, fusion/evidence models, tests | Repository evidence present | Controller: full mixed-source workflow trace |
| Schema Registry and restricted Text-to-SQL | registry/planner/renderer/validator/executor, SQL tests | Repository evidence present | Controller: live schema and read-only MySQL round-trip |
| Evidence validation and per-Claim guard | evidence package, tamper/refusal tests | Repository evidence present | Controller: end-to-end supported/tampered/insufficient cases |
| LangGraph SQL/RAG orchestration and budgets | agents package, graph integration/E2E tests | Repository evidence present | Controller: SQL-only, RAG-only, mixed, conflict, retry and fallback traces |
| Redis checkpoint projection, TTL, resume, idempotency | checkpoint implementation and Redis tests | Repository evidence present | Controller: live Redis recovery and TTL gate |
| FastAPI and complete CLI surface | API/CLI/server modules and contract tests; eval/report delegation | Repository evidence present | Controller: live API smoke |
| Development set at least 36 queries | 44 public cases and separate reference artifact | Repository evidence present | Audit recounts JSONL |
| Private holdout at least 15, ignored | first aggregate records 44 cases; private tree and consumption lock are ignored/untracked | Repository evidence plus local lock present | Audit checks aggregate count, ignore rule, lock existence, and tracking state |
| No evaluation-label/index/prompt leakage | separated artifacts, runtime adapter accepts questions only, leakage auditor/tests | Repository evidence present | Audit executes zero-leakage validation against the ignored private split |
| Freeze before holdout and one-shot consumption | holdout preflight/consume code, ignored atomic lock, committed frozen snapshot | Repository evidence present | First holdout already consumed; never rerun or tune on it |
| Development optimization with paired evidence | immutable cycle and 14 linked run bundles | Committed synthetic evidence | Local CPU only; not real dense/reranker acceptance |
| First private holdout | immutable redacted aggregate and frozen snapshot | Committed synthetic evidence | Local CPU retrieval only; generation/Judge/live systems unmeasured |
| Rule metrics and Judge error semantics | evaluator/report contracts and tests | Repository evidence present | Judge remains `judge_not_run`, scores null, errors categorized |
| No reuse of ecommerce results | trade evaluator/report paths and docs cite only trade reports | Repository evidence present | Existing ecommerce artifacts remain separately named historical data |
| Repository hygiene | security tests and `scripts.verify_repository` | Repository evidence present | Controller reruns full tracked secret/host-path scan and `git diff --check` |

## Spec acceptance items

| Acceptance item | Evidence available now | Truthful disposition |
| --- | --- | --- |
| 1. Compose health | compose definition and tests | Controller-final live gate |
| 2. Seven-table DDL, keys, indexes, read-only grant | migration/init SQL and tests | Controller-final live gate for deployed DB |
| 3. Every source/file fixture parses or quarantines as specified | fixtures, parser/router tests | Controller-final foundation smoke |
| 4. Chunk golden samples and 100% metadata validation | unit/integration tests and build guards | Repository evidence present; live build is controller gate |
| 5. Milvus insert/search/filter/reconnect/cleanup | real-test code and smoke command | Controller-final live gate; no Task 10 Docker receipt |
| 6. BM25/Dense/WRRF/Reranker per-query trace | trace contracts/tests; local CPU public rows for development | Real BGE/Milvus/reranker trace is controller-final |
| 7. LangGraph SQL+RAG trace, Redis resume, limits/fallback | graph/checkpoint code and tests | Controller-final live gate |
| 8. Evidence Validator and tampered Claim rejection | validator/guard tests | Repository evidence present; full live workflow still gated |
| 9. Development paired delta and one-shot holdout | immutable cycle, redacted aggregate, snapshot, ignored lock | Committed synthetic evidence only |
| 10. Full tests plus secret/host scan | local Task 10 gates and audit command | Controller reruns full suite and live-dependent tests |
| 11. README reports only trade measurements with limitations | root README and hygiene check | Repository evidence present |
| 12. No legacy ecommerce runtime path; main unchanged | absence checks and branch diff | Controller verifies final diff and remote baseline |

## Measured artifact facts

- Development `full_rerank`: precision `0.075231` to `0.194907`; recall `0.472222` to `0.472222`; zero paired recall/precision regressions.
- First holdout: Recall@10 `0.5833333333333334`; context precision `0.20601851851851852`; 44 completed retrieval cases.
- Adapter: local CPU hash cosine, BM25, lexical reranker.
- Generation: `not_run`. Judge: `judge_not_run`; faithfulness/relevance null.

These facts do not demonstrate production use, customer/buyer coverage, answer safety, business decisions, real Milvus/BGE behavior, or live service health.
