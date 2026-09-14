# Data contract and trust boundaries

## Synthetic source contract

The reviewed catalog declares source identity, `source_type`, file type, path or canonical URL, authorization, and synthetic status. The router accepts only supported catalog entries beneath the owned corpus root. Unsafe paths, unapproved URLs, oversized inputs, secrets, and unexpected parser failures are rejected or quarantined.

The demonstration corpus covers company websites, B2B products, industry news, social posts, text PDFs, a scanned-PDF fixture, and customs-derived profiles. The corpus manifest records source-relative paths, byte/content hashes, parser state, and provenance. Generated records must carry a synthetic marker.

## Chunk metadata

Every indexable chunk is validated before publication. The canonical fields include:

- `chunk_id`, `document_id`, `entity_id`, `source_id`, `source_type`, and `fact_type`;
- `country_code`, `region`, `hs_code`, and `language`;
- `published_at`, validity window, ingestion time, and source revision;
- content hash, canonical URL or safe source locator, and parser/chunker versions;
- source weight/profile version, confidence, sensitivity, synthetic status, and parse degradation.

IDs and hashes bind BM25, Milvus rows, retrieval traces, Evidence, and reports. Index publication requires a non-empty build with complete metadata. A query compiles typed filters into a controlled expression over approved scalar fields.

## MySQL contract

`db/migrations/001_schema.sql` defines seven foreign-trade dimension/fact tables:

1. `countries`
2. `companies`
3. `hs_codes`
4. `products`
5. `data_sources`
6. `company_products`
7. `trade_records`

Migration/seed credentials are distinct from the query role. `db/init/010_users.sh` grants the query account `SELECT` only. The live Schema Registry reads the deployed schema and combines it with an allowlist of tables, columns, joins, aggregation fields, aliases, and sensitive-field rules.

A query plan must compile to one allowed `SELECT` or controlled CTE. The validator rejects DDL/DML, `SELECT *`, multiple statements, comments used for bypass, system tables, unregistered columns/functions/joins, cartesian joins, and missing policy filters. It injects limits and policy predicates and enforces timeout and scan budgets.

## Evidence and claims

SQL and RAG results normalize to the Evidence schema. Evidence includes stable identity, branch, entity/fact identity, validity, source provenance, confidence, locator, and payload hash. `SqlEvidence` additionally records normalized SQL, fingerprint, schema/dataset version, injected filters, aggregation grain, execution duration, row count, result hash, and raw record locator.

Claims contain text, status, typed fact dimensions, and Evidence IDs. Numeric claims must bind to compatible SQL Evidence. Conflicted or insufficient core evidence cannot be averaged or silently accepted. The public API returns safe Evidence locators rather than private source bodies or full SQL results.

## Evaluation separation

Evaluation cases, reference Evidence, reference claims, reference matches, and business decisions are separate artifacts. Only the question is passed to the runtime adapter. Reference material is used after execution for deterministic scoring and optional judging.

No question, reference claim, reference Evidence, decision label, or reference source excerpt may enter the corpus manifest, BM25/Milvus index, or generation context. `scripts.validate_trade_eval` checks public/private partition overlap and reference-label contamination against indexable corpus content. The private holdout, its references, preflight state, execution rows, and one-shot consumption lock remain under the ignored `data/eval/private/` tree.

The public holdout bundle contains aggregates and frozen hashes only. It intentionally omits `per_query.jsonl`, questions, case IDs, reference payloads, decision labels, and raw judge errors.
