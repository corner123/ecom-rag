# Synthetic Foreign-Trade Intelligence Agent

> **Evidence boundary:** this repository is a synthetic, non-production engineering demonstration. Its committed evaluation reports use a local CPU hash-token cosine retriever, BM25, and a lexical reranker. They do **not** establish real Milvus/BGE acceptance, generated-answer quality, live buyer coverage, business-decision validity, production readiness, or performance on customer data.

The project combines a read-only MySQL query path with a source-grounded RAG path. It ingests synthetic website, B2B, news, social, PDF, scanned-PDF, and customs-derived records; builds canonical metadata; retrieves with BM25, dense search, weighted RRF, filters, and reranking; and runs a LangGraph workflow that binds each factual claim to reviewable evidence. Redis stores a bounded checkpoint projection.

## What is implemented

- Seven-table foreign-trade schema, deterministic synthetic seed, read-only query role, schema registry, restricted query plans, SQLGlot policy validation, and `SqlEvidence`.
- Source routing, parsing, quarantine, chunking, provenance manifests, entity resolution, deduplication, conflict handling, and canonical metadata.
- Pinned BGE-M3 embedding contract, Milvus collection contract, BM25, filter-first dense search, weighted RRF, pinned BGE reranker contract, and retrieval traces.
- LangGraph SQL/RAG orchestration, evidence validation, claim guard, structured refusal, retry/step/LLM limits, Redis checkpoint projection, FastAPI routes, and CLI commands.
- Public development evaluation, ignored private holdout workflow, hash freeze, leakage checks, immutable reports, and deterministic rule metrics.

The architecture and trust boundaries are described in [docs/architecture.md](docs/architecture.md), the schemas in [docs/data-contract.md](docs/data-contract.md), evaluation in [docs/evaluation.md](docs/evaluation.md), and operating procedures in [docs/operations.md](docs/operations.md). The requirement-by-requirement status is in [docs/completion-audit.md](docs/completion-audit.md).

## Measured synthetic results

The public development cycle contains 44 queries. For the `full_rerank` arm, context precision changed from `0.075231` to `0.194907`, Recall@10 stayed `0.472222`, and there were zero paired recall or precision regressions. The accepted change was explicit calendar-month filtering. These figures come from one sequential local CPU pass and are not neural-retrieval or production latency measurements.

The first private holdout publication contains aggregates only. It records 44 completed retrieval cases, Recall@10 `0.5833333333333334`, and context precision `0.20601851851851852`. Generation was unavailable, so generation is `not_run`; the LLM Judge is `judge_not_run` with null faithfulness and relevance scores. Rule metrics remain authoritative.

- [Development paired report](data/eval/trade_intel/reports_public/cycle-d2f1d18c65a549e2a191a989fd734236/report.md)
- [First holdout aggregate](data/eval/trade_intel/reports_public/report-7cd261cf288521093c50d1083afec460ea2bb2c624cbea16e46abdfe8d70daf8-local-build-a3137d12491fb54f1a058fefa47cc96e-full_rerank-20260910T214832Z-7bb1aa43/report.md)
- [Frozen public holdout snapshot](data/eval/trade_intel/holdout_snapshot.json)

Older ecommerce datasets and reports remain historical artifacts in separate paths. They are never inputs to, or evidence for, the trade-agent results above.

## Local setup

Python 3.12 is the supported local test runtime. Create the environment and install the locked project dependencies:

```sh
uv sync --frozen
.venv/bin/python -c "import trade_agent"
```

The local `.venv` may be created without the `pip` module; the container-only dependency check is listed in [docs/operations.md](docs/operations.md).

Generate the deterministic demonstration corpus in a new directory:

```sh
.venv/bin/python -m trade_agent.cli bootstrap-demo --output demo/trade_intel_seed --clean
```

The service-backed path requires MySQL, Milvus, etcd, MinIO, Redis, the pinned BGE model artifacts, and locally supplied secrets. Follow [docs/operations.md](docs/operations.md); the committed local evaluation reports are not substitutes for those live checks.

## CLI

```text
trade-intel bootstrap-demo [generator options]
trade-intel db migrate|seed [options]
trade-intel ingest [ingestion options]
trade-intel index [index options]
trade-intel query QUESTION [--top-k N] [--api-url URL]
trade-intel eval [evaluation options]
trade-intel smoke [foundation|retrieval]
trade-intel verify-report [report path or --latest ROOT --kind ROLE]
```

The evaluation command runs the synthetic local evaluation adapter. It does not silently switch to Milvus/BGE or online generation.

## Repository audit

Run the deterministic repository and artifact audit without Docker:

```sh
.venv/bin/python -m scripts.verify_repository
```

It verifies required tracked evidence, immutable public report checksums, CLI delegation, the ignored one-shot holdout lock, absence of public holdout rows/labels, removal of legacy ecommerce runtime paths, and tracked-deliverable hygiene. It does not report live infrastructure acceptance. The controller-final Docker, MySQL, Milvus/BGE, Redis, and full workflow gates are listed in the completion audit.
