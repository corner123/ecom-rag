# Architecture

## Product boundary

This repository implements a synthetic foreign-trade intelligence agent. Every generated company, transaction, product, post, article, document, label, and measured result is synthetic. The system demonstrates contracts and failure behavior; it does not establish operational coverage or business outcomes.

## Runtime topology

```text
client / trade-intel CLI
          |
          v
       FastAPI
          |
          v
   LangGraph coordinator <------ Redis bounded checkpoints
      |              |
      |              +---- RAG plan -> metadata filter
      |                              -> BM25 + Milvus dense
      |                              -> weighted RRF
      |                              -> BGE reranker
      |                              -> RAG Evidence
      |
      +---- SQL plan -> Schema Registry -> SQLGlot policy gate
                                       -> read-only MySQL
                                       -> SqlEvidence
          |
          v
 entity/dedup/conflict fusion -> Evidence Validator -> generator
          -> per-Claim guard -> answer, limitation, or refusal
```

Docker Compose declares the API plus MySQL, Milvus, etcd, MinIO, and Redis. Service health checks establish process availability. `/ready` additionally checks the MySQL schema fingerprint, Milvus collection/build contract, Redis, and active build identity.

## Ingestion and indexing

`trade_agent.data` routes reviewed catalog entries to parsers for website/B2B/news/social structured records, text PDF, and scanned PDF. Parsers either emit validated records or an explicit quarantine result; a degraded parser is recorded rather than hidden. Source-specific chunkers produce `ChunkMetadata`, and the manifest freezes content identity and provenance.

`TradeIndexBuilder` builds BM25 from the same frozen chunks used for dense indexing. The production dense path requires a live, content-verified BGE-M3 manager and a matching Milvus collection contract. Milvus stores vectors plus filterable scalar fields. A bundle is published only after sparse and dense membership and checksum checks agree.

The local evaluation adapter is a separate implementation-time instrument. It uses SHA-256 token vectors, BM25, and lexical reranking over the synthetic corpus. Its reports cannot serve as evidence that the production dense path ran.

## Query orchestration

The intent parser produces a bounded route plan and typed `RetrievalFilter`; an LLM never writes Milvus filter expressions. Independent SQL and RAG branches can execute concurrently. Each branch returns typed evidence or a machine-readable failure. Entity binding, provenance, validity windows, aggregation grain, and conflicts survive fusion.

The Evidence Validator checks the requested entity, geography, time, units, source authority, independence, freshness, and conflict state. One bounded retrieval rewrite is permitted when evidence is insufficient. The generator receives validated evidence only. The Claim Guard then verifies claim/evidence binding and can delete a claim, rewrite with a limitation, retrieve once, or refuse.

## State and failure model

`TradeIntelState` carries references, route plans, evidence IDs, conflicts, budgets, status, and errors. Redis persistence uses an allowlisted projection: it excludes the original question, full SQL result rows, source bodies, and secrets. Checkpoint namespaces include thread and run identity, enforce idempotency, and use TTLs.

Policy, schema, SQL AST, evidence conflict, and unsupported-claim failures are deterministic and are not retried. Only configured transient failures can consume the retry budget. Step, retry, candidate, token, and LLM-call budgets terminate in an explicit fallback or refusal.

## Acceptance boundary

Unit and contract evidence is committed for the components above. The public evaluation artifacts establish only synthetic local retrieval measurements. Live Docker health, seven-table MySQL permissions, real BGE loading, Milvus insert/filter/reconnect/cleanup, Redis recovery, and full SQL+RAG traces require controller-final execution; see [completion-audit.md](completion-audit.md).
