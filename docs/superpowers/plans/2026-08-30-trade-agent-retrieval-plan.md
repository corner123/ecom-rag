# Trade Agent Indexing and Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn canonical trade chunks into real BGE vectors and BM25 postings, persist them in build-scoped Milvus collections, and deliver filter-first Weighted RRF plus BGE reranking, entity resolution, deduplication, and conflict arbitration.

**Architecture:** Each immutable build owns one Milvus collection and one BM25 artifact keyed by identical `chunk_id` values. A typed retrieval plan compiles metadata filters before ANN, runs dense and sparse retrieval concurrently, fuses ranks with auditable source/fact priors, reranks bounded candidates, then normalizes entities and facts before returning Evidence candidates.

**Tech Stack:** pymilvus 2.6.17, Milvus 2.6.22, BAAI/bge-small-zh-v1.5, BAAI/bge-reranker-base, sentence-transformers, rank-bm25, Pydantic v2, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-08-29-foreign-trade-agent-overhaul-design.md`

## Global Constraints

- Run after the foundation/data plan; consume only validated `ChunkRecord` and immutable `BuildManifest`.
- Milvus tests in `tests/milvus` must use the real Compose service; fake clients and skips cannot satisfy the task.
- `source_weight` is a ranking prior, never an automatic truth decision.
- Query filters are typed and allowlisted; no LLM-produced raw Milvus expression is executable.
- All retrieval results carry component ranks/scores, profile versions, build ID, and degradation state.
- Formal comparisons use the same corpus, chunks, queries, candidate budget, and Top-K.

---

### Task 1: Implement a Versioned BGE Embedding Manager

**Files:**
- Create: `trade_agent/index/embeddings.py`
- Create: `trade_agent/index/contracts.py`
- Create: `trade_agent/index/__init__.py`
- Create: `tests/unit/test_embeddings.py`

**Interfaces:**
- Produces: `EmbeddingContract(provider, model_name, revision, dimension, normalized)`.
- Produces: `BgeEmbeddingManager.embed_documents(texts: Sequence[str]) -> np.ndarray`.
- Produces: `BgeEmbeddingManager.embed_query(text: str) -> np.ndarray`.

- [ ] **Step 1: Write normalization and contract tests**

```python
def test_embeddings_are_normalized_and_dimension_is_discovered(manager):
    vectors = manager.embed_documents(["HS 850440 charger", "采购增长"])
    assert vectors.shape == (2, manager.contract.dimension)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert manager.contract.model_name == "BAAI/bge-small-zh-v1.5"
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/unit/test_embeddings.py -q`

Expected: FAIL because embedding manager is absent.

- [ ] **Step 3: Implement lazy model loading and batching**

```python
class BgeEmbeddingManager:
    def __init__(self, model_name: str, revision: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self._model: SentenceTransformer | None = None

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._load().encode(list(texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)
```

Model revision and observed dimension are persisted in the build contract. Hash/test embeddings may exist only behind an explicit `TEST_EMBEDDING_PROVIDER=deterministic` setting and never satisfy Milvus smoke or final evaluation.

- [ ] **Step 4: Run tests**

Run: `docker compose run --rm api pytest tests/unit/test_embeddings.py -q`

Expected: PASS and model contract records a nonzero dimension.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/index tests/unit/test_embeddings.py
git commit -m "feat: add versioned BGE embeddings"
```

### Task 2: Implement Build-Scoped Milvus Storage and Contract Validation

**Files:**
- Create: `trade_agent/index/milvus_store.py`
- Create: `tests/unit/test_milvus_contract.py`
- Create: `tests/milvus/test_milvus_roundtrip.py`

**Interfaces:**
- Produces: `TradeMilvusStore.create(build: BuildManifest, embedding: EmbeddingContract) -> CollectionContract`.
- Produces: `replace_chunks(chunks: Sequence[ChunkRecord]) -> IndexWriteSummary`.
- Produces: `search(vector: np.ndarray, top_k: int, filter_: RetrievalFilter | None) -> list[DenseHit]`.
- Produces: `validate(expected: CollectionContract) -> CollectionStats`.

- [ ] **Step 1: Write schema/contract and real round-trip tests**

```python
@pytest.mark.milvus
def test_real_milvus_filter_first_roundtrip(real_milvus, embedded_chunks):
    store = real_milvus.create_for_test(embedded_chunks.contract)
    store.replace_chunks(embedded_chunks.records)
    hits = store.search(
        embedded_chunks.query_vector,
        top_k=5,
        filter_=RetrievalFilter(region="North America", hs_codes=["850440"]),
    )
    assert hits
    assert all(hit.metadata.region == "North America" for hit in hits)
    assert all(hit.metadata.hs_code == "850440" for hit in hits)
```

- [ ] **Step 2: Verify unit failure before implementation**

Run: `docker compose run --rm api pytest tests/unit/test_milvus_contract.py -q`

Expected: FAIL because store/contracts are absent.

- [ ] **Step 3: Implement collection creation, scalar fields, HNSW, and strict validation**

The collection name is `trade_intel_chunks_` followed by the first 20 lowercase hexadecimal characters of `build_id`. Materialize every filter field from the spec, store full metadata as JSON, use HNSW/COSINE, and persist build/model/dimension/count/owner metadata in a companion contract collection or collection properties.

```python
def compile_filter(filter_: RetrievalFilter) -> str:
    clauses: list[str] = []
    if filter_.region:
        clauses.append(f"region == {milvus_quote(filter_.region)}")
    if filter_.hs_codes:
        clauses.append(f"hs_code in {milvus_string_list(filter_.hs_codes)}")
    return " and ".join(clauses)
```

The compiler accepts only model fields and escapes values; callers cannot pass raw strings.

- [ ] **Step 4: Run unit and real service tests**

Run: `docker compose up -d etcd minio milvus --wait`

Run: `docker compose run --rm api pytest tests/unit/test_milvus_contract.py -q`

Run: `docker compose run --rm api pytest -m milvus tests/milvus/test_milvus_roundtrip.py -q`

Expected: real insert, flush/load, unfiltered and filtered search, reconnect validation, and test collection cleanup all pass.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/index/milvus_store.py tests/unit/test_milvus_contract.py tests/milvus/test_milvus_roundtrip.py
git commit -m "feat: persist trade chunks in real Milvus"
```

### Task 3: Implement Persistent BM25 with Exact-Identifier Tokenization

**Files:**
- Create: `trade_agent/retrieval/tokenizer.py`
- Create: `trade_agent/retrieval/bm25.py`
- Create: `trade_agent/retrieval/models.py`
- Create: `tests/unit/test_bm25.py`

**Interfaces:**
- Produces: `TradeTokenizer.tokenize(text: str) -> list[str]`.
- Produces: `BM25Index.build(chunks: Sequence[ChunkRecord]) -> BM25Index`.
- Produces: `BM25Index.search(query: str, top_k: int, allowed_chunk_ids: set[str] | None) -> list[SparseHit]`.

- [ ] **Step 1: Write exact identifier tests**

```python
@pytest.mark.parametrize("token", ["850440", "ABC-TRADING-LLC", "C37700", "SKU-GAN65W"])
def test_tokenizer_preserves_trade_identifiers(tokenizer, token):
    assert token.casefold() in tokenizer.tokenize(f"查询 {token} 最近采购")

def test_bm25_prefers_exact_hs_code(index):
    hits = index.search("HS 850440", top_k=3)
    assert hits[0].metadata.hs_code == "850440"
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/unit/test_bm25.py -q`

Expected: FAIL because tokenizer/index do not exist.

- [ ] **Step 3: Implement deterministic tokenization, build, save, and load**

```python
IDENTIFIER = re.compile(r"[A-Za-z]+(?:[-_.][A-Za-z0-9]+)+|[A-Za-z]*\d+[A-Za-z0-9-]*")

def tokenize(text: str) -> list[str]:
    identifiers = [match.group(0).casefold() for match in IDENTIFIER.finditer(text)]
    normalized = normalize_unicode(text)
    return identifiers + cjk_bigrams(normalized) + latin_words(normalized)
```

Persist ordered chunk IDs, tokenized corpus, BM25 parameters, tokenizer version, build ID, and checksum atomically.

- [ ] **Step 4: Run tests**

Run: `docker compose run --rm api pytest tests/unit/test_bm25.py -q`

Expected: PASS with exact HS/model/company ranking assertions.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/retrieval tests/unit/test_bm25.py
git commit -m "feat: add exact-aware BM25 retrieval"
```

### Task 4: Implement Typed Retrieval Planning and Metadata Filtering

**Files:**
- Create: `trade_agent/retrieval/filters.py`
- Create: `trade_agent/retrieval/planner.py`
- Create: `tests/unit/test_retrieval_filters.py`
- Create: `tests/integration/test_filter_first.py`

**Interfaces:**
- Produces: `RetrievalFilter(region, country_codes, hs_codes, entity_ids, source_types, fact_types, published_after, published_before, is_synthetic)`.
- Produces: `RetrievalPlanner.plan(QueryIntent) -> RetrievalPlan`.
- Produces: identical filtering semantics for Milvus pre-filter and BM25 allowed chunk IDs.

- [ ] **Step 1: Write allowlist and filter-consistency tests**

```python
def test_unknown_filter_field_is_rejected():
    with pytest.raises(ValidationError):
        RetrievalFilter.model_validate({"sql": "drop table companies"})

def test_dense_and_sparse_see_same_filtered_ids(filtered_fixture):
    plan = RetrievalPlan(filter=RetrievalFilter(region="Europe", hs_codes=["790700"]))
    assert filtered_fixture.dense_ids(plan) == filtered_fixture.sparse_ids(plan)
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/unit/test_retrieval_filters.py -q`

Expected: FAIL because filter/planner do not exist.

- [ ] **Step 3: Implement filter models and planners**

The planner preserves HS/entity identifiers, normalizes controlled region/country values, rejects contradictory ranges, and records extraction confidence. Low-confidence constraints are not silently applied; they remain unfiltered and are exposed in trace.

- [ ] **Step 4: Run unit and integration tests**

Run: `docker compose run --rm api pytest tests/unit/test_retrieval_filters.py tests/integration/test_filter_first.py -q`

Expected: PASS; trace proves Milvus filter is applied before ANN and BM25 uses the same candidate set.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/retrieval/filters.py trade_agent/retrieval/planner.py tests/unit/test_retrieval_filters.py tests/integration/test_filter_first.py
git commit -m "feat: add filter-first retrieval planning"
```

### Task 5: Implement Auditable Weighted RRF and Retrieval Profiles

**Files:**
- Create: `trade_agent/retrieval/fusion.py`
- Create: `trade_agent/retrieval/profiles.py`
- Create: `trade_agent/config/retrieval_profiles.yaml`
- Create: `tests/unit/test_weighted_rrf.py`

**Interfaces:**
- Produces: `weighted_rrf(rankings: Mapping[str, Sequence[RetrievalHit]], profile: RetrievalProfile) -> list[FusedHit]`.
- Produces: `RetrievalProfile(profile_id, version, rrf_k, retriever_weights, source_fact_priors, candidate_limits)`.

- [ ] **Step 1: Write formula and source-prior boundary tests**

```python
def test_weighted_rrf_uses_rank_not_raw_score():
    result = weighted_rrf(rankings_fixture(), profile_fixture(rrf_k=60))
    assert result[0].components["dense"].rank == 1
    assert result[0].components["bm25"].rank == 2

def test_source_prior_does_not_resolve_conflict():
    hits = weighted_rrf(conflicting_status_rankings(), profile_fixture())
    assert {hit.fact_value for hit in hits[:2]} == {"operating", "closed"}
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/unit/test_weighted_rrf.py -q`

Expected: FAIL because fusion/profile code is absent.

- [ ] **Step 3: Implement formula and versioned YAML loading**

```python
def weighted_rrf(rankings, profile):
    by_chunk: dict[str, FusedAccumulator] = {}
    for retriever, hits in rankings.items():
        alpha = profile.retriever_weights[retriever]
        for rank, hit in enumerate(hits, start=1):
            prior = profile.prior(hit.metadata.source_type, hit.metadata.fact_type)
            by_chunk.setdefault(hit.chunk_id, FusedAccumulator(hit)).add(
                retriever, rank, alpha / (profile.rrf_k + rank), prior
            )
    return finalize_fused_hits(by_chunk, profile)
```

Store both relevance subtotal and bounded source prior contribution so the trace is explainable.

- [ ] **Step 4: Run tests**

Run: `docker compose run --rm api pytest tests/unit/test_weighted_rrf.py -q`

Expected: PASS for exact formula, tie-breaking, missing retriever, and conflict preservation.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/retrieval/fusion.py trade_agent/retrieval/profiles.py trade_agent/config/retrieval_profiles.yaml tests/unit/test_weighted_rrf.py
git commit -m "feat: fuse trade retrieval with weighted RRF"
```

### Task 6: Add BGE Cross-Encoder Reranking and Degradation Semantics

**Files:**
- Create: `trade_agent/retrieval/reranker.py`
- Create: `tests/unit/test_reranker.py`

**Interfaces:**
- Produces: `BgeReranker.rerank(query: str, hits: Sequence[FusedHit], top_k: int) -> RerankOutcome`.
- Produces: `RerankOutcome(hits, model_contract, degraded, error_code, latency_ms)`.

- [ ] **Step 1: Write order and failure tests**

```python
def test_reranker_records_score_and_contract(fake_cross_encoder):
    outcome = BgeReranker(model=fake_cross_encoder).rerank("ABC growth", fused_hits(), 3)
    assert outcome.hits[0].rerank_score is not None
    assert outcome.model_contract.model_name

def test_reranker_failure_is_explicit(failing_cross_encoder):
    outcome = BgeReranker(model=failing_cross_encoder).rerank("q", fused_hits(), 3)
    assert outcome.degraded is True
    assert outcome.error_code == "reranker_unavailable"
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/unit/test_reranker.py -q`

Expected: FAIL because reranker is absent.

- [ ] **Step 3: Implement bounded cross-encoder reranking**

Only the profile's bounded fused candidates reach the model. Stable tie-breaking uses pre-rerank order. A strict profile raises `RerankerUnavailable`; a permissive profile returns explicit degraded results.

- [ ] **Step 4: Run tests with fake and real model smoke**

Run: `docker compose run --rm api pytest tests/unit/test_reranker.py -q`

Run: `docker compose run --rm api python -c 'from trade_agent.retrieval.reranker import smoke; smoke()'`

Expected: tests pass; real model smoke scores two supplied passages.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/retrieval/reranker.py tests/unit/test_reranker.py
git commit -m "feat: rerank bounded trade candidates"
```

### Task 7: Implement Entity Resolution, Deduplication, and Conflict Arbitration

**Files:**
- Create: `trade_agent/entities/models.py`
- Create: `trade_agent/entities/resolver.py`
- Create: `trade_agent/entities/dedup.py`
- Create: `trade_agent/entities/conflicts.py`
- Create: `tests/unit/test_entity_resolution.py`
- Create: `tests/unit/test_deduplication.py`
- Create: `tests/unit/test_conflicts.py`

**Interfaces:**
- Produces: `EntityResolver.resolve(EntityMention) -> ResolutionOutcome`.
- Produces: `EvidenceDeduplicator.cluster(candidates: Sequence[EvidenceCandidate]) -> list[EvidenceCluster]`.
- Produces: `ConflictArbitrator.arbitrate(clusters: Sequence[EvidenceCluster], as_of: datetime) -> ArbitrationOutcome`.

- [ ] **Step 1: Write high-risk entity and temporal conflict tests**

```python
def test_same_name_different_country_is_ambiguous(resolver):
    outcome = resolver.resolve(EntityMention(name="Global Trading", country_code=None))
    assert outcome.status == "ambiguous"

def test_historical_purchase_and_current_closure_can_both_be_true(arbitrator):
    outcome = arbitrator.arbitrate(history_and_closure_clusters(), AS_OF)
    assert {fact.status for fact in outcome.facts} == {"supported"}
    assert outcome.lead_status == "low"
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/unit/test_entity_resolution.py tests/unit/test_deduplication.py tests/unit/test_conflicts.py -q`

Expected: FAIL because entity modules are absent.

- [ ] **Step 3: Implement deterministic-first resolution and explainable arbitration**

Resolution order is registration ID, website domain, exact alias+country, normalized name+country, bounded fuzzy candidates, then ambiguous. Dedup keys combine canonical URL, exact content hash, source lineage, and near-duplicate similarity. Conflict groups use entity/fact/time/unit/grain and produce reason codes for reliability, recency, directness, and independence.

- [ ] **Step 4: Run tests**

Run: `docker compose run --rm api pytest tests/unit/test_entity_resolution.py tests/unit/test_deduplication.py tests/unit/test_conflicts.py -q`

Expected: PASS including same-name rejection, syndication clustering, stale-vs-current, and unresolved high-authority conflict.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/entities tests/unit/test_entity_resolution.py tests/unit/test_deduplication.py tests/unit/test_conflicts.py
git commit -m "feat: normalize and arbitrate trade evidence"
```

### Task 8: Assemble the Retrieval Service and Real Index-Build CLI

**Files:**
- Create: `trade_agent/retrieval/service.py`
- Create: `trade_agent/index/builder.py`
- Create: `scripts/build_trade_index.py`
- Create: `scripts/smoke_milvus_roundtrip.py`
- Create: `tests/integration/test_retrieval_service.py`
- Create: `tests/milvus/test_full_index.py`

**Interfaces:**
- Produces: `TradeIndexBuilder.build(manifest: BuildManifest) -> TradeIndexBundle`.
- Produces: `RetrievalService.retrieve(query: str, plan: RetrievalPlan) -> RetrievalOutcome`.

- [ ] **Step 1: Write end-to-end retrieval assertions**

```python
def test_exact_hs_and_semantic_growth_are_both_retrievable(service):
    exact = service.retrieve("HS850440 ABC Trading", plan(top_k=10))
    semantic = service.retrieve("最近采购活跃度明显提高的客户", plan(top_k=10))
    assert any(hit.metadata.hs_code == "850440" for hit in exact.hits)
    assert any(hit.metadata.fact_type == "purchase_growth" for hit in semantic.hits)
    assert all(hit.trace.build_id == service.build_id for hit in exact.hits + semantic.hits)
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/integration/test_retrieval_service.py -q`

Expected: FAIL because service/builder are absent.

- [ ] **Step 3: Implement index build and retrieval orchestration**

Build embeds chunks in batches, validates no duplicate IDs, writes immutable Milvus and BM25 artifacts, validates counts/contracts, and writes an atomic bundle descriptor. Retrieval performs filter candidate resolution, concurrent sparse/dense calls, WRRF, rerank, diversity cap, entity normalization, dedup, and trace assembly.

- [ ] **Step 4: Run the real index and retrieval suite**

Run: `docker compose up -d etcd minio milvus --wait`

Run: `docker compose run --rm api python -m scripts.build_trade_index --manifest /work/data/manifests/trade-build.json`

Run: `docker compose run --rm api pytest tests/integration/test_retrieval_service.py tests/milvus/test_full_index.py -q`

Run: `docker compose run --rm api python -m scripts.smoke_milvus_roundtrip`

Expected: real Milvus collection and BM25 artifact have identical chunk IDs; exact, semantic, filtered, and mixed queries pass; round-trip cleans its temporary collection.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/index/builder.py trade_agent/retrieval/service.py scripts/build_trade_index.py scripts/smoke_milvus_roundtrip.py tests/integration/test_retrieval_service.py tests/milvus/test_full_index.py
git commit -m "feat: deliver hybrid trade retrieval"
```
