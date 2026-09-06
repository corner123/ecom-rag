from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from trade_agent.data.manifest import BuildManifest, canonical_json
from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog
from trade_agent.index.milvus_store import (
    CollectionContract,
    CollectionStats,
    DenseHit,
    IndexWriteSummary,
    chunk_ids_sha256,
    collection_name_for_build_id,
)
from trade_agent.retrieval.service import RetrievalService
from trade_agent.retrieval import (
    BM25Index,
    FusedHit,
    QueryIntent,
    load_retrieval_profile,
)


def _build(tmp_path: Path, *, attributes=None, html=None) -> BuildManifest:
    root = tmp_path / "corpus"
    (root / "manifests").mkdir(parents=True)
    source = root / "site.html"
    source.write_text(
        html or "<h1>Example exporter</h1><p>Verified synthetic trade evidence.</p>"
        "<h1>Gardening</h1><p>Flowers trees green leaves.</p>"
        "<h1>Astronomy</h1><p>Planets orbit distant stars.</p>",
        encoding="utf-8",
    )
    (root / "manifests" / "corpus_manifest.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "path": "site.html",
                        **(attributes or {}),
                        "content_hash": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "expected_entity": "Example exporter",
                        "fact_type": "company_status",
                        "file_type": "html",
                        "source_type": "official_website",
                        "ingested_at": "2026-08-30T00:00:00+00:00",
                        "publish_time": "2026-08-30T00:00:00+00:00",
                        "valid_from": "2026-08-30T00:00:00+00:00",
                        "is_synthetic": True,
                        "locator": {"section": "overview"},
                        "reference_claim_ids": ["CLAIM-1"],
                    }
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        "\n".join(
            [
                "catalog_version: 1",
                f"root: {root.as_posix()}",
                "synthetic_notice: test-only",
                "sources:",
                "  - source_type: official_website",
                "    file_types: [html]",
                "    paths: [site.html]",
                "    url_pattern: https://company-*.example",
            ]
        ),
        encoding="utf-8",
    )
    return IngestionPipeline().run(
        SourceCatalog.from_yaml(catalog_path),
        tmp_path / "build.json",
    )


def _contract(build: BuildManifest) -> CollectionContract:
    return CollectionContract.model_validate(
        {
            "collection_name": collection_name_for_build_id(build.build_id),
            "build_id": build.build_id,
            "build_fingerprint": build.fingerprint,
            "build_chunk_count": len(build.chunks),
            "build_chunk_ids_sha256": chunk_ids_sha256(build.chunk_ids),
            "metadata_schema_version": build.metadata_schema_version,
            "embedding_provider": "sentence-transformers",
            "embedding_model": "BAAI/bge-m3",
            "embedding_requested_revision": "5617a9f61b028005a4858fdac845db406aefb181",
            "embedding_resolved_revision": "5617a9f61b028005a4858fdac845db406aefb181",
            "embedding_dimension": 1024,
            "embedding_normalized": True,
            "embedding_dtype": "float32",
            "embedding_library_version": "3.4.1",
            "embedding_artifact_manifest_sha256": "3a862f1d0a8543acc13e9faa5e6d6d1f916ee609b264960be8337e6ba509856b",
            "embedding_verified_artifact_count": 10,
            "hnsw_m": 32,
            "hnsw_ef_construction": 256,
            "default_ef_search": 128,
        }
    )


class FakeEmbeddingManager:
    def embed_query(self, text: str) -> np.ndarray:
        vector = np.zeros(1024, dtype=np.float32)
        vector[0 if "trade" in text.casefold() else 1] = 1.0
        return vector


class FakeStore:
    def __init__(self, build: BuildManifest) -> None:
        self.contract = _contract(build)
        self.last_filter = "unset"
        self.last_timeout = None
        self.last_top_k = None

    def search(
        self,
        vector: np.ndarray,
        *,
        top_k: int,
        filter_,
        timeout_seconds: float | None = None,
    ) -> tuple[DenseHit, ...]:
        self.last_filter = filter_
        self.last_timeout = timeout_seconds
        self.last_top_k = top_k
        del vector
        return ()


def _service(tmp_path: Path, *, reranker=None):
    build = _build(tmp_path)
    chunks = tuple(snapshot.restore() for snapshot in build.chunks)
    bm25 = BM25Index.build(chunks, build_id=build.build_id)
    store = FakeStore(build)
    manager = FakeEmbeddingManager()
    service = RetrievalService(
        build=build,
        bm25=bm25,
        milvus=store,
        embedding_manager=manager,
        profile=load_retrieval_profile(),
        reranker=reranker,
    )
    return service, build, store


def test_service_validates_one_exact_build_and_emits_fused_trace(tmp_path) -> None:
    service, build, store = _service(tmp_path)
    result = service.search(
        QueryIntent(query="Verified synthetic trade evidence", is_synthetic=True),
        top_k=3,
    )

    assert result.plan.filter.is_synthetic is True
    assert result.build_id == build.build_id
    assert result.filter_expression == 'is_synthetic == {is_synthetic_0}'
    assert store.last_filter == result.plan.filter
    assert "Verified synthetic" in result.fused_hits[0].record.content
    assert result.profile.profile_id == "balanced-v1"
    assert result.reranked_hits is None
    assert result.rerank_degraded is False


def test_service_passes_a_finite_timeout_to_dense_transport(tmp_path) -> None:
    service, _, store = _service(tmp_path)

    service.search(
        QueryIntent(query="Verified synthetic trade evidence", is_synthetic=True),
        top_k=3,
        transport_timeout_seconds=0.25,
    )

    assert store.last_timeout == 0.25


def test_milvus_store_passes_timeout_to_sdk_search(tmp_path) -> None:
    from trade_agent.index.milvus_store import TradeMilvusStore

    build = _build(tmp_path)
    captured = {}

    class Client:
        def search(self, **kwargs):
            captured.update(kwargs)
            return [[]]

    store = object.__new__(TradeMilvusStore)
    store.client = Client()
    store._contract = _contract(build)
    vector = np.zeros(1024, dtype=np.float32)
    vector[0] = 1.0

    assert store.search(
        vector, top_k=3, filter_=None, timeout_seconds=0.25
    ) == ()
    assert captured["timeout"] == 0.25


def test_service_rejects_mismatched_bm25_build(tmp_path) -> None:
    build = _build(tmp_path)
    chunks = tuple(snapshot.restore() for snapshot in build.chunks)
    bm25 = BM25Index.build(
        chunks,
        build_id="build_" + "0" * 32,
    )
    with pytest.raises(ValueError, match="build"):
        RetrievalService(
            build=build,
            bm25=bm25,
            milvus=FakeStore(build),
            embedding_manager=FakeEmbeddingManager(),
            profile=load_retrieval_profile(),
        )


def test_service_optional_reranker_receives_bounded_fused_hits(tmp_path) -> None:
    class FakeReranker:
        def __init__(self) -> None:
            self.hits = None

        def rerank(self, query: str, hits, top_k: int):
            self.hits = tuple(hits)
            first = hits[0]
            from trade_agent.retrieval import (
                RerankedHit,
                RerankOutcome,
                RerankerContract,
            )

            contract = RerankerContract(
                provider="test",
                model_name="BAAI/bge-reranker-v2-m3",
                revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
                max_length=32,
                library_version="test",
            )

            return RerankOutcome(
                hits=(
                    RerankedHit.model_validate(
                        {
                            **first.model_dump(mode="python"),
                            "rerank_score": 9.0,
                            "pre_rerank_rank": first.rank,
                            "rank": 1,
                        }
                    ),
                ),
                model_contract=contract,
                degraded=False,
                error_code=None,
                latency_ms=1.0,
            )

    reranker = FakeReranker()
    service, _, _ = _service(tmp_path, reranker=reranker)
    result = service.search(QueryIntent(query="trade evidence"), top_k=1)

    assert reranker.hits is not None
    assert len(reranker.hits) <= 100
    assert result.reranked_hits is not None
    assert result.reranked_hits[0].rerank_score == 9.0
    assert result.rerank_contract is not None
    assert result.rerank_contract.provider == "test"


def test_retrieve_returns_selected_hits_with_trace_and_explicit_disabled_rerank(tmp_path):
    from trade_agent.retrieval import RetrievalPlanner
    service, build, _ = _service(tmp_path)
    plan = RetrievalPlanner().plan(QueryIntent(query="trade evidence"))
    outcome = service.retrieve(plan.query, plan, top_k=1)
    assert len(outcome.hits) == 1
    hit = outcome.hits[0]
    assert hit.trace.build_id == build.build_id
    assert hit.trace.components["bm25"].raw_score > 0
    assert hit.trace.profile_version == "trade-source-profiles-v1"
    assert outcome.degradation == ("reranker_unavailable",)
    assert hit.trace.dedupe_cluster_id
    assert hit.trace.entity_resolution.status == "unresolved"
    with pytest.raises(ValueError, match="query"):
        service.retrieve("different query", plan)


def test_same_build_sparse_subset_is_rejected(tmp_path):
    build = _build(tmp_path)
    bm25 = BM25Index.build([build.chunks[0].restore()], build_id=build.build_id)
    with pytest.raises(ValueError, match="BM25"):
        RetrievalService(build=build, bm25=bm25, milvus=FakeStore(build),
                         embedding_manager=FakeEmbeddingManager(), profile=load_retrieval_profile())


def test_hybrid_recall_runs_concurrently(tmp_path, monkeypatch):
    from threading import Barrier
    service, _, store = _service(tmp_path)
    barrier = Barrier(2, timeout=3)
    sparse_search = service._bm25.search
    def sparse(*args, **kwargs):
        barrier.wait()
        return sparse_search(*args, **kwargs)
    def dense(*args, **kwargs):
        barrier.wait()
        return ()
    monkeypatch.setattr(service._bm25, "search", sparse)
    monkeypatch.setattr(store, "search", dense)
    assert service.search(QueryIntent(query="trade evidence"), top_k=1).fused_hits


def test_builder_publishes_reloads_and_rejects_tampering(tmp_path):
    from trade_agent.index.builder import TradeIndexBuilder, TradeIndexBundle
    build = _build(tmp_path)
    store = BuildStore(build)
    builder = TradeIndexBuilder(output_dir=tmp_path / "indexes", milvus=store,
                                embedding_manager=BuildManager(build))
    bundle = builder.build(build)
    assert bundle.descriptor_path.is_file()
    loaded = TradeIndexBundle.load(bundle.descriptor_path, milvus=store,
                                  embedding_manager=BuildManager(build))
    assert loaded.build == build
    assert loaded.bm25.chunk_count == 3
    original = bundle.descriptor_path.read_bytes()
    repeated = builder.build(build)
    assert repeated.descriptor == bundle.descriptor
    assert repeated.descriptor_path.read_bytes() == original
    bundle.bm25_path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        TradeIndexBundle.load(bundle.descriptor_path, milvus=store,
                              embedding_manager=BuildManager(build))


def test_builder_does_not_publish_unverified_dense_index(tmp_path):
    from trade_agent.index.builder import TradeIndexBuilder
    build = _build(tmp_path)
    store = BuildStore(build, bad_ids=True)
    with pytest.raises(ValueError, match="dense"):
        TradeIndexBuilder(output_dir=tmp_path / "indexes", milvus=store,
                          embedding_manager=BuildManager(build)).build(build)
    assert not list((tmp_path / "indexes").rglob("bundle.json"))
    assert store.dropped


class BuildManager(FakeEmbeddingManager):
    def __init__(self, build):
        from trade_agent.index.contracts import EmbeddingContract
        self.contract = object()


class BuildStore(FakeStore):
    def __init__(self, build, bad_ids=False):
        super().__init__(build)
        from types import SimpleNamespace
        self.rows = tuple(s.restore() for s in build.chunks)
        self.corrupt_dense = False
        self.corrupt_vector = False
        self.client = SimpleNamespace(has_collection=lambda name: False, query_iterator=self.iterator)
        self.bad_ids = bad_ids
        self.dropped = False
    def iterator(self, **kwargs):
        from types import SimpleNamespace
        from trade_agent.index.milvus_store import materialize_chunk
        vector = np.zeros(1024, dtype=np.float32)
        vector[0] = 1.0
        if self.corrupt_vector:
            vector[0], vector[1] = 0.0, 1.0
        rows = [materialize_chunk(chunk, vector) for chunk in self.rows]
        if self.corrupt_dense:
            rows[0]["hs_code"] = "850440"
        batches = iter([rows, []])
        return SimpleNamespace(next=lambda: next(batches), close=lambda: None)

    def create(self, build, embedding):
        return self.contract
    def open(self, build, embedding):
        return self.contract
    def replace_chunks(self, chunks):
        self.rows = tuple(chunks)
    def validate(self, contract):
        from types import SimpleNamespace
        return SimpleNamespace(row_count=contract.build_chunk_count,
                               exact_chunk_ids=not self.bad_ids, loaded=True,
                               contract_state="complete")
    def drop_owned_collection(self, contract, **kwargs):
        self.dropped = True


def test_reranking_considers_candidates_beyond_final_top_k(tmp_path):
    from trade_agent.retrieval import BgeReranker, RerankerContract
    class Scores:
        def predict(self, pairs, **kwargs):
            return np.arange(len(pairs), dtype=np.float32)
    contract = RerankerContract(provider="test", model_name="BAAI/bge-reranker-v2-m3",
        revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", max_length=32, library_version="test")
    service, build, store = _service(tmp_path, reranker=BgeReranker(model=Scores(), contract=contract))
    hits = tuple(DenseHit(chunk_id=s.chunk_id, record=s.restore(), score=float(3-i), rank=i+1,
        build_id=build.build_id, filter_expression="", filter_expression_version="trade-filter-v1")
        for i, s in enumerate(build.chunks))
    store.search = lambda *args, **kwargs: hits
    result = service.search(QueryIntent(query="trade evidence"), top_k=1)
    assert result.reranked_hits[0].chunk_id == result.fused_hits[-1].chunk_id
    assert len(result.hits) == 1
    assert len(result.hits[0].trace.duplicate_chunk_ids) == 1


def test_dense_payload_tampering_is_rejected(tmp_path):
    service, build, store = _service(tmp_path)
    record = build.chunks[0].restore()
    record.metadata.source_weight = 0.99
    hit = DenseHit(chunk_id=record.metadata.chunk_id, record=record, score=1.0, rank=1,
        build_id=build.build_id, filter_expression="", filter_expression_version="trade-filter-v1")
    store.search = lambda *args, **kwargs: (hit,)
    with pytest.raises(ValueError, match="payload"):
        service.search(QueryIntent(query="trade evidence"), top_k=1)


def test_source_diversity_caps_final_evidence(tmp_path):
    service, build, store = _service(tmp_path)
    service.source_diversity_cap = 1
    store.search = lambda *args, **kwargs: tuple(DenseHit(chunk_id=s.chunk_id,
        record=s.restore(), score=float(3-i), rank=i+1, build_id=build.build_id,
        filter_expression="", filter_expression_version="trade-filter-v1")
        for i, s in enumerate(build.chunks))
    outcome = service.search(QueryIntent(query="trade evidence"), top_k=3)
    assert len(outcome.hits) == 1
    assert len(outcome.diversity_dropped_chunk_ids) == 2


def test_entity_resolution_precedes_cross_document_dedup(tmp_path):
    from trade_agent.entities.models import EntityRecord
    build = _build(tmp_path, attributes={"company": "Exporter", "country_code": "US"})
    store = FakeStore(build)
    store.search = lambda *args, **kwargs: tuple(DenseHit(chunk_id=s.chunk_id,
        record=s.restore(), score=float(3-i), rank=i+1, build_id=build.build_id,
        filter_expression="", filter_expression_version="trade-filter-v1")
        for i, s in enumerate(build.chunks))
    service = RetrievalService(build=build,
        bm25=BM25Index.build([s.restore() for s in build.chunks], build_id=build.build_id),
        milvus=store, embedding_manager=FakeEmbeddingManager(), profile=load_retrieval_profile(),
        entity_registry=[EntityRecord(entity_id="buyer-1", canonical_name="Exporter",
                                     normalized_name="Exporter", country_code="US")])
    outcome = service.search(QueryIntent(query="trade evidence"), top_k=3)
    assert len(outcome.hits) == 1
    assert outcome.hits[0].trace.entity_resolution.entity_id == "buyer-1"
    assert len(outcome.hits[0].trace.duplicate_chunk_ids) == 3
    assert "canonical_url" in outcome.hits[0].trace.dedupe_reasons


def test_bundle_rejects_rechecksummed_token_tampering(tmp_path):
    from trade_agent.index.builder import TradeIndexBuilder, TradeIndexBundle
    build = _build(tmp_path)
    store, manager = BuildStore(build), BuildManager(build)
    bundle = TradeIndexBuilder(output_dir=tmp_path / "indexes", milvus=store,
                               embedding_manager=manager).build(build)
    raw = json.loads(bundle.bm25_path.read_text())
    raw["payload"]["tokenized_corpus"][0] = ["corrupted"]
    raw["checksum"] = hashlib.sha256(canonical_json(raw["payload"]).encode()).hexdigest()
    bundle.bm25_path.write_text(canonical_json(raw))
    descriptor = json.loads(bundle.descriptor_path.read_text())
    descriptor["bm25_sha256"] = hashlib.sha256(bundle.bm25_path.read_bytes()).hexdigest()
    bundle.descriptor_path.write_text(canonical_json(descriptor))
    with pytest.raises(ValueError, match="token"):
        TradeIndexBundle.load(bundle.descriptor_path, milvus=store, embedding_manager=manager)


def test_returned_records_cannot_mutate_the_live_sparse_index(tmp_path):
    service, _, _ = _service(tmp_path)
    first = service.search(QueryIntent(query="trade evidence"), top_k=1)
    first.hits[0].record.metadata.source_weight = 0.99
    again = service.search(QueryIntent(query="trade evidence"), top_k=1)
    assert again.hits[0].metadata.source_weight == 0.5


def test_degraded_reranker_keeps_ranked_evidence_and_explains_failure(tmp_path):
    from trade_agent.retrieval import BgeReranker, RerankerContract
    class FailedModel:
        def predict(self, *args, **kwargs):
            raise RuntimeError("offline failure")
    contract = RerankerContract(provider="test", model_name="BAAI/bge-reranker-v2-m3",
        revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", max_length=32, library_version="test")
    service, _, _ = _service(tmp_path, reranker=BgeReranker(model=FailedModel(),
        contract=contract, strict_mode=False))
    outcome = service.search(QueryIntent(query="trade evidence"), top_k=1)
    assert outcome.hits
    assert outcome.rerank_degraded
    assert outcome.degradation == ("reranker:reranker_unavailable",)
    assert outcome.hits[0].trace.degradation == outcome.degradation
    assert outcome.hits[0].trace.rerank_score is None


def test_builder_reuses_verified_dense_without_reinsertion_or_cleanup(tmp_path):
    from trade_agent.index.builder import TradeIndexBuilder
    build = _build(tmp_path)
    store = BuildStore(build)
    store.client.has_collection = lambda name: True
    def forbidden(*args, **kwargs):
        raise AssertionError("existing dense collection must remain untouched")
    store.create = store.replace_chunks = store.drop_owned_collection = forbidden
    bundle = TradeIndexBuilder(output_dir=tmp_path / "indexes", milvus=store,
                               embedding_manager=BuildManager(build)).build(build)
    assert bundle.bm25.chunk_count == 3


def test_corrupt_existing_bundle_is_rejected_without_repair(tmp_path):
    from trade_agent.index.builder import TradeIndexBuilder
    build = _build(tmp_path)
    store = BuildStore(build)
    builder = TradeIndexBuilder(output_dir=tmp_path / "indexes", milvus=store,
                                embedding_manager=BuildManager(build))
    bundle = builder.build(build)
    bundle.bm25_path.write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        builder.build(build)
    assert bundle.bm25_path.read_text() == "corrupt"
    assert not store.dropped


def test_bundle_rejects_dense_scalar_tampering_even_with_exact_ids(tmp_path):
    from trade_agent.index.builder import TradeIndexBuilder, TradeIndexBundle
    build = _build(tmp_path)
    store, manager = BuildStore(build), BuildManager(build)
    bundle = TradeIndexBuilder(output_dir=tmp_path / "indexes", milvus=store,
                               embedding_manager=manager).build(build)
    store.corrupt_dense = True
    with pytest.raises(ValueError, match="dense.*payload"):
        TradeIndexBundle.load(bundle.descriptor_path, milvus=store, embedding_manager=manager)


def test_bundle_rejects_changed_normalized_vectors(tmp_path):
    from trade_agent.index.builder import TradeIndexBuilder, TradeIndexBundle
    build = _build(tmp_path)
    store, manager = BuildStore(build), BuildManager(build)
    bundle = TradeIndexBuilder(output_dir=tmp_path / "indexes", milvus=store,
                               embedding_manager=manager).build(build)
    store.corrupt_vector = True
    with pytest.raises(ValueError, match="dense.*checksum"):
        TradeIndexBundle.load(bundle.descriptor_path, milvus=store, embedding_manager=manager)


@pytest.fixture(scope="module")
def disjoint_recall_build(tmp_path_factory):
    # 512 lexical matches in 1,025 documents gives positive BM25 IDF. Dense
    # recall uses 512 different documents, reproducing the 1,024-hit union.
    html = "".join(f"<h1>Section{index}</h1><p>{'needle' if index < 512 else 'background'}</p>"
                   for index in range(1025))
    return _build(tmp_path_factory.mktemp("disjoint-recall"), html=html)


@pytest.mark.parametrize("limits, expected", [
    ({"dense": 512, "bm25": 512}, {"dense": 512, "bm25": 512, "output": 100}),
    ({}, {"dense": 100, "bm25": 100, "output": 100}),
    ({"dense": 512, "bm25": 512, "output": 37}, {"dense": 512, "bm25": 512, "output": 37}),
    ({"dense": 512, "bm25": 512, "output": 512}, {"dense": 512, "bm25": 512, "output": 512}),
])
def test_effective_candidate_limits_bound_disjoint_recall_and_trace(disjoint_recall_build, limits, expected):
    from trade_agent.retrieval import BgeReranker, RerankerContract
    build = disjoint_recall_build
    chunks = tuple(snapshot.restore() for snapshot in build.chunks)
    dense_chunks = [chunk for chunk in chunks if "needle" not in chunk.content][:512]
    store = FakeStore(build)
    def dense_search(vector, *, top_k, filter_):
        return tuple(DenseHit(chunk_id=chunk.metadata.chunk_id, record=chunk,
            score=float(512 - index), rank=index + 1, build_id=build.build_id,
            filter_expression="", filter_expression_version="trade-filter-v1")
            for index, chunk in enumerate(dense_chunks[:top_k]))
    store.search = dense_search
    class Scores:
        def predict(self, pairs, **kwargs):
            return np.arange(len(pairs), dtype=np.float32)
    contract = RerankerContract(provider="test", model_name="BAAI/bge-reranker-v2-m3",
        revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", max_length=32, library_version="test")
    profile = load_retrieval_profile().model_copy(update={"candidate_limits": dict(limits)})
    original = profile.model_dump(mode="python")
    service = RetrievalService(build=build,
        bm25=BM25Index.build(chunks, build_id=build.build_id), milvus=store,
        embedding_manager=FakeEmbeddingManager(), profile=profile,
        reranker=BgeReranker(model=Scores(), contract=contract))
    outcome = service.search(QueryIntent(query="needle"), top_k=1)
    assert len(outcome.dense_hits) == expected["dense"]
    assert len(outcome.sparse_hits) == expected["bm25"]
    assert {hit.chunk_id for hit in outcome.dense_hits}.isdisjoint(
        hit.chunk_id for hit in outcome.sparse_hits)
    assert len(outcome.fused_hits) == len(outcome.reranked_hits) == expected["output"]
    assert outcome.hits and not outcome.rerank_degraded
    assert outcome.profile.candidate_limits == expected
    assert service.profile.candidate_limits == expected
    assert profile.model_dump(mode="python") == original


def test_runtime_candidate_budget_caps_recall_fusion_rerank_and_trace(
    disjoint_recall_build,
) -> None:
    from trade_agent.retrieval import BgeReranker, RerankerContract

    build = disjoint_recall_build
    chunks = tuple(snapshot.restore() for snapshot in build.chunks)
    dense_chunks = [chunk for chunk in chunks if "needle" not in chunk.content][:512]
    store = FakeStore(build)
    seen_dense_limits: list[int] = []

    def dense_search(vector, *, top_k, filter_):
        del vector, filter_
        seen_dense_limits.append(top_k)
        return tuple(
            DenseHit(
                chunk_id=chunk.metadata.chunk_id,
                record=chunk,
                score=float(512 - index),
                rank=index + 1,
                build_id=build.build_id,
                filter_expression="",
                filter_expression_version="trade-filter-v1",
            )
            for index, chunk in enumerate(dense_chunks[:top_k])
        )

    store.search = dense_search

    class Scores:
        def predict(self, pairs, **kwargs):
            return np.arange(len(pairs), dtype=np.float32)

    contract = RerankerContract(
        provider="test",
        model_name="BAAI/bge-reranker-v2-m3",
        revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        max_length=32,
        library_version="test",
    )
    profile = load_retrieval_profile().model_copy(
        update={"candidate_limits": {"dense": 100, "bm25": 100, "output": 100}}
    )
    service = RetrievalService(
        build=build,
        bm25=BM25Index.build(chunks, build_id=build.build_id),
        milvus=store,
        embedding_manager=FakeEmbeddingManager(),
        profile=profile,
        reranker=BgeReranker(model=Scores(), contract=contract),
    )

    outcome = service.search(
        QueryIntent(query="needle"), top_k=4, candidate_limit=4
    )

    assert seen_dense_limits == [4]
    assert len(outcome.dense_hits) == 4
    assert len(outcome.sparse_hits) == 4
    assert len(outcome.fused_hits) == len(outcome.reranked_hits) == 4
    assert outcome.profile.candidate_limits == {
        "dense": 4,
        "bm25": 4,
        "output": 4,
    }
    assert service.profile.candidate_limits == {
        "dense": 100,
        "bm25": 100,
        "output": 100,
    }
