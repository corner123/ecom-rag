"""Real BGE-M3, BGE reranker and Milvus; failures are never skipped."""
import os
from collections import Counter
import pytest

from trade_agent.retrieval import QueryIntent, RetrievalPlanner, load_retrieval_profile
from trade_agent.retrieval.reranker import BgeReranker
from trade_agent.retrieval.service import RetrievalService

pytestmark = [pytest.mark.integration, pytest.mark.milvus, pytest.mark.model]


@pytest.fixture(scope="module")
def service(full_index_bundle):
    bundle, store, manager = full_index_bundle
    from trade_agent.db.seed import generate_trade_seed
    from trade_agent.entities.models import EntityRecord
    seed = generate_trade_seed()
    countries = {country.id: country.country_code for country in seed.countries}
    registry = [EntityRecord(entity_id=str(company.id), canonical_name=company.company_name,
        normalized_name=company.normalized_name, country_code=countries[company.country_id],
        website_domain=company.website_domain, registration_id=company.registration_id)
        for company in seed.companies]
    return RetrievalService(build=bundle.build, bm25=bundle.bm25, milvus=store,
        embedding_manager=manager, profile=load_retrieval_profile(), entity_registry=registry,
        reranker=BgeReranker(cache_dir=os.environ.get("MODELS__EMBEDDING_CACHE_DIR"),
                             local_files_only=True, max_length=512, batch_size=8))


def test_exact_hs_and_semantic_growth_are_both_retrievable(service):
    planner = RetrievalPlanner()
    exact_plan = planner.plan(QueryIntent(query="HS850440 charger procurement"))
    exact = service.retrieve(exact_plan.query, exact_plan)
    semantic_plan = planner.plan(QueryIntent(query="最近采购活跃度明显提高的客户"))
    semantic = service.retrieve(semantic_plan.query, semantic_plan)
    assert any(hit.metadata.hs_code == "850440" for hit in exact.hits)
    # Current source schema uses trade_activity; purchase_growth is an obsolete plan example.
    assert any(hit.metadata.fact_type.value == "trade_activity" for hit in semantic.hits)
    for outcome in (exact, semantic):
        assert outcome.hits and not outcome.degradation
        assert outcome.rerank_contract.provider == "sentence-transformers"
        assert max(Counter(h.metadata.source_type for h in outcome.hits).values()) <= 3
        assert len({h.trace.dedupe_cluster_id for h in outcome.hits}) == len(outcome.hits)
        for hit in outcome.hits:
            assert hit.trace.build_id == service.build_id
            assert hit.trace.components
            assert hit.trace.profile_version == "trade-source-profiles-v1"
            assert hit.trace.rerank_score is not None


def test_filtered_mixed_query_returns_only_allowed_evidence(service):
    outcome = service.search(QueryIntent(query="HS850440 最近采购增长客户",
        hs_codes=("850440",), source_types=("customs_profile",),
        fact_types=("trade_activity",), is_synthetic=True), top_k=10)
    assert outcome.hits
    assert all(h.metadata.hs_code == "850440" for h in outcome.hits)
    assert all(h.metadata.source_type.value == "customs_profile" for h in outcome.hits)
    assert all(h.metadata.is_synthetic for h in outcome.hits)
    assert all(h.trace.entity_resolution.status == "resolved" for h in outcome.hits)
    assert {h.chunk_id for h in outcome.sparse_hits} <= set(service.build.chunk_ids)
    empty = service.search(QueryIntent(query="HS850440", entity_ids=("nonexistent-buyer",)), top_k=3)
    assert not empty.hits and not empty.dense_hits and not empty.sparse_hits
