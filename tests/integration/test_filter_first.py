from __future__ import annotations

from trade_agent.retrieval.filters import RetrievalFilter, compile_filter_binding
from trade_agent.retrieval.planner import QueryIntent, RetrievalPlanner, candidate_chunk_ids


def test_dense_candidate_universe_and_bm25_allowed_ids_match() -> None:
    plan = RetrievalPlanner().plan(
        QueryIntent(
            query="Harbor 850440 procurement",
            region="Asia",
            hs_codes=["850440"],
            extraction_confidence=0.9,
        )
    )
    dense_universe = {
        "chunk_asia_hs": {"region": "Asia", "hs_code": "850440"},
        "chunk_other_region": {"region": "Europe", "hs_code": "850440"},
        "chunk_other_hs": {"region": "Asia", "hs_code": "730890"},
    }
    sparse_universe = {
        "chunk_asia_hs": {"region": "Asia", "hs_code": "850440"},
        "chunk_other_region": {"region": "Europe", "hs_code": "850440"},
        "chunk_other_hs": {"region": "Asia", "hs_code": "730890"},
    }
    dense_ids = candidate_chunk_ids(dense_universe, plan.filter)
    sparse_ids = candidate_chunk_ids(sparse_universe, plan.filter)
    assert dense_ids == sparse_ids == {"chunk_asia_hs"}
    assert compile_filter_binding(plan.filter).version == "trade-filter-v1"
