from __future__ import annotations

import pytest


@pytest.mark.model
def test_real_bge_reranker_smoke() -> None:
    from trade_agent.retrieval.fusion import FusedHit
    from trade_agent.retrieval.reranker import BgeReranker

    def fused(chunk_id: str, content: str) -> FusedHit:
        return FusedHit(
            chunk_id=chunk_id,
            record={"content": content},
            score=1.0,
            rank=1,
            relevance_subtotal=1.0,
            source_prior=1.0,
            prior_contribution=1.0,
            profile_id="balanced-v1",
            profile_version="trade-source-profiles-v1",
            components={},
        )

    outcome = BgeReranker().rerank(
        "HS 850440 charger procurement growth",
        [
            fused("irrelevant", "Scanned PDF unrelated synthetic policy"),
            fused("relevant", "HS 850440 USB-C charger procurement increased"),
        ],
        2,
    )
    assert outcome.degraded is False
    assert outcome.model_contract.revision == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    assert outcome.hits[0].chunk_id == "relevant"
    assert outcome.hits[0].rerank_score is not None
