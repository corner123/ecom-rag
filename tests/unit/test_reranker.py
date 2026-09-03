from __future__ import annotations

from typing import Any

import pytest

from trade_agent.retrieval.fusion import FusedHit
from trade_agent.retrieval.reranker import BgeReranker


def fused_hits() -> list[FusedHit]:
    def hit(chunk_id: str) -> FusedHit:
        return FusedHit(
            chunk_id=chunk_id,
            record={"content": f"{chunk_id} trade evidence"},
            score=0.1,
            rank=1,
            relevance_subtotal=0.1,
            source_prior=1.0,
            prior_contribution=0.1,
            profile_id="balanced-v1",
            profile_version="trade-source-profiles-v1",
            components={},
        )

    return [hit("second"), hit("first"), hit("third")]


def fake_contract():
    from trade_agent.retrieval.reranker import RerankerContract

    return RerankerContract(
        model_name="BAAI/bge-reranker-v2-m3",
        revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        max_length=8192,
        library_version="fake-v1",
    )


class FakeCrossEncoder:
    def __init__(self, scores: list[float] | None = None, error: Exception | None = None) -> None:
        self.scores = scores
        self.error = error
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        self.calls.append(list(pairs))
        if self.error is not None:
            raise self.error
        return self.scores or []


def test_reranker_records_score_contract_and_stable_order() -> None:
    model = FakeCrossEncoder([0.1, 0.9, 0.4])
    outcome = BgeReranker(
        model=model,
        contract=fake_contract(),
    ).rerank("ABC growth", fused_hits(), 3)
    assert outcome.degraded is False
    assert outcome.error_code is None
    assert outcome.model_contract.model_name == "BAAI/bge-reranker-v2-m3"
    assert [hit.chunk_id for hit in outcome.hits] == ["first", "third", "second"]
    assert [hit.pre_rerank_rank for hit in outcome.hits] == [2, 3, 1]
    assert outcome.hits[0].rerank_score == 0.9
    assert outcome.hits[0].components == {}
    assert model.calls[0][0] == ("ABC growth", "second trade evidence")


def test_reranker_preserves_pre_rerank_order_for_ties() -> None:
    outcome = BgeReranker(
        model=FakeCrossEncoder([0.5, 0.5, 0.5]),
        contract=fake_contract(),
    ).rerank("q", fused_hits(), 3)
    assert [hit.chunk_id for hit in outcome.hits] == ["second", "first", "third"]


def test_strict_reranker_failure_raises() -> None:
    reranker = BgeReranker(
        model=FakeCrossEncoder(error=RuntimeError("model offline")),
        contract=fake_contract(),
        strict_mode=True,
    )
    from trade_agent.retrieval.reranker import RerankerUnavailable

    with pytest.raises(RerankerUnavailable, match="reranker_unavailable"):
        reranker.rerank("q", fused_hits(), 3)


def test_permissive_reranker_failure_is_explicit() -> None:
    outcome = BgeReranker(
        model=FakeCrossEncoder(error=RuntimeError("model offline")),
        contract=fake_contract(),
        strict_mode=False,
    ).rerank("q", fused_hits(), 3)
    assert outcome.degraded is True
    assert outcome.error_code == "reranker_unavailable"
    assert all(hit.rerank_score is None for hit in outcome.hits)
    assert [hit.chunk_id for hit in outcome.hits] == ["second", "first", "third"]


def test_reranker_rejects_duplicate_candidates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        BgeReranker(model=FakeCrossEncoder([1.0]), contract=fake_contract()).rerank(
            "q",
            [*fused_hits(), fused_hits()[0]],
            3,
        )
