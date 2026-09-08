from __future__ import annotations

from types import SimpleNamespace

import pytest

from trade_agent.evaluation.models import (
    EvaluationCase,
    ReferenceClaimRecord,
    ReferenceEvidenceRecord,
)
from trade_agent.evaluation.retrieval_metrics import evaluate_retrieval


def _evidence(index: int, branch: str = "rag") -> str:
    return f"{branch}_" + f"{index:x}" * 64


def _claim(index: int) -> str:
    return "claim_" + f"{index:x}" * 64


def _case(*, answerable: bool = True, claim_ids: tuple[str, ...] | None = None) -> EvaluationCase:
    if claim_ids is None:
        claim_ids = (_claim(1),) if answerable else ()
    return EvaluationCase.validated_fixture(answerable=answerable, key_claim_ids=claim_ids)


def _references(mapping: dict[str, tuple[str, ...]], *, required_evidence: tuple[str, ...] = ()):
    records = [
        ReferenceClaimRecord(
            claim_id=claim_id,
            reference_evidence_set_id="reference-set-development-001",
            evidence_ids=evidence_ids,
            claim_text=f"Reference claim {position}.",
        )
        for position, (claim_id, evidence_ids) in enumerate(mapping.items(), start=1)
    ]
    records.extend(
        ReferenceEvidenceRecord(
            reference_evidence_id=f"reference-evidence-{position}",
            reference_evidence_set_id="reference-set-development-001",
            evidence_id=evidence_id,
            required=True,
        )
        for position, evidence_id in enumerate(required_evidence, start=1)
    )
    return tuple(records)


def _outcome(ids: list[str], **attributes):
    defaults = {
        "hits": tuple(SimpleNamespace(chunk_id=evidence_id) for evidence_id in ids),
        "dense_hits": (),
        "sparse_hits": (),
        "fused_hits": (),
        "reranked_hits": None,
        "degradation": (),
    }
    defaults.update(attributes)
    return SimpleNamespace(**defaults)


def test_recall_precision_and_context_recall_are_hand_calculated() -> None:
    e1, e2, n1, n2 = (_evidence(index) for index in range(1, 5))
    c1, c2 = _claim(1), _claim(2)

    metrics = evaluate_retrieval(
        case=_case(claim_ids=(c1, c2)),
        references=_references({c1: (e1,), c2: (e2,)}, required_evidence=(e1, e2)),
        retrieval_outcome=_outcome([e1, n1, e2, n2]),
        k=10,
    )

    assert metrics.recall_at_10 == 1.0
    assert metrics.context_precision == 0.5
    assert metrics.context_recall == 1.0
    assert metrics.reciprocal_rank == 1.0
    assert metrics.evidence_ranks == {e1: 1, e2: 3}


def test_partial_and_empty_answerable_results_use_explicit_zero_policies() -> None:
    e1, e2, n1 = _evidence(1), _evidence(2), _evidence(3)
    c1, c2 = _claim(1), _claim(2)
    case = _case(claim_ids=(c1, c2))
    references = _references({c1: (e1,), c2: (e2,)})

    partial = evaluate_retrieval(case, references, _outcome([n1, e2]), k=10)
    empty = evaluate_retrieval(case, references, _outcome([]), k=10)

    assert (partial.recall_at_10, partial.context_precision, partial.context_recall) == (0.5, 0.5, 0.5)
    assert partial.reciprocal_rank == 0.5
    assert partial.evidence_ranks == {e1: None, e2: 2}
    assert (empty.recall_at_10, empty.context_precision, empty.context_recall) == (0.0, 0.0, 0.0)
    assert empty.reciprocal_rank == 0.0


def test_duplicate_hits_do_not_inflate_relevance_and_first_rank_wins() -> None:
    e1, noise = _evidence(1), _evidence(2)
    c1 = _claim(1)

    metrics = evaluate_retrieval(
        _case(claim_ids=(c1,)),
        _references({c1: (e1,)}),
        _outcome([e1, e1, noise]),
    )

    assert metrics.retrieved_evidence_ids == (e1, e1, noise)
    assert metrics.context_precision == pytest.approx(1 / 3)
    assert metrics.recall_at_10 == 1.0
    assert metrics.evidence_ranks == {e1: 1}


def test_cutoff_is_applied_without_losing_the_recomputable_trace() -> None:
    e1, e2, noise = _evidence(1), _evidence(2), _evidence(3)
    c1, c2 = _claim(1), _claim(2)

    metrics = evaluate_retrieval(
        _case(claim_ids=(c1, c2)),
        _references({c1: (e1,), c2: (e2,)}),
        _outcome([noise, e1, e2]),
        k=2,
    )

    assert metrics.cutoff == 2
    assert metrics.retrieved_evidence_ids == (noise, e1, e2)
    assert metrics.evidence_ranks == {e1: 2, e2: 3}
    assert (metrics.recall_at_10, metrics.context_precision, metrics.context_recall) == (0.5, 0.5, 0.5)
    assert metrics.reciprocal_rank == 0.5


def test_answerable_case_without_reference_evidence_is_rejected() -> None:
    c1 = _claim(1)

    with pytest.raises(ValueError, match="answerable cases require at least one reference evidence ID"):
        evaluate_retrieval(_case(claim_ids=(c1,)), (), _outcome([]))


def test_unanswerable_cases_measure_clean_retrieval_and_noise_without_perfect_recall() -> None:
    noise1, noise2 = _evidence(1), _evidence(2)

    clean = evaluate_retrieval(_case(answerable=False), (), _outcome([]))
    noisy = evaluate_retrieval(_case(answerable=False), (), _outcome([noise1, noise2]))

    assert clean.recall_at_10 is None
    assert clean.context_recall is None
    assert clean.reciprocal_rank is None
    assert clean.context_precision is None
    assert clean.retrieval_noise_count == 0
    assert clean.retrieval_noise_rate == 0.0
    assert clean.clean_unanswerable_retrieval is True
    assert noisy.retrieval_noise_count == 2
    assert noisy.retrieval_noise_rate == 1.0
    assert noisy.clean_unanswerable_retrieval is False


def test_trace_preserves_candidate_counts_filter_status_degradation_and_latency() -> None:
    e1 = _evidence(1)
    c1 = _claim(1)
    outcome = _outcome(
        [e1],
        dense_hits=(object(), object(), object()),
        sparse_hits=(object(), object()),
        fused_hits=(object(), object()),
        reranked_hits=(object(),),
        filter_candidate_count=17,
        pre_filter_candidate_count=29,
        status="degraded",
        degradation=("reranker:timeout",),
        latency_ms=12.5,
    )

    metrics = evaluate_retrieval(_case(claim_ids=(c1,)), _references({c1: (e1,)}), outcome)

    assert metrics.candidate_counts == {
        "dense": 3,
        "sparse": 2,
        "fused": 2,
        "reranked": 1,
        "selected": 1,
    }
    assert metrics.filter_counts == {"filter_candidate_count": 17, "pre_filter_candidate_count": 29}
    assert metrics.status == "degraded"
    assert metrics.degradation == ("reranker:timeout",)
    assert metrics.latency_ms == 12.5


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_cutoff_must_be_a_positive_integer(k) -> None:
    e1 = _evidence(1)
    c1 = _claim(1)

    with pytest.raises(ValueError, match="positive integer"):
        evaluate_retrieval(_case(claim_ids=(c1,)), _references({c1: (e1,)}), _outcome([e1]), k=k)
