"""Deterministic retrieval metrics computed only from reference and runtime IDs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from trade_agent.evaluation.models import EvaluationCase


@dataclass(frozen=True)
class RetrievalMetrics:
    """One recomputable retrieval score and its supporting trace.

    ``recall_at_10`` retains the conventional public metric name while
    ``cutoff`` records the actual requested cutoff. Recall-style metrics are
    undefined for unanswerable cases and therefore use ``None`` rather than an
    artificial perfect score.
    """

    cutoff: int
    recall_at_10: float | None
    context_precision: float | None
    context_recall: float | None
    reciprocal_rank: float | None
    retrieved_evidence_ids: tuple[str, ...]
    evidence_ranks: Mapping[str, int | None]
    candidate_counts: Mapping[str, int]
    filter_counts: Mapping[str, int]
    degradation: tuple[str, ...]
    status: str | None
    latency_ms: float | None
    retrieval_noise_count: int
    retrieval_noise_rate: float
    clean_unanswerable_retrieval: bool | None


def evaluate_retrieval(
    case: EvaluationCase,
    references: Iterable[Any],
    retrieval_outcome: Any,
    k: int = 10,
) -> RetrievalMetrics:
    """Score ranked evidence IDs against the private references for ``case``.

    Relevant evidence is the union of required reference-evidence records and
    evidence IDs attached to the case's key reference claims. Duplicate runtime
    hits retain their positions but receive relevance credit only once.
    """

    if not isinstance(case, EvaluationCase):
        raise TypeError("case must be an EvaluationCase")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")

    records = tuple(references)
    claim_evidence, required_evidence = _reference_ids(case, records)
    retrieved_ids = tuple(_hit_id(hit) for hit in getattr(retrieval_outcome, "hits", ()))
    evidence_ranks = {
        evidence_id: _first_rank(retrieved_ids, evidence_id)
        for evidence_id in sorted(required_evidence)
    }
    candidate_counts = _candidate_counts(retrieval_outcome, len(retrieved_ids))
    filter_counts = _named_counts(
        retrieval_outcome,
        (
            "filter_candidate_count",
            "filtered_candidate_count",
            "pre_filter_candidate_count",
            "post_filter_candidate_count",
        ),
        "filter_counts",
    )
    degradation = tuple(str(item) for item in getattr(retrieval_outcome, "degradation", ()))
    status = getattr(retrieval_outcome, "status", None)
    latency_ms = _latency(retrieval_outcome)

    if not case.answerable:
        noise_count = len(retrieved_ids[:k])
        return RetrievalMetrics(
            cutoff=k,
            recall_at_10=None,
            context_precision=None,
            context_recall=None,
            reciprocal_rank=None,
            retrieved_evidence_ids=retrieved_ids,
            evidence_ranks=evidence_ranks,
            candidate_counts=candidate_counts,
            filter_counts=filter_counts,
            degradation=degradation,
            status=status,
            latency_ms=latency_ms,
            retrieval_noise_count=noise_count,
            retrieval_noise_rate=0.0 if not retrieved_ids[:k] else 1.0,
            clean_unanswerable_retrieval=noise_count == 0,
        )

    if not required_evidence:
        raise ValueError("answerable cases require at least one reference evidence ID")

    top_ids = retrieved_ids[:k]
    relevant_credit = _unique_relevant_count(top_ids, required_evidence)
    covered_claims = sum(
        bool(set(evidence_ids).intersection(top_ids))
        for claim_id, evidence_ids in claim_evidence.items()
        if claim_id in case.key_claim_ids
    )
    first_relevant_rank = next(
        (rank for rank, evidence_id in enumerate(top_ids, start=1) if evidence_id in required_evidence),
        None,
    )
    return RetrievalMetrics(
        cutoff=k,
        recall_at_10=relevant_credit / len(required_evidence),
        context_precision=relevant_credit / len(top_ids) if top_ids else 0.0,
        context_recall=covered_claims / len(case.key_claim_ids),
        reciprocal_rank=0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
        retrieved_evidence_ids=retrieved_ids,
        evidence_ranks=evidence_ranks,
        candidate_counts=candidate_counts,
        filter_counts=filter_counts,
        degradation=degradation,
        status=status,
        latency_ms=latency_ms,
        retrieval_noise_count=len(top_ids) - relevant_credit,
        retrieval_noise_rate=(len(top_ids) - relevant_credit) / len(top_ids) if top_ids else 0.0,
        clean_unanswerable_retrieval=None,
    )


def _reference_ids(
    case: EvaluationCase, records: tuple[Any, ...]
) -> tuple[dict[str, tuple[str, ...]], set[str]]:
    claim_evidence: dict[str, tuple[str, ...]] = {}
    required_evidence: set[str] = set()
    for record in records:
        if getattr(record, "reference_evidence_set_id", None) != case.reference_evidence_set_id:
            continue
        claim_id = getattr(record, "claim_id", None)
        if claim_id in case.key_claim_ids and hasattr(record, "evidence_ids"):
            evidence_ids = tuple(record.evidence_ids)
            claim_evidence[claim_id] = evidence_ids
            required_evidence.update(evidence_ids)
        if getattr(record, "required", False) is True and hasattr(record, "evidence_id"):
            required_evidence.add(record.evidence_id)
    return claim_evidence, required_evidence


def _hit_id(hit: Any) -> str:
    evidence_id = getattr(hit, "evidence_id", None)
    if evidence_id is None:
        evidence_id = getattr(hit, "chunk_id", None)
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError("each retrieval hit must expose a nonblank evidence_id or chunk_id")
    return evidence_id


def _first_rank(retrieved_ids: tuple[str, ...], evidence_id: str) -> int | None:
    try:
        return retrieved_ids.index(evidence_id) + 1
    except ValueError:
        return None


def _unique_relevant_count(retrieved_ids: tuple[str, ...], required: set[str]) -> int:
    credited: set[str] = set()
    for evidence_id in retrieved_ids:
        if evidence_id in required:
            credited.add(evidence_id)
    return len(credited)


def _candidate_counts(outcome: Any, selected_count: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, attribute in (
        ("dense", "dense_hits"),
        ("sparse", "sparse_hits"),
        ("fused", "fused_hits"),
        ("reranked", "reranked_hits"),
    ):
        candidates = getattr(outcome, attribute, None)
        if candidates is not None:
            counts[name] = len(candidates)
    counts["selected"] = selected_count
    counts.update(_named_counts(outcome, ("candidate_count",), "candidate_counts"))
    return counts


def _named_counts(outcome: Any, names: tuple[str, ...], mapping_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    supplied = getattr(outcome, mapping_name, None)
    if supplied is not None:
        if not isinstance(supplied, Mapping):
            raise ValueError(f"{mapping_name} must be a mapping when present")
        for name, value in supplied.items():
            counts[str(name)] = _valid_count(value, str(name))
    for name in names:
        value = getattr(outcome, name, None)
        if value is not None:
            counts[name] = _valid_count(value, name)
    return counts


def _valid_count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _latency(outcome: Any) -> float | None:
    value = getattr(outcome, "latency_ms", None)
    if value is None:
        value = getattr(outcome, "rerank_latency_ms", None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError("latency_ms must be a nonnegative number")
    return float(value)
