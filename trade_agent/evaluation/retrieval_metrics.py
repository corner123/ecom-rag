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

    Required units are the union of reference Evidence IDs and reviewed
    reference-match keys. Duplicate runtime hits retain their positions but
    receive relevance credit only once.
    """

    if not isinstance(case, EvaluationCase):
        raise TypeError("case must be an EvaluationCase")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")

    records = tuple(references)
    claim_evidence, required_evidence, reference_matches = _reference_targets(case, records)
    hits = tuple(getattr(retrieval_outcome, "hits", ()))
    retrieved_ids = tuple(_hit_id(hit) for hit in hits)
    matched_units = tuple(
        _matched_units(hit, evidence_id, required_evidence, reference_matches)
        for hit, evidence_id in zip(hits, retrieved_ids, strict=True)
    )
    required_units = required_evidence | {
        match.reference_match_id for match in reference_matches
    }
    evidence_ranks = {
        required_key: _first_matching_rank(matched_units, required_key)
        for required_key in sorted(required_units)
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

    if not required_units:
        raise ValueError(
            "answerable cases require at least one reference evidence ID or reference match"
        )

    top_ids = retrieved_ids[:k]
    top_matched_units = matched_units[:k]
    credited_units: set[str] = set()
    relevant_positions = 0
    for units in top_matched_units:
        new_units = units - credited_units
        if new_units:
            relevant_positions += 1
            credited_units.update(new_units)
    relevant_credit = len(credited_units)
    covered_claims = sum(
        any(
            evidence_ranks.get(required_key) is not None
            and evidence_ranks[required_key] <= k
            for required_key in claim_evidence.get(claim_id, ())
        )
        for claim_id in case.key_claim_ids
    )
    first_relevant_rank = next(
        (rank for rank, units in enumerate(top_matched_units, start=1) if units),
        None,
    )
    return RetrievalMetrics(
        cutoff=k,
        recall_at_10=relevant_credit / len(required_units),
        context_precision=relevant_positions / len(top_ids) if top_ids else 0.0,
        context_recall=covered_claims / len(case.key_claim_ids),
        reciprocal_rank=0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
        retrieved_evidence_ids=retrieved_ids,
        evidence_ranks=evidence_ranks,
        candidate_counts=candidate_counts,
        filter_counts=filter_counts,
        degradation=degradation,
        status=status,
        latency_ms=latency_ms,
        retrieval_noise_count=len(top_ids) - relevant_positions,
        retrieval_noise_rate=(len(top_ids) - relevant_positions) / len(top_ids) if top_ids else 0.0,
        clean_unanswerable_retrieval=None,
    )


def _reference_targets(
    case: EvaluationCase, records: tuple[Any, ...]
) -> tuple[dict[str, tuple[str, ...]], set[str], tuple[Any, ...]]:
    claim_evidence: dict[str, tuple[str, ...]] = {}
    required_evidence: set[str] = set()
    reference_matches: list[Any] = []
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
        if isinstance(getattr(record, "reference_match_id", None), str):
            reference_matches.append(record)
    match_keys = tuple(match.reference_match_id for match in reference_matches)
    for claim_id in case.key_claim_ids:
        if not claim_evidence.get(claim_id):
            claim_evidence[claim_id] = match_keys
    return claim_evidence, required_evidence, tuple(reference_matches)


def _hit_id(hit: Any) -> str:
    evidence_id = _read(hit, "evidence_id")
    if evidence_id is None:
        evidence_id = _read(hit, "chunk_id")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError("each retrieval hit must expose a nonblank evidence_id or chunk_id")
    return evidence_id


def _first_matching_rank(matched_units: tuple[frozenset[str], ...], required_key: str) -> int | None:
    return next(
        (rank for rank, units in enumerate(matched_units, start=1) if required_key in units),
        None,
    )


def _matched_units(
    hit: Any,
    evidence_id: str,
    required_evidence: set[str],
    reference_matches: tuple[Any, ...],
) -> frozenset[str]:
    units = {evidence_id} if evidence_id in required_evidence else set()
    direct_match_id = _lookup(_hit_layers(hit), ("reference_match_id",))
    if isinstance(direct_match_id, str):
        units.update(
            match.reference_match_id
            for match in reference_matches
            if match.reference_match_id == direct_match_id
        )
        return frozenset(units)
    candidates = [
        match.reference_match_id
        for match in reference_matches
        if _dimensions_match(hit, match)
    ]
    # A dimensional hit must identify exactly one reviewed match. Direct
    # reference_match_id is available for otherwise ambiguous sources.
    if len(candidates) == 1:
        units.add(candidates[0])
    return frozenset(units)


def _dimensions_match(hit: Any, reference_match: Any) -> bool:
    layers = _hit_layers(hit)
    runtime_branch = _lookup(layers, ("branch",))
    if runtime_branch is not None and _scalar(runtime_branch) != reference_match.branch:
        return False

    if reference_match.branch == "sql" and reference_match.sql_dimensions:
        dimensions = _runtime_dimensions(layers)
        return all(
            _scalar(
                dimensions.get(name, _lookup(layers, _sql_dimension_aliases(name)))
            )
            == _scalar(expected)
            for name, expected in reference_match.sql_dimensions.items()
        )

    matched_dimensions = 0
    for expected, runtime_names in (
        (reference_match.source_type, ("source_type",)),
        (reference_match.entity, ("entity", "company", "company_name")),
        (reference_match.event, ("event",)),
        *_event_dimensions(reference_match.event),
    ):
        runtime_value = _lookup(layers, runtime_names)
        if runtime_value is None:
            continue
        if _scalar(runtime_value) != _scalar(expected):
            return False
        matched_dimensions += 1

    matched_identity = False
    identities = (
        ("path", ("path", "source_path", "relative_path")),
        ("chunk_hash", ("chunk_hash", "content_hash")),
        ("canonical_url", ("canonical_url",)),
        ("source_revision", ("source_revision", "parent_document_hash")),
    )
    for reference_name, runtime_names in identities:
        runtime_value = _lookup(layers, runtime_names)
        if runtime_value is None:
            continue
        if _scalar(runtime_value) == _scalar(getattr(reference_match, reference_name)):
            matched_identity = True
    return matched_identity or matched_dimensions >= 2


def _event_dimensions(event: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    parts = event.split(":")
    if len(parts) == 2 and len(parts[0]) == 7 and parts[0][4:5] == "-":
        return (
            (parts[0], ("calendar_month",)),
            (parts[1], ("hs_code",)),
        )
    if len(parts) == 4 and parts[0] == "product" and parts[2] == "hs":
        return (
            (parts[1], ("product_id",)),
            (parts[3], ("hs_code",)),
        )
    if len(parts) == 2 and parts[0] == "news":
        return ((parts[1], ("canonical_story_id",)),)
    if len(parts) == 2 and parts[0] == "post":
        return ((parts[1], ("post_id",)),)
    return ()


def _sql_dimension_aliases(name: str) -> tuple[str, ...]:
    if name == "company":
        return ("company", "company_name")
    return (name,)


def _hit_layers(hit: Any) -> tuple[Any, ...]:
    layers: list[Any] = []

    def append(value: Any) -> None:
        if value is None or any(value is item for item in layers):
            return
        layers.append(value)
        source_locator = _read(value, "source_locator")
        if source_locator is not None:
            append(source_locator)
        raw = _read(value, "raw")
        if raw is not None:
            append(raw)
        manifest_locator = _read(value, "manifest_locator")
        if manifest_locator is not None:
            append(manifest_locator)

    append(hit)
    append(_read(hit, "metadata"))
    record = _read(hit, "record")
    if record is not None:
        append(record)
        append(_read(record, "metadata"))
    return tuple(layers)


def _runtime_dimensions(layers: tuple[Any, ...]) -> dict[str, Any]:
    dimensions: dict[str, Any] = {}
    for layer in layers:
        for name in ("sql_dimensions", "aggregation_info", "raw"):
            supplied = _read(layer, name)
            if isinstance(supplied, Mapping):
                dimensions.update(supplied)
        if isinstance(layer, Mapping):
            dimensions.update(layer)
    return dimensions


def _lookup(layers: tuple[Any, ...], names: tuple[str, ...]) -> Any:
    for layer in layers:
        for name in names:
            value = _read(layer, name)
            if value is not None:
                return value
    return None


def _read(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _scalar(value: Any) -> Any:
    enum_value = getattr(value, "value", None)
    return enum_value if enum_value is not None else str(value) if not isinstance(value, (str, int, float, bool)) else value


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
