"""Deterministic claim grounding metrics for generated answers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class FaithfulnessMetrics:
    """Claim-level grounding result with enough identities for recomputation."""

    faithfulness: float | None
    policy: Literal["scored", "not_scored_no_checked_answer", "not_scored_no_factual_claims"]
    checked_count: int
    supported_count: int
    checked_factual_claim_ids: tuple[str, ...]
    supported_factual_claim_ids: tuple[str, ...]
    unsupported_factual_claim_ids: tuple[str, ...]
    invalid_citation_ids: tuple[str, ...]


def evidence_coverage(answer: Any) -> float:
    """Return bound factual claims divided by all factual claims."""
    factual = tuple(claim for claim in _claims(answer) if _is_factual(claim))
    if not factual:
        return 0.0
    bound = sum(bool(_evidence_ids(claim)) for claim in factual)
    return bound / len(factual)


def faithfulness(answer: Any, evidence: Any, guard: Any) -> FaithfulnessMetrics:
    """Measure factual claims retained by the guard and bound to valid evidence."""
    if _read(guard, "accepted", default=False) is not True or _read(
        answer, "refusal_reason", default=None
    ) is not None:
        return _unscored("not_scored_no_checked_answer")

    factual = tuple(claim for claim in _claims(answer) if _is_factual(claim))
    if not factual:
        return _unscored("not_scored_no_factual_claims")

    evidence_ids = frozenset(_evidence_id(item) for item in _sequence(evidence, "evidence"))
    guarded_claim_ids = frozenset(
        _claim_id(claim, index)
        for index, claim in enumerate(_claims(guard))
        if _is_factual(claim)
    )
    checked_ids: list[str] = []
    supported_ids: list[str] = []
    unsupported_ids: list[str] = []
    invalid_ids: set[str] = set()
    for index, claim in enumerate(factual):
        claim_id = _claim_id(claim, index)
        citations = _evidence_ids(claim)
        invalid = tuple(citation for citation in citations if citation not in evidence_ids)
        invalid_ids.update(invalid)
        checked_ids.append(claim_id)
        supported = (
            _read(claim, "status") == "supported"
            and bool(citations)
            and not invalid
            and claim_id in guarded_claim_ids
        )
        (supported_ids if supported else unsupported_ids).append(claim_id)

    return FaithfulnessMetrics(
        faithfulness=len(supported_ids) / len(checked_ids),
        policy="scored",
        checked_count=len(checked_ids),
        supported_count=len(supported_ids),
        checked_factual_claim_ids=tuple(checked_ids),
        supported_factual_claim_ids=tuple(supported_ids),
        unsupported_factual_claim_ids=tuple(unsupported_ids),
        invalid_citation_ids=tuple(sorted(invalid_ids)),
    )


def _unscored(
    policy: Literal["not_scored_no_checked_answer", "not_scored_no_factual_claims"]
) -> FaithfulnessMetrics:
    return FaithfulnessMetrics(
        faithfulness=None,
        policy=policy,
        checked_count=0,
        supported_count=0,
        checked_factual_claim_ids=(),
        supported_factual_claim_ids=(),
        unsupported_factual_claim_ids=(),
        invalid_citation_ids=(),
    )


def _claims(answer: Any) -> tuple[Any, ...]:
    return _sequence(_read(answer, "claims", default=()), "claims")


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return tuple(value)


def _is_factual(claim: Any) -> bool:
    explicit = _read(claim, "factual", default=None)
    if explicit is not None:
        if not isinstance(explicit, bool):
            raise ValueError("claim factual flags must be boolean")
        return explicit
    return _read(claim, "status", default=None) not in {"analysis", "recommendation"}


def _claim_id(claim: Any, index: int) -> str:
    value = _read(claim, "claim_id", "id", default=None)
    if value is None:
        return f"claim-index-{index}"
    if not isinstance(value, str) or not value.strip():
        raise ValueError("claim IDs must be nonblank strings")
    return value


def _evidence_ids(claim: Any) -> tuple[str, ...]:
    values = _sequence(_read(claim, "evidence_ids", "citations", default=()), "evidence_ids")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("citation IDs must be nonblank strings")
    return values


def _evidence_id(item: Any) -> str:
    value = item if isinstance(item, str) else _read(item, "evidence_id", "id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("evidence items must expose nonblank evidence IDs")
    return value


def _read(item: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(item, Mapping) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default
