"""Deterministic business metrics with explicit label provenance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class LeadMetrics:
    """Precision at N with the exact IDs and label source needed to audit it."""

    precision: float
    label_source: Literal["synthetic_reference", "human_review"]
    top_n: int
    evaluated_count: int
    relevant_count: int
    recommended_lead_ids: tuple[str, ...]
    relevant_lead_ids: tuple[str, ...]


def lead_precision(recommendations: Any, labels: Any, top_n: int) -> LeadMetrics:
    """Measure precision among available recommendations up to ``top_n``."""
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    recommended_ids = _recommendation_ids(recommendations)[:top_n]
    label_values, label_source = _labels(labels)
    relevant_ids = tuple(
        lead_id for lead_id in recommended_ids if label_values.get(lead_id) is True
    )
    evaluated_count = len(recommended_ids)
    return LeadMetrics(
        precision=len(relevant_ids) / evaluated_count if evaluated_count else 0.0,
        label_source=label_source,
        top_n=top_n,
        evaluated_count=evaluated_count,
        relevant_count=len(relevant_ids),
        recommended_lead_ids=recommended_ids,
        relevant_lead_ids=relevant_ids,
    )


def _recommendation_ids(value: Any) -> tuple[str, ...]:
    items = _read(value, "lead_ids", "recommendations", default=value)
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise TypeError("recommendations must be a sequence")
    return tuple(_identifier(item) for item in items)


def _labels(value: Any) -> tuple[dict[str, bool], Literal["synthetic_reference", "human_review"]]:
    source = _read(value, "label_source", default=None)
    label_source: Literal["synthetic_reference", "human_review"] = (
        "human_review"
        if source in {"human_review", "human-reviewed", "human_reviewed"}
        else "synthetic_reference"
    )
    items = _read(value, "labels", default=value)
    if isinstance(items, Mapping):
        return {_identifier(key): _boolean_label(label) for key, label in items.items()}, label_source
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise TypeError("labels must be a mapping or sequence")
    result: dict[str, bool] = {}
    for item in items:
        lead_id = _identifier(_read(item, "lead_id", "id", "company_id"))
        result[lead_id] = _boolean_label(_read(item, "relevant", "label", "is_relevant"))
    return result, label_source


def _identifier(item: Any) -> str:
    if not isinstance(item, str):
        item = _read(item, "lead_id", "id", "company_id")
    if not isinstance(item, str) or not item.strip():
        raise ValueError("lead IDs must be nonblank strings")
    return item.strip()


def _boolean_label(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("lead relevance labels must be boolean")
    return value


def _read(item: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(item, Mapping) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default
