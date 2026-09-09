"""Deterministic classification metrics for conflict-fusion decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class ClassificationMetrics:
    """Recomputable conflict classification and escalation measurements."""

    accuracy: float
    macro_f1: float
    confusion_matrix: Mapping[str, Mapping[str, int]]
    escalation_correctness: float | None
    labels: tuple[str, ...]
    sample_count: int
    abstain_count: int
    refusal_count: int


def conflict_metrics(
    predictions: Sequence[Any], references: Sequence[Any]
) -> ClassificationMetrics:
    """Measure exact conflict labels and the related escalation decisions."""
    predicted = tuple(predictions)
    expected = tuple(references)
    if len(predicted) != len(expected):
        raise ValueError("predictions and references must have the same length")

    predicted_labels = tuple(_label(item) for item in predicted)
    reference_labels = tuple(_label(item) for item in expected)
    labels = tuple(sorted(set((*predicted_labels, *reference_labels))))
    matrix = {
        actual: {
            selected: sum(
                ref == actual and pred == selected
                for pred, ref in zip(predicted_labels, reference_labels, strict=True)
            )
            for selected in labels
        }
        for actual in labels
    }
    sample_count = len(expected)
    correct = sum(
        prediction == reference
        for prediction, reference in zip(predicted_labels, reference_labels, strict=True)
    )
    escalation_pairs = tuple(
        (_escalates(prediction, predicted_label), _escalates(reference, reference_label))
        for prediction, reference, predicted_label, reference_label in zip(
            predicted, expected, predicted_labels, reference_labels, strict=True
        )
    )

    return ClassificationMetrics(
        accuracy=correct / sample_count if sample_count else 0.0,
        macro_f1=_macro_f1(predicted_labels, reference_labels, labels),
        confusion_matrix=MappingProxyType(
            {key: MappingProxyType(value) for key, value in matrix.items()}
        ),
        escalation_correctness=(
            sum(prediction == reference for prediction, reference in escalation_pairs)
            / len(escalation_pairs)
            if escalation_pairs
            else None
        ),
        labels=labels,
        sample_count=sample_count,
        abstain_count=sum(label == "abstain" for label in predicted_labels),
        refusal_count=sum(label == "refuse" for label in predicted_labels),
    )


def _label(item: Any) -> str:
    value = item if isinstance(item, str) else _read(
        item, "label", "status", "decision", "prediction", "reference"
    )
    if not isinstance(value, str) or not value.strip():
        raise ValueError("each conflict item must expose a nonblank label")
    return value.strip().lower()


def _escalates(item: Any, label: str) -> bool:
    explicit = _read(
        item,
        "requires_escalation",
        "escalation_required",
        "escalate",
        default=None,
    )
    if explicit is not None:
        if not isinstance(explicit, bool):
            raise ValueError("escalation decisions must be boolean")
        return explicit
    return label in {"abstain", "conflict", "escalate"}


def _macro_f1(
    predictions: tuple[str, ...], references: tuple[str, ...], labels: tuple[str, ...]
) -> float:
    if not labels:
        return 0.0
    scores: list[float] = []
    for label in labels:
        true_positive = sum(
            prediction == label and reference == label
            for prediction, reference in zip(predictions, references, strict=True)
        )
        false_positive = sum(
            prediction == label and reference != label
            for prediction, reference in zip(predictions, references, strict=True)
        )
        false_negative = sum(
            prediction != label and reference == label
            for prediction, reference in zip(predictions, references, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(scores) / len(scores)


def _read(item: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(item, Mapping) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default
