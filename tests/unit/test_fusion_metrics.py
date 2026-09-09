from __future__ import annotations

import pytest

from trade_agent.evaluation.fusion_metrics import conflict_metrics


def test_conflict_metrics_report_classification_and_escalation_results() -> None:
    predictions = (
        {"label": "conflict", "requires_escalation": True},
        {"label": "clear", "requires_escalation": False},
        {"label": "abstain", "requires_escalation": True},
        {"label": "refuse", "requires_escalation": False},
    )
    references = (
        {"label": "conflict", "requires_escalation": True},
        {"label": "clear", "requires_escalation": False},
        {"label": "conflict", "requires_escalation": True},
        {"label": "refuse", "requires_escalation": False},
    )

    result = conflict_metrics(predictions, references)

    assert result.accuracy == 0.75
    assert result.macro_f1 == pytest.approx(2 / 3)
    assert result.confusion_matrix["conflict"] == {
        "abstain": 1,
        "clear": 0,
        "conflict": 1,
        "refuse": 0,
    }
    assert result.escalation_correctness == 1.0
    assert result.abstain_count == 1
    assert result.refusal_count == 1
    assert result.sample_count == 4


def test_conflict_metrics_have_explicit_empty_input_policy() -> None:
    result = conflict_metrics((), ())

    assert result.accuracy == 0.0
    assert result.macro_f1 == 0.0
    assert result.escalation_correctness is None
    assert result.confusion_matrix == {}
    assert result.sample_count == 0


def test_conflict_metrics_require_aligned_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        conflict_metrics(("conflict",), ())
