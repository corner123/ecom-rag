from __future__ import annotations

import pytest

from trade_agent.evaluation.business_metrics import lead_precision


def test_synthetic_lead_precision_is_not_human_review_metric() -> None:
    result = lead_precision(
        {"lead_ids": ("A", "B", "C")},
        {"labels": {"A": True, "B": False, "C": True}},
        2,
    )

    assert result.precision == 0.5
    assert result.label_source == "synthetic_reference"
    assert result.recommended_lead_ids == ("A", "B")
    assert result.relevant_lead_ids == ("A",)
    assert result.relevant_count == 1
    assert result.evaluated_count == 2


def test_explicit_human_review_provenance_is_preserved() -> None:
    result = lead_precision(
        ("A", "B"),
        {"labels": {"A": True, "B": True}, "label_source": "human_review"},
        1,
    )

    assert result.precision == 1.0
    assert result.label_source == "human_review"


def test_lead_precision_uses_available_recommendations_and_has_zero_policy() -> None:
    partial = lead_precision(("A",), {"A": True}, 3)
    empty = lead_precision((), {}, 3)

    assert partial.precision == 1.0
    assert partial.evaluated_count == 1
    assert empty.precision == 0.0
    assert empty.evaluated_count == 0


def test_lead_precision_rejects_nonpositive_cutoff() -> None:
    with pytest.raises(ValueError, match="positive"):
        lead_precision(("A",), {"A": True}, 0)
