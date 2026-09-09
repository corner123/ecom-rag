from __future__ import annotations

import json
from pathlib import Path

import pytest

from trade_agent.evaluation.error_analysis import HoldoutPolicyError
from trade_agent.evaluation.models import EvaluationSnapshot, RunManifest
from trade_agent.evaluation.optimizer import DevelopmentOptimizer, OptimizationObjective
from trade_agent.evaluation.runner import EvaluationRun


def _candidate(
    root: Path,
    run_id: str,
    values: dict[str, tuple[float, float, float | None, float, str]],
    *,
    dataset_role: str = "development",
    backend_status: str = "available",
) -> EvaluationRun:
    path = root / run_id
    path.mkdir(parents=True)
    snapshot = EvaluationSnapshot(
        snapshot_id=f"snapshot-{run_id}",
        dataset_hash="a" * 64,
        reference_hash="b" * 64,
        corpus_hash="c" * 64,
        index_hash="d" * 64,
        profile_hash="e" * 64,
        model_hash="f" * 64,
        prompt_hash="1" * 64,
        evaluator_hash="2" * 64,
        code_hash="3" * 64,
    )
    manifest = RunManifest(
        run_id=run_id,
        snapshot=snapshot,
        backend_statuses={"dense": backend_status},
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "trade-eval-run/v1",
                "run": manifest.model_dump(mode="json"),
                "dataset_role": dataset_role,
                "arm": "dense",
                "enabled_components": ["dense"],
                "artifacts": {"per_query": "per_query.jsonl", "aggregate": "aggregate.json"},
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for case_id, (recall, precision, faithfulness, latency, status) in values.items():
        rows.append(
            {
                "result": {
                    "run_id": run_id,
                    "case_id": case_id,
                    "status": status,
                    "retrieved_evidence_ids": [],
                    "produced_claim_ids": [],
                    "backend_statuses": {"dense": backend_status},
                    "latency_ms": latency,
                },
                "metrics": {
                    "retrieval": {
                        "recall_at_10": recall,
                        "context_precision": precision,
                    },
                    "generation": None
                    if faithfulness is None
                    else {"faithfulness": {"faithfulness": faithfulness, "policy": "scored"}},
                    "fusion": None,
                    "business": None,
                },
                "judge": {
                    "status": "available",
                    "scores": {"faithfulness": 0.0},
                    "error": None,
                    "coverage": 1.0,
                },
            }
        )
    (path / "per_query.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    aggregate = {"case_count": len(rows)}
    (path / "aggregate.json").write_text(json.dumps(aggregate), encoding="utf-8")
    return EvaluationRun(path=path, manifest=manifest, aggregate=aggregate)


def _optimizer() -> DevelopmentOptimizer:
    """Keep focused fixtures small while production defaults to the 36-case floor."""

    return DevelopmentOptimizer(minimum_case_count=1)


def test_optimizer_prioritizes_paired_recall_then_rule_precision_and_faithfulness(tmp_path: Path) -> None:
    baseline = _candidate(
        tmp_path,
        "baseline",
        {"case-1": (0.5, 0.5, 0.5, 20.0, "completed"), "case-2": (0.5, 0.5, 0.5, 20.0, "completed")},
    )
    precision_candidate = _candidate(
        tmp_path,
        "precision",
        {"case-1": (0.75, 0.8, 0.6, 25.0, "completed"), "case-2": (0.75, 0.8, 0.6, 25.0, "completed")},
    )
    recall_candidate = _candidate(
        tmp_path,
        "recall",
        {"case-1": (1.0, 0.4, 0.4, 25.0, "completed"), "case-2": (1.0, 0.4, 0.4, 25.0, "completed")},
    )

    decision = _optimizer().select(
        [baseline, precision_candidate, recall_candidate],
        OptimizationObjective(
            baseline_run_id="baseline",
            max_mean_latency_increase_ms=10.0,
            max_refusal_rate_increase=0.0,
        ),
    )

    assert decision.selected_run_id == "recall"
    assert decision.paired_deltas["recall_at_10"].delta == 0.5
    assert decision.paired_deltas["recall_at_10"].paired_count == 2
    assert decision.paired_deltas["faithfulness"].delta == pytest.approx(-0.1)
    # Judge faithfulness is deliberately zero in every row; deterministic rule metrics remain authoritative.
    assert decision.selected_metrics["faithfulness"] == 0.4


def test_optimizer_applies_latency_and_refusal_constraints_before_ranking(tmp_path: Path) -> None:
    baseline = _candidate(tmp_path, "baseline", {"case": (0.5, 0.5, 0.5, 20.0, "completed")})
    slow = _candidate(tmp_path, "slow", {"case": (1.0, 1.0, 1.0, 45.0, "completed")})
    unsafe = _candidate(tmp_path, "unsafe", {"case": (1.0, 1.0, 1.0, 20.0, "refused")})
    eligible = _candidate(tmp_path, "eligible", {"case": (0.75, 0.6, 0.6, 25.0, "completed")})

    decision = _optimizer().select(
        [baseline, slow, unsafe, eligible],
        OptimizationObjective(
            baseline_run_id="baseline",
            max_mean_latency_increase_ms=10.0,
            max_refusal_rate_increase=0.0,
        ),
    )

    assert decision.selected_run_id == "eligible"
    assert decision.rejected_candidates == {
        "slow": ("latency_constraint",),
        "unsafe": ("refusal_constraint",),
    }


def test_optimizer_refuses_holdout_input_before_reading_candidate_rows(tmp_path: Path) -> None:
    holdout = _candidate(
        tmp_path,
        "holdout",
        {"private-case": (1.0, 1.0, 1.0, 1.0, "completed")},
        dataset_role="holdout",
    )
    (holdout.path / "per_query.jsonl").unlink()

    with pytest.raises(HoldoutPolicyError, match="holdout"):
        _optimizer().select(
            [holdout],
            OptimizationObjective(baseline_run_id="holdout"),
        )


def test_optimizer_refuses_protected_holdout_path_without_reading_it(tmp_path: Path) -> None:
    protected = tmp_path / "data" / "eval" / "private" / "trade_intel" / "run-1"
    manifest = RunManifest(
        run_id="unreadable-holdout",
        snapshot=EvaluationSnapshot(
            snapshot_id="snapshot-unreadable",
            dataset_hash="a" * 64,
            reference_hash="b" * 64,
            corpus_hash="c" * 64,
            index_hash="d" * 64,
            profile_hash="e" * 64,
            model_hash="f" * 64,
            prompt_hash="1" * 64,
            evaluator_hash="2" * 64,
            code_hash="3" * 64,
        ),
        backend_statuses={"dense": "available"},
    )
    run = EvaluationRun(protected, manifest, {})

    with pytest.raises(HoldoutPolicyError, match="holdout"):
        _optimizer().select(
            [run], OptimizationObjective(baseline_run_id="unreadable-holdout")
        )


def test_no_change_decision_reports_actual_rule_metric_coverage(tmp_path: Path) -> None:
    baseline = _candidate(
        tmp_path,
        "baseline",
        {
            "case-measured": (1.0, 1.0, 0.8, 10.0, "completed"),
            "case-unmeasured": (1.0, 1.0, None, 10.0, "completed"),
        },
    )

    decision = _optimizer().select(
        [baseline], OptimizationObjective(baseline_run_id="baseline")
    )

    assert decision.accepted_change is False
    assert decision.paired_deltas["faithfulness"].paired_count == 1


@pytest.mark.parametrize(
    ("candidate_status", "backend_status"),
    [("failed", "available"), ("completed", "degraded")],
)
def test_optimizer_rejects_unhealthy_candidate_even_when_retrieval_improves(
    tmp_path: Path,
    candidate_status: str,
    backend_status: str,
) -> None:
    baseline = _candidate(
        tmp_path, "baseline", {"case": (0.5, 0.5, 0.5, 20.0, "completed")}
    )
    unhealthy = _candidate(
        tmp_path,
        "unhealthy",
        {"case": (1.0, 1.0, None, 20.0, candidate_status)},
        backend_status=backend_status,
    )

    decision = _optimizer().select(
        [baseline, unhealthy], OptimizationObjective(baseline_run_id="baseline")
    )

    assert decision.selected_run_id == "baseline"
    assert decision.accepted_change is False
    assert decision.rejected_candidates == {"unhealthy": ("execution_constraint",)}


def test_optimizer_enforces_development_sample_floor(tmp_path: Path) -> None:
    baseline = _candidate(
        tmp_path, "baseline", {"case": (1.0, 1.0, 1.0, 20.0, "completed")}
    )

    with pytest.raises(ValueError, match="at least 36"):
        DevelopmentOptimizer().select(
            [baseline], OptimizationObjective(baseline_run_id="baseline")
        )
