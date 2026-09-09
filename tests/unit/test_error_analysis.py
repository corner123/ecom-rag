from __future__ import annotations

import json
from pathlib import Path

import pytest

from trade_agent.evaluation.error_analysis import ErrorAnalyzer, HoldoutPolicyError
from trade_agent.evaluation.models import EvaluationSnapshot, RunManifest
from trade_agent.evaluation.runner import EvaluationRun


def _run(
    root: Path,
    run_id: str,
    arm: str,
    rows: list[dict],
    *,
    dataset_role: str = "development",
) -> EvaluationRun:
    path = root / run_id
    path.mkdir(parents=True)
    snapshot = EvaluationSnapshot(
        snapshot_id=f"snapshot-{run_id}",
        dataset_hash="1" * 64,
        reference_hash="2" * 64,
        corpus_hash="3" * 64,
        index_hash="4" * 64,
        profile_hash="5" * 64,
        model_hash="6" * 64,
        prompt_hash="7" * 64,
        evaluator_hash="8" * 64,
        code_hash="9" * 64,
    )
    manifest = RunManifest(
        run_id=run_id,
        snapshot=snapshot,
        backend_statuses={"dense": "available", "bm25": "available"},
    )
    manifest_doc = {
        "schema_version": "trade-eval-run/v1",
        "run": manifest.model_dump(mode="json"),
        "dataset_role": dataset_role,
        "arm": arm,
        "enabled_components": {
            "dense": ["dense"],
            "bm25": ["bm25"],
            "wrrf": ["dense", "bm25", "wrrf"],
            "wrrf_filter": ["dense", "bm25", "wrrf", "filter"],
            "full_rerank": ["dense", "bm25", "wrrf", "filter", "reranker"],
        }[arm],
        "artifacts": {"per_query": "per_query.jsonl", "aggregate": "aggregate.json"},
    }
    (path / "manifest.json").write_text(json.dumps(manifest_doc), encoding="utf-8")
    (path / "aggregate.json").write_text(json.dumps({"case_count": len(rows)}), encoding="utf-8")
    (path / "per_query.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return EvaluationRun(path=path, manifest=manifest, aggregate={"case_count": len(rows)})


def _row(
    run_id: str,
    case_id: str,
    recall: float | None,
    *,
    precision: float | None = 1.0,
    degradation: list[str] | None = None,
) -> dict:
    return {
        "result": {
            "run_id": run_id,
            "case_id": case_id,
            "status": "completed",
            "retrieved_evidence_ids": [],
            "produced_claim_ids": [],
            "backend_statuses": {"dense": "available", "bm25": "available"},
            "latency_ms": 10.0,
        },
        "trace": {
            "backend_statuses": {"dense": "available", "bm25": "available"},
            "candidate_counts": {},
            "timings_ms": {},
            "degradation": degradation or [],
            "hits": [],
            "budget": {"candidate_limit": 100, "top_k": 10},
        },
        "errors": [],
        "metrics": {
            "retrieval": None
            if recall is None
            else {
                "recall_at_10": recall,
                "context_precision": precision,
                "evidence_ranks": {"required-evidence": None if recall < 1 else 1},
                "filter_counts": {},
                "retrieval_noise_count": 0,
            },
            "generation": None,
            "fusion": None,
            "business": None,
        },
        "generation_outcome": None,
        "judge": {"status": "judge_not_run", "scores": None, "error": None, "coverage": 0.0},
    }


def test_error_analyzer_assigns_actionable_bucket_from_measured_arm_delta(tmp_path: Path) -> None:
    dense = _run(tmp_path, "run-dense", "dense", [_row("run-dense", "hs-case", 0.0)])
    bm25 = _run(tmp_path, "run-bm25", "bm25", [_row("run-bm25", "hs-case", 1.0)])

    analysis = ErrorAnalyzer().analyze([dense, bm25])

    assert len(analysis.cases) == 1
    assert analysis.cases[0].bucket == "keyword_miss"
    assert analysis.cases[0].suggested_component == "tokenizer_or_bm25"
    assert analysis.cases[0].metric_delta == 1.0


def test_error_analyzer_detects_filter_and_reranker_regressions_from_pairs(tmp_path: Path) -> None:
    fused = _run(tmp_path, "run-wrrf", "wrrf", [_row("run-wrrf", "filter-case", 1.0)])
    filtered = _run(
        tmp_path,
        "run-filter",
        "wrrf_filter",
        [_row("run-filter", "filter-case", 0.0)],
    )
    reranked = _run(
        tmp_path,
        "run-rerank",
        "full_rerank",
        [_row("run-rerank", "filter-case", 0.0, precision=0.0)],
    )

    analysis = ErrorAnalyzer().analyze([fused, filtered, reranked])

    assert [(case.run_id, case.bucket) for case in analysis.cases] == [
        ("run-filter", "filter_false_negative"),
        ("run-rerank", "reranker_regression"),
    ]
    assert analysis.bucket_counts == {
        "filter_false_negative": 1,
        "reranker_regression": 1,
    }


def test_error_analyzer_reports_backend_degradation_without_inventing_metric_zero(tmp_path: Path) -> None:
    run = _run(
        tmp_path,
        "run-degraded",
        "dense",
        [_row("run-degraded", "backend-case", None, degradation=["dense:failed"])],
    )

    analysis = ErrorAnalyzer().analyze([run])

    assert analysis.cases[0].bucket == "backend_degraded"
    assert analysis.cases[0].observed_value is None


def test_error_analyzer_refuses_holdout_before_reading_per_query(tmp_path: Path) -> None:
    run = _run(
        tmp_path,
        "run-holdout",
        "dense",
        [_row("run-holdout", "private-case", 0.0)],
        dataset_role="holdout",
    )
    (run.path / "per_query.jsonl").unlink()

    with pytest.raises(HoldoutPolicyError, match="holdout"):
        ErrorAnalyzer().analyze([run])


def test_error_analyzer_reads_actual_conflict_metric_shape(tmp_path: Path) -> None:
    row = _row("run-conflict", "conflict-case", 1.0)
    row["metrics"]["fusion"] = {
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "escalation_correctness": 0.0,
        "sample_count": 1,
    }
    run = _run(tmp_path, "run-conflict", "wrrf", [row])

    analysis = ErrorAnalyzer().analyze([run])

    assert analysis.cases[0].bucket == "conflict_missed"


def test_error_analyzer_maps_full_recall_noise_to_source_prior(tmp_path: Path) -> None:
    run = _run(
        tmp_path,
        "run-noisy",
        "wrrf",
        [_row("run-noisy", "noisy-case", 1.0, precision=0.5)],
    )

    analysis = ErrorAnalyzer().analyze([run])

    assert analysis.cases[0].bucket == "wrong_source_prior"
    assert analysis.cases[0].suggested_component == "fusion_weights"


def test_error_analyzer_flags_answer_that_bypasses_rejecting_guard(tmp_path: Path) -> None:
    row = _row("run-unsafe", "unsafe-case", 1.0)
    row["generation_outcome"] = {
        "status": "available",
        "answer": {"claims": [], "refusal_reason": None},
        "guard": {"accepted": False, "claims": [], "refusal_reason": "claim_unsupported"},
    }
    run = _run(tmp_path, "run-unsafe", "wrrf", [row])

    analysis = ErrorAnalyzer().analyze([run])

    assert analysis.cases[0].bucket == "unsafe_decision"
    assert analysis.cases[0].suggested_component == "decision_policy"
