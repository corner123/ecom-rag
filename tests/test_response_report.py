from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_core.evaluation.response_report import (
    ResponseReportValidationError,
    aggregate_response_records,
    build_response_comparison_from_artifacts,
    build_response_comparison_report,
    compare_response_records,
    load_report_metadata,
    write_response_report,
)


def _record(
    sample_id: str,
    *,
    answerable: bool,
    score: float | None,
    refused: bool = False,
    retrieval_score: float | None = 0.5,
    attempted: bool | None = None,
    generation_usage=None,
):
    attempted = (answerable and not refused) if attempted is None else attempted
    return {
        "id": sample_id,
        "answerable": answerable,
        "refused": refused,
        "generation_succeeded": answerable and not refused,
        "generation_status": {"attempted": attempted},
        "generation_usage": generation_usage,
        "metrics": {
            "context_precision": score,
            "context_recall": score,
            "faithfulness": score,
            "answer_relevancy": score,
            "answer_correctness": score,
        },
        "deterministic_retrieval_metrics": {
            "hit_at_k": retrieval_score if answerable else None,
            "required_claim_recall_at_k": retrieval_score if answerable else None,
            "mrr": retrieval_score if answerable else None,
            "source_precision_at_k": retrieval_score if answerable else None,
            "source_option_recall": retrieval_score if answerable else None,
        },
        "deterministic_retrieval_metric_error": None,
        "metric_errors": (
            []
            if score is not None or not answerable
            else [{"code": "judge_timeout"}]
        ),
        "retrieval_latency_ms": 10.0,
        "generation_latency_ms": 90.0,
        "total_latency_ms": 100.0,
    }


def test_aggregation_never_turns_missing_judge_scores_into_zero():
    summary = aggregate_response_records(
        [
            _record("a", answerable=True, score=0.8),
            _record("b", answerable=True, score=None),
            _record("c", answerable=False, score=None, refused=True),
        ]
    )

    assert summary["metrics"]["faithfulness"]["mean"] == pytest.approx(0.8)
    assert summary["metrics"]["faithfulness"]["count"] == 1
    assert summary["metrics"]["faithfulness"]["eligible"] == 2
    assert summary["judge_coverage"] == pytest.approx(0.5)
    assert summary["publishable"] is False
    assert summary["metric_failure_codes"] == {"judge_timeout": 1}
    assert summary["refusal_f1"] == 1.0
    assert summary["artifact_complete"] is True
    assert summary["response_metrics_publishable"] is False
    assert summary["deterministic_retrieval_metrics"]["hit_at_k"]["mean"] == 0.5
    assert summary["deterministic_retrieval_metrics"]["hit_at_k"]["eligible"] == 2


def test_retrieval_metrics_exclude_unanswerable_and_preserve_missing():
    answerable_scored = _record(
        "a", answerable=True, score=0.8, retrieval_score=1.0
    )
    answerable_missing = _record(
        "b", answerable=True, score=0.8, retrieval_score=None
    )
    answerable_missing["deterministic_retrieval_metric_error"] = {
        "code": "retrieval_failed"
    }
    unanswerable = _record(
        "n", answerable=False, score=None, refused=True, retrieval_score=99.0
    )
    # A malformed legacy value on an unanswerable record must not enter means.
    unanswerable["deterministic_retrieval_metrics"]["hit_at_k"] = 99.0

    summary = aggregate_response_records(
        [answerable_scored, answerable_missing, unanswerable]
    )

    hit = summary["deterministic_retrieval_metrics"]["hit_at_k"]
    assert hit == {
        "mean": 1.0,
        "count": 1,
        "eligible": 2,
        "coverage": 0.5,
    }
    assert summary["deterministic_retrieval_coverage"] == 0.5
    assert summary["deterministic_retrieval_metric_failure_codes"] == {
        "retrieval_failed": 1
    }
    assert summary["publishable"] is False


@pytest.mark.parametrize("failure_stage", ["retrieval", "generation"])
def test_infrastructure_failure_prevents_artifact_completion(failure_stage: str):
    failed = _record("failed", answerable=True, score=None, retrieval_score=None)
    failed["generation_succeeded"] = False
    failed["generation_status"] = None
    failed["failure"] = {
        "stage": failure_stage,
        "code": f"{failure_stage}_error",
    }

    summary = aggregate_response_records([failed])

    assert summary["run_execution_coverage"] == 0.0
    assert summary["artifact_complete"] is False
    assert summary["response_metrics_publishable"] is False
    assert summary["infrastructure_failure_codes"] == {
        f"{failure_stage}_error": 1
    }


def test_system_refusal_is_semantic_outcome_not_judge_or_run_failure():
    answered = _record("answered", answerable=True, score=0.8)
    refused = _record(
        "refused",
        answerable=True,
        score=None,
        refused=True,
        attempted=False,
    )
    refused["generation_status"] = {"attempted": False, "status": "refused"}
    refused["metric_errors"] = []

    summary = aggregate_response_records([answered, refused])

    assert summary["run_execution_coverage"] == 1.0
    assert summary["judge_execution_coverage"] == 1.0
    assert summary["judge_score_coverage"] == 1.0
    assert summary["answerable_response_coverage"] == 0.5
    assert summary["artifact_complete"] is True
    assert summary["response_metrics_publishable"] is True


def test_token_usage_aggregation_counts_only_attempted_model_calls():
    answered = _record(
        "a",
        answerable=True,
        score=0.8,
        attempted=True,
        generation_usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_cache_hit_tokens": 40,
        },
    )
    refused_with_stale_legacy_usage = _record(
        "n",
        answerable=False,
        score=None,
        refused=True,
        attempted=False,
        generation_usage={
            "prompt_tokens": 999,
            "completion_tokens": 999,
            "total_tokens": 1998,
        },
    )

    summary = aggregate_response_records(
        [answered, refused_with_stale_legacy_usage]
    )

    usage = summary["generation_usage"]
    assert usage["attempted_call_count"] == 1
    assert usage["reported_usage_count"] == 1
    assert usage["coverage"] == 1.0
    assert usage["tokens"]["prompt_tokens"]["total"] == 100
    assert usage["tokens"]["completion_tokens"]["total"] == 20
    assert usage["tokens"]["total_tokens"]["total"] == 120
    assert usage["tokens"]["prompt_cache_miss_tokens"]["total"] is None


def test_paired_comparison_requires_the_same_ids_and_reports_delta():
    baseline = [
        _record("a", answerable=True, score=0.5, retrieval_score=0.25)
    ]
    candidate = [
        _record("a", answerable=True, score=0.7, retrieval_score=0.75)
    ]
    comparison = compare_response_records(baseline, candidate, bootstrap_repeats=50)
    assert comparison["faithfulness"]["mean_delta"] == pytest.approx(0.2)
    assert comparison["faithfulness"]["paired_count"] == 1
    assert comparison["hit_at_k"]["mean_delta"] == pytest.approx(0.5)
    assert comparison["hit_at_k"]["paired_count"] == 1

    with pytest.raises(ValueError, match="same question ids"):
        compare_response_records(baseline, [_record("b", answerable=True, score=0.7)])


def test_response_report_writes_json_markdown_and_visualization(tmp_path: Path):
    baseline = [
        _record("a", answerable=True, score=0.5),
        _record("n", answerable=False, score=None, refused=True),
    ]
    candidate = [
        _record("a", answerable=True, score=0.7),
        _record("n", answerable=False, score=None, refused=True),
    ]
    report = build_response_comparison_report(
        baseline_name="baseline",
        candidate_name="candidate",
        baseline_records=baseline,
        candidate_records=candidate,
        metadata={"build_id": "build-test"},
    )
    json_path, markdown_path, plot_path = write_response_report(
        report, tmp_path / "comparison"
    )

    assert json.loads(json_path.read_text("utf-8"))["metadata"]["build_id"] == "build-test"
    assert "paired bootstrap" in markdown_path.read_text("utf-8")
    assert plot_path is not None and plot_path.stat().st_size > 1_000


def _artifact(records, **metadata_overrides):
    metadata = {
        "dataset_file_sha256": "file-sha",
        "dataset_canonical_sha256": "canonical-sha",
        "manifest_build_id": "build-test",
        "index_catalog_sha256": "catalog-sha",
        "top_k": 5,
        **metadata_overrides,
    }
    return {
        "schema_version": "engineering-response-experiment/v1",
        "stage": "judged",
        "metadata": metadata,
        "records": records,
    }


def _write_artifact(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_artifact_comparison_validates_identity_and_preserves_user_metadata(
    tmp_path: Path,
):
    baseline_records = [_record("a", answerable=True, score=0.5)]
    candidate_records = [_record("a", answerable=True, score=0.7)]
    baseline_records[0]["question"] = "What changed?"
    candidate_records[0]["question"] = "What changed?"
    baseline_path = _write_artifact(
        tmp_path / "baseline.json", _artifact(baseline_records)
    )
    candidate_path = _write_artifact(
        tmp_path / "candidate.json", _artifact(candidate_records)
    )

    report = build_response_comparison_from_artifacts(
        baseline_path=baseline_path,
        candidate_path=candidate_path,
        baseline_name="before",
        candidate_name="after",
        user_metadata={"suite": "development"},
    )

    inputs = report["metadata"]["comparison_inputs"]
    assert inputs["manifest_build_id"] == "build-test"
    assert inputs["question_count"] == 1
    assert len(inputs["baseline_artifact_sha256"]) == 64
    assert report["metadata"]["user"] == {"suite": "development"}
    assert report["paired_deltas"]["faithfulness"]["mean_delta"] == pytest.approx(
        0.2
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update(stage="generated"), "stage='judged'"),
        (
            lambda payload: payload["metadata"].update(top_k=10),
            "metadata.top_k must match",
        ),
        (
            lambda payload: payload["records"][0].update(id="different"),
            "same question ids",
        ),
        (
            lambda payload: payload["records"][0].update(question="Changed"),
            "question text differs",
        ),
        (
            lambda payload: payload["records"][0].update(answerable=False),
            "answerable label differs",
        ),
    ],
)
def test_artifact_comparison_fails_closed_on_incompatible_inputs(
    tmp_path: Path, mutate, message: str
):
    before = _record("a", answerable=True, score=0.5)
    after = _record("a", answerable=True, score=0.7)
    before["question"] = after["question"] = "Same question"
    baseline = _artifact([before])
    candidate = _artifact([after])
    mutate(candidate)
    baseline_path = _write_artifact(tmp_path / "baseline.json", baseline)
    candidate_path = _write_artifact(tmp_path / "candidate.json", candidate)

    with pytest.raises(ResponseReportValidationError, match=message):
        build_response_comparison_from_artifacts(
            baseline_path=baseline_path,
            candidate_path=candidate_path,
            baseline_name="before",
            candidate_name="after",
        )


def test_metadata_json_accepts_file_and_inline_object(tmp_path: Path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text('{"suite": "holdout"}', encoding="utf-8")

    assert load_report_metadata(metadata_path) == {"suite": "holdout"}
    assert load_report_metadata('{"suite": "development"}') == {
        "suite": "development"
    }
    with pytest.raises(ResponseReportValidationError, match="JSON object"):
        load_report_metadata("[]")
