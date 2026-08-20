from __future__ import annotations

import json
from pathlib import Path

import pytest

from main import main


def _record(sample_id: str, score: float) -> dict:
    return {
        "id": sample_id,
        "question": "What is the decision?",
        "answerable": True,
        "refused": False,
        "generation_succeeded": True,
        "metrics": {
            "context_precision": score,
            "context_recall": score,
            "faithfulness": score,
            "answer_relevancy": score,
            "answer_correctness": score,
        },
        "deterministic_retrieval_metrics": {
            "hit_at_k": 1.0,
            "required_claim_recall_at_k": 1.0,
            "mrr": 1.0,
            "source_precision_at_k": 0.2,
            "source_option_recall": 1.0,
        },
        "metric_errors": [],
        "retrieval_latency_ms": 10.0,
        "generation_latency_ms": 90.0,
        "total_latency_ms": 100.0,
    }


def _write(path: Path, score: float) -> None:
    payload = {
        "schema_version": "engineering-response-experiment/v1",
        "stage": "judged",
        "metadata": {
            "dataset_file_sha256": "file-sha",
            "dataset_canonical_sha256": "canonical-sha",
            "manifest_build_id": "build-test",
            "index_catalog_sha256": "catalog-sha",
            "top_k": 5,
        },
        "records": [_record("q1", score)],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_response_report_cli_writes_three_outputs_without_model_calls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    metadata = tmp_path / "metadata.json"
    output = tmp_path / "reports" / "comparison"
    _write(baseline, 0.5)
    _write(candidate, 0.7)
    metadata.write_text('{"suite": "development"}', encoding="utf-8")

    result = main(
        [
            "engineering-response-report",
            "--baseline-artifact",
            str(baseline),
            "--candidate-artifact",
            str(candidate),
            "--baseline-name",
            "before",
            "--candidate-name",
            "after",
            "--output-prefix",
            str(output),
            "--metadata-json",
            str(metadata),
            "--minimum-judge-coverage",
            "0.9",
        ]
    )

    assert result == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["schema_version"] == "engineering-response-comparison/v1"
    assert output.with_suffix(".json").is_file()
    assert output.with_suffix(".md").is_file()
    assert output.with_suffix(".png").stat().st_size > 1_000
    assert json.loads(output.with_suffix(".json").read_text("utf-8"))["metadata"][
        "user"
    ] == {"suite": "development"}

    with pytest.raises(SystemExit, match="refuses to overwrite"):
        main(
            [
                "engineering-response-report",
                "--baseline-artifact",
                str(baseline),
                "--candidate-artifact",
                str(candidate),
                "--baseline-name",
                "before",
                "--candidate-name",
                "after",
                "--output-prefix",
                str(output),
            ]
        )
