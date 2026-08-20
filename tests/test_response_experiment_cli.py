from __future__ import annotations

import json

import pytest

from main import build_parser, main


def _required_response_eval_args() -> list[str]:
    return [
        "engineering-response-eval",
        "--dataset",
        "dataset.jsonl",
        "--snapshot",
        "snapshot.json",
        "--output",
        "artifact.json",
        "--profile-name",
        "baseline",
        "--generate-only",
    ]


def test_response_eval_cli_defaults_to_legacy_sufficiency_profile() -> None:
    args = build_parser().parse_args(_required_response_eval_args())

    assert args.sufficiency_profile == "legacy_exact_slash"
    assert args.support_selection_profile == "legacy_first"


def test_response_eval_cli_accepts_only_versioned_sufficiency_profiles() -> None:
    parser = build_parser()
    args = parser.parse_args(
        _required_response_eval_args()
        + ["--sufficiency-profile", "split_natural_slash_concepts"]
    )
    assert args.sufficiency_profile == "split_natural_slash_concepts"

    with pytest.raises(SystemExit):
        parser.parse_args(
            _required_response_eval_args()
            + ["--sufficiency-profile", "ad-hoc-untracked-policy"]
        )


def test_response_eval_cli_forwards_sufficiency_profile(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_run_response_experiment(**kwargs):
        captured.update(kwargs)
        return {
            "schema_version": "engineering-response-experiment/v1",
            "stage": "generated",
            "metadata": {"experiment_id": "experiment-test"},
            "records": [],
        }

    monkeypatch.setattr(
        "rag_core.evaluation.response_experiment.run_response_experiment",
        fake_run_response_experiment,
    )
    result = main(
        _required_response_eval_args()
        + [
            "--mini-repo",
            ".",
            "--sufficiency-profile",
            "split_natural_slash_concepts",
            "--support-selection-profile",
            "query_aware_diverse",
        ]
    )

    assert result == 0
    assert captured["sufficiency_profile"] == "split_natural_slash_concepts"
    assert captured["support_selection_profile"] == "query_aware_diverse"
    assert json.loads(capsys.readouterr().out)["experiment_id"] == (
        "experiment-test"
    )


def test_response_eval_cli_forwards_safe_retry_source(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_run_response_experiment(**kwargs):
        captured.update(kwargs)
        return {
            "schema_version": "engineering-response-experiment/v1",
            "stage": "judged",
            "metadata": {"experiment_id": "experiment-test"},
            "records": [],
        }

    monkeypatch.setattr(
        "rag_core.evaluation.response_experiment.run_response_experiment",
        fake_run_response_experiment,
    )
    args = [
        "engineering-response-eval",
        "--dataset",
        "dataset.jsonl",
        "--snapshot",
        "snapshot.json",
        "--output",
        "retried.json",
        "--profile-name",
        "baseline",
        "--judge-only",
        "--retry-from",
        "partial.json",
    ]

    assert main(args) == 0
    assert captured["retry_from_path"] == "partial.json"
    assert captured["resume"] is False
    assert json.loads(capsys.readouterr().out)["stage"] == "judged"


def test_response_eval_cli_rejects_retry_from_with_legacy_resume() -> None:
    with pytest.raises(SystemExit, match="mutually exclusive"):
        main(
            [
                "engineering-response-eval",
                "--dataset",
                "dataset.jsonl",
                "--snapshot",
                "snapshot.json",
                "--output",
                "retried.json",
                "--profile-name",
                "baseline",
                "--judge-only",
                "--replay",
                "generated.json",
                "--retry-from",
                "partial.json",
                "--resume",
            ]
        )
