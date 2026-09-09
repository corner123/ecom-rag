from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_trade_report import main as verify_main
from trade_agent.data.manifest import canonical_json
from trade_agent.evaluation.models import EvaluationSnapshot, RunManifest
from trade_agent.evaluation.report import ReportWriter, verify_report_bundle
from trade_agent.evaluation.runner import EvaluationRun


def _row(run_id: str, case_id: str) -> dict:
    return {
        "result": {
            "run_id": run_id,
            "case_id": case_id,
            "status": "completed",
            "retrieved_evidence_ids": [],
            "produced_claim_ids": [],
            "backend_statuses": {"dense": "available"},
            "latency_ms": 10.0,
        },
        "trace": {
            "backend_statuses": {"dense": "available"},
            "candidate_counts": {},
            "timings_ms": {},
            "degradation": [],
            "hits": [],
            "budget": {"candidate_limit": 100, "top_k": 10},
        },
        "errors": [],
        "metrics": {
            "retrieval": {
                "recall_at_10": 1.0,
                "context_precision": 1.0,
                "evidence_ranks": {},
                "filter_counts": {},
                "retrieval_noise_count": 0,
            },
            "generation": None,
            "fusion": None,
            "business": None,
        },
        "generation_outcome": None,
        "judge": {
            "status": "judge_not_run",
            "scores": None,
            "error": "provider was not configured",
            "coverage": 0.0,
        },
    }


def _run(root: Path, *, role: str = "development") -> EvaluationRun:
    run_id = f"run-{role}"
    path = root / run_id
    path.mkdir(parents=True)
    snapshot = EvaluationSnapshot(
        snapshot_id=f"snapshot-{role}",
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
        backend_statuses={"dense": "available"},
    )
    case_count = 36 if role == "development" else 15
    manifest_doc = {
        "schema_version": "trade-eval-run/v1",
        "run": manifest.model_dump(mode="json"),
        "arm": "dense",
        "created_at": "2026-08-30T12:34:56+00:00",
        "dataset_role": role,
        "case_count": case_count,
        "enabled_components": ["dense"],
        "disabled_components": ["bm25"],
        "actual_components": ["dense"],
        "budget": {"candidate_limit": 100, "top_k": 10},
        "profile": {"arm": "dense", "dense_weight": 1.0},
        "adapter": {
            "backend": "test-adapter",
            "build_id": "build_0123456789abcdef0123456789abcdef",
            "model_id": "test-model",
            "prompt_id": "test-prompt",
            "runtime": {"python": "3.12.11"},
        },
        "evaluator_id": "trade-deterministic-metrics/v1",
        "artifacts": {"per_query": "per_query.jsonl", "aggregate": "aggregate.json"},
    }
    aggregate = {
        "case_count": case_count,
        "rule_metrics": {"recall_at_10": 1.0},
        "judge": {"status": "judge_not_run", "scores": None, "error": "provider was not configured"},
    }
    (path / "manifest.json").write_text(canonical_json(manifest_doc) + "\n", encoding="utf-8")
    (path / "aggregate.json").write_text(canonical_json(aggregate) + "\n", encoding="utf-8")
    (path / "per_query.jsonl").write_text(
        "".join(canonical_json(_row(run_id, f"{role}-case-{index:02d}")) + "\n" for index in range(case_count)),
        encoding="utf-8",
    )
    return EvaluationRun(path=path, manifest=manifest, aggregate=aggregate)


@pytest.fixture
def writer() -> ReportWriter:
    return ReportWriter(
        now=lambda: "20260830T123456Z",
        nonce=lambda: "abcdef12",
    )


@pytest.fixture
def run(tmp_path: Path) -> EvaluationRun:
    return _run(tmp_path / "runs")


@pytest.fixture
def bundle(writer: ReportWriter, run: EvaluationRun, tmp_path: Path):
    return writer.write(run, tmp_path / "reports")


def test_report_writer_refuses_existing_run_directory(writer, run, tmp_path):
    writer.write(run, tmp_path / "reports")
    with pytest.raises(FileExistsError):
        writer.write(run, tmp_path / "reports")


def test_development_bundle_contains_immutable_evidence_and_verifies(bundle):
    expected = {
        "manifest.json",
        "aggregate.json",
        "per_query.jsonl",
        "environment_manifest.json",
        "model_manifest.json",
        "config_manifest.json",
        "error_analysis.json",
        "report.json",
        "report.md",
        "checksums.sha256",
    }
    assert {item.name for item in bundle.root.iterdir()} == expected
    assert bundle.root.name == (
        f"report-{'9' * 64}-build_0123456789abcdef0123456789abcdef-"
        "dense-20260830T123456Z-abcdef12"
    )
    assert bundle.manifest == bundle.root / "manifest.json"
    assert bundle.aggregate == bundle.root / "aggregate.json"
    assert bundle.per_query == bundle.root / "per_query.jsonl"
    assert bundle.environment_manifest == bundle.root / "environment_manifest.json"
    assert bundle.model_manifest == bundle.root / "model_manifest.json"
    assert bundle.config_manifest == bundle.root / "config_manifest.json"
    assert verify_report_bundle(bundle.root).valid is True


def test_modified_report_fails_checksum(bundle):
    bundle.report_json.write_text("{}", encoding="utf-8")
    result = verify_report_bundle(bundle.root)
    assert result.valid is False
    assert "checksum mismatch: report.json" in result.errors


def test_missing_and_extra_files_fail_verification(bundle):
    bundle.error_analysis.unlink()
    (bundle.root / "unexpected.txt").write_text("surprise", encoding="utf-8")
    result = verify_report_bundle(bundle.root)
    assert result.valid is False
    assert "missing file: error_analysis.json" in result.errors
    assert "unexpected file: unexpected.txt" in result.errors


@pytest.mark.parametrize(
    "bad_line, expected_error",
    [
        (None, "duplicate checksum path: report.json"),
        ("0" * 64 + "  ../outside.json\n", "unsafe checksum path: ../outside.json"),
        ("0" * 64 + "  ..\\outside.json\n", "unsafe checksum path: ..\\outside.json"),
    ],
)
def test_duplicate_and_unsafe_checksum_entries_fail(bundle, bad_line, expected_error):
    checksum_path = bundle.checksums
    lines = checksum_path.read_text(encoding="utf-8").splitlines(keepends=True)
    if bad_line is None:
        bad_line = next(line for line in lines if line.endswith("  report.json\n"))
    checksum_path.write_text("".join(lines) + bad_line, encoding="utf-8")
    result = verify_report_bundle(bundle.root)
    assert result.valid is False
    assert expected_error in result.errors


def test_invalid_checksum_encoding_fails_without_crashing(bundle):
    bundle.checksums.write_bytes(b"\xff\xfe")
    result = verify_report_bundle(bundle.root)
    assert result.valid is False
    assert "invalid checksum encoding" in result.errors


def test_holdout_bundle_is_aggregate_only_and_redaction_is_verified(writer, tmp_path):
    run = _run(tmp_path / "runs", role="holdout")
    private_row = run.path / "per_query.jsonl"
    private_row.write_text(
        private_row.read_text(encoding="utf-8")
        + canonical_json({"question": "SECRET HOLDOUT QUESTION", "decision_text": "SECRET LABEL"})
        + "\n",
        encoding="utf-8",
    )
    bundle = writer.write(run, tmp_path / "reports")
    assert bundle.per_query is None
    assert not (bundle.root / "per_query.jsonl").exists()
    public_text = bundle.report_json.read_text(encoding="utf-8") + bundle.report_markdown.read_text(encoding="utf-8")
    assert "SECRET" not in public_text
    assert json.loads(bundle.error_analysis.read_text(encoding="utf-8"))["status"] == "not_applicable"
    assert verify_report_bundle(bundle.root).valid is True

    bundle.report_markdown.write_text(
        bundle.report_markdown.read_text(encoding="utf-8") + "\nQuestion: SECRET HOLDOUT QUESTION\n",
        encoding="utf-8",
    )
    _rewrite_checksum(bundle.checksums, bundle.report_markdown)
    result = verify_report_bundle(bundle.root)
    assert result.valid is False
    assert "holdout redaction failure: report.md" in result.errors


def test_holdout_writer_rejects_private_fields_in_aggregate(writer, tmp_path):
    run = _run(tmp_path / "runs", role="holdout")
    unsafe_aggregate = {**run.aggregate, "question": "SECRET HOLDOUT QUESTION"}
    (run.path / "aggregate.json").write_text(canonical_json(unsafe_aggregate) + "\n", encoding="utf-8")
    unsafe_run = EvaluationRun(run.path, run.manifest, unsafe_aggregate)

    with pytest.raises(ValueError, match="aggregate contains private fields"):
        writer.write(unsafe_run, tmp_path / "reports")


def test_holdout_verifier_rejects_private_report_with_recomputed_checksums(writer, tmp_path):
    bundle = writer.write(_run(tmp_path / "runs", role="holdout"), tmp_path / "reports")
    aggregate = json.loads((bundle.root / "aggregate.json").read_text(encoding="utf-8"))
    aggregate["decision_text"] = "SECRET LABEL"
    (bundle.root / "aggregate.json").write_text(canonical_json(aggregate) + "\n", encoding="utf-8")
    report = json.loads(bundle.report_json.read_text(encoding="utf-8"))
    report["aggregate"] = aggregate
    bundle.report_json.write_text(canonical_json(report) + "\n", encoding="utf-8")
    _rewrite_checksum(bundle.checksums, bundle.root / "aggregate.json")
    _rewrite_checksum(bundle.checksums, bundle.report_json)

    result = verify_report_bundle(bundle.root)
    assert result.valid is False
    assert "holdout redaction failure: report.json" in result.errors


def _rewrite_checksum(checksums: Path, artifact: Path) -> None:
    from hashlib import sha256

    replacement = f"{sha256(artifact.read_bytes()).hexdigest()}  {artifact.name}\n"
    lines = checksums.read_text(encoding="utf-8").splitlines(keepends=True)
    checksums.write_text(
        "".join(replacement if line.endswith(f"  {artifact.name}\n") else line for line in lines),
        encoding="utf-8",
    )


def test_cli_verifies_path_and_latest_kind(bundle, capsys):
    assert verify_main([str(bundle.root)]) == 0
    direct = json.loads(capsys.readouterr().out)
    assert direct == {"errors": [], "path": str(bundle.root), "valid": True}

    assert verify_main(["--latest", str(bundle.root.parent), "--kind", "development"]) == 0
    latest = json.loads(capsys.readouterr().out)
    assert latest["path"] == str(bundle.root)
