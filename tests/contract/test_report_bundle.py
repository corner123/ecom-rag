from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_trade_report import main as verify_main
from trade_agent.data.manifest import canonical_json
from trade_agent.evaluation.models import EvaluationSnapshot, RunManifest
from trade_agent.evaluation.judge import OptionalJudge
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
        "judge": OptionalJudge().evaluate({}).to_dict(),
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
        "status_counts": {"completed": case_count},
        "retrieval": {
            name: {"value": 1.0, "scored_count": case_count, "total_count": case_count}
            for name in ("recall_at_10", "context_precision", "context_recall", "reciprocal_rank")
        },
        "judge_coverage": 0.0,
        "degradation": [],
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
    published = json.loads(bundle.aggregate.read_text(encoding="utf-8"))
    assert published["judge"] == {
        "coverage": 0.0,
        "error_categories": {"judge_not_run": 36},
        "error_count": 36,
        "errors": ["missing api_key, provider, client, model"],
        "model": None,
        "model_hash": OptionalJudge().evaluate({}).model_hash,
        "prompt_hash": OptionalJudge().evaluate({}).prompt_hash,
        "provider": None,
        "scores": {"faithfulness": None, "relevance": None},
        "status": "judge_not_run",
        "status_counts": {"judge_not_run": 36},
        "temperature": 0.0,
    }
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
    rows = [json.loads(line) for line in private_row.read_text(encoding="utf-8").splitlines()]
    rows[0]["question"] = "SECRET HOLDOUT QUESTION"
    rows[0]["decision_text"] = "SECRET LABEL"
    private_row.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
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


def test_holdout_writer_projects_aggregate_through_strict_allowlist(writer, tmp_path):
    run = _run(tmp_path / "runs", role="holdout")
    unsafe_aggregate = {
        **run.aggregate,
        "question": "SECRET HOLDOUT QUESTION",
        "notes": "Escalate Acme immediately",
        "nested": {"anything": "SECRET REFERENCE CLAIM"},
        "degradation": ["SECRET source excerpt", "Acme:failed", "dense:failed"],
    }
    (run.path / "aggregate.json").write_text(canonical_json(unsafe_aggregate) + "\n", encoding="utf-8")
    unsafe_run = EvaluationRun(run.path, run.manifest, unsafe_aggregate)
    rows = [json.loads(line) for line in (run.path / "per_query.jsonl").read_text().splitlines()]
    for row in rows:
        row["judge"]["errors"] = ["Escalate Acme immediately"]
    (run.path / "per_query.jsonl").write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )

    bundle = writer.write(unsafe_run, tmp_path / "reports")
    published = json.loads(bundle.aggregate.read_text(encoding="utf-8"))
    assert set(published) == {
        "schema_version", "case_count", "status_counts", "retrieval", "judge", "degradation"
    }
    public = bundle.aggregate.read_text(encoding="utf-8") + bundle.report_markdown.read_text(encoding="utf-8")
    assert "SECRET" not in public
    assert "Acme" not in public
    assert published["degradation"] == ["dense:failed"]
    assert published["judge"]["errors"] == []
    assert published["judge"]["error_count"] == 15
    assert published["judge"]["error_categories"] == {"judge_not_run": 15}


def test_failed_judge_preserves_null_scores_errors_and_identifiers(writer, tmp_path):
    run = _run(tmp_path / "runs")

    def fail_client(**_):
        raise RuntimeError("judge unavailable")

    failed = OptionalJudge(
        api_key="test", provider="test-provider", model="judge-v1", client=fail_client
    ).evaluate({}).to_dict()
    rows = [json.loads(line) for line in (run.path / "per_query.jsonl").read_text().splitlines()]
    for row in rows:
        row["judge"] = failed
    (run.path / "per_query.jsonl").write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )

    bundle = writer.write(run, tmp_path / "reports")
    judge = json.loads(bundle.aggregate.read_text(encoding="utf-8"))["judge"]
    assert judge["status"] == "judge_failed"
    assert judge["scores"] == {"faithfulness": None, "relevance": None}
    assert judge["errors"] == ["RuntimeError: judge unavailable"]
    assert {name: judge[name] for name in ("provider", "model", "temperature")} == {
        "provider": "test-provider", "model": "judge-v1", "temperature": 0.0,
    }
    assert len(judge["prompt_hash"]) == len(judge["model_hash"]) == 64
    assert verify_report_bundle(bundle.root).valid is True


def test_completed_real_shape_judge_reports_frozen_temperature(writer, tmp_path):
    run = _run(tmp_path / "runs")
    completed = OptionalJudge(
        api_key="test",
        provider="test-provider",
        model="judge-v1",
        client=lambda **_: '{"faithfulness":0.75,"relevance":1.0}',
    ).evaluate({}).to_dict()
    rows = [json.loads(line) for line in (run.path / "per_query.jsonl").read_text().splitlines()]
    for row in rows:
        row["judge"] = completed
    (run.path / "per_query.jsonl").write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )
    aggregate = {**run.aggregate, "judge_coverage": 1.0}
    (run.path / "aggregate.json").write_text(canonical_json(aggregate) + "\n", encoding="utf-8")
    run = EvaluationRun(run.path, run.manifest, aggregate)

    bundle = writer.write(run, tmp_path / "reports")
    judge = json.loads(bundle.aggregate.read_text(encoding="utf-8"))["judge"]
    assert judge["status"] == "judge_completed"
    assert judge["scores"] == {"faithfulness": 0.75, "relevance": 1.0}
    assert judge["temperature"] == 0.0
    assert judge["provider"] == "test-provider"
    assert judge["model"] == "judge-v1"
    assert judge["error_count"] == 0
    assert verify_report_bundle(bundle.root).valid is True


@pytest.mark.parametrize("judge_update", [
    {
        "status": "judge_completed", "status_counts": {"judge_failed": 15},
        "scores": {"faithfulness": None, "relevance": None}, "coverage": 0.0,
        "error_count": 15, "error_categories": {"judge_failed": 15},
    },
    {
        "status": "judge_completed", "status_counts": {"judge_completed": 15},
        "scores": {"faithfulness": None, "relevance": None}, "coverage": 1.0,
        "error_count": 0, "error_categories": {},
    },
    {
        "status": "judge_not_run", "status_counts": {"judge_not_run": 15},
        "scores": {"faithfulness": 0.5, "relevance": 0.5}, "coverage": 0.0,
    },
    {
        "status": "judge_completed", "status_counts": {"judge_completed": 15},
        "scores": {"faithfulness": 0.5, "relevance": 0.5}, "coverage": 0.0,
        "error_count": 0, "error_categories": {},
    },
])
def test_contradictory_holdout_judge_summary_fails_with_recomputed_checksums(
    writer, tmp_path, judge_update
):
    bundle = writer.write(_run(tmp_path / "runs", role="holdout"), tmp_path / "reports")
    aggregate = json.loads(bundle.aggregate.read_text(encoding="utf-8"))
    aggregate["judge"].update(judge_update)
    bundle.aggregate.write_text(canonical_json(aggregate) + "\n", encoding="utf-8")
    report = json.loads(bundle.report_json.read_text(encoding="utf-8"))
    report["aggregate"] = aggregate
    bundle.report_json.write_text(canonical_json(report) + "\n", encoding="utf-8")
    _rewrite_checksum(bundle.checksums, bundle.aggregate)
    _rewrite_checksum(bundle.checksums, bundle.report_json)

    result = verify_report_bundle(bundle.root)
    assert result.valid is False
    assert any("judge summary" in error for error in result.errors)


def test_fresh_development_bundle_with_error_case_tuple_verifies(writer, tmp_path):
    run = _run(tmp_path / "runs")
    rows = [json.loads(line) for line in (run.path / "per_query.jsonl").read_text().splitlines()]
    rows[0]["metrics"]["retrieval"]["recall_at_10"] = 0.0
    (run.path / "per_query.jsonl").write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )
    aggregate = dict(run.aggregate)
    aggregate["retrieval"] = dict(aggregate["retrieval"])
    aggregate["retrieval"]["recall_at_10"] = {
        "value": 35 / 36, "scored_count": 36, "total_count": 36,
    }
    (run.path / "aggregate.json").write_text(canonical_json(aggregate) + "\n", encoding="utf-8")
    run = EvaluationRun(path=run.path, manifest=run.manifest, aggregate=aggregate)

    bundle = writer.write(run, tmp_path / "reports")
    analysis = json.loads(bundle.error_analysis.read_text(encoding="utf-8"))
    assert analysis["cases"][0]["evidence"] == []
    assert verify_report_bundle(bundle.root).valid is True


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


@pytest.mark.parametrize(
    ("artifact_name", "mutate", "expected"),
    [
        ("config_manifest.json", lambda value: value["snapshot"].update(index_hash="f" * 64), "config manifest mismatch"),
        ("model_manifest.json", lambda value: value.update(model_hash="f" * 64), "model manifest mismatch"),
        ("environment_manifest.json", lambda value: value.update(backend="different"), "environment manifest mismatch"),
        ("error_analysis.json", lambda value: value.update(dataset_hash="f" * 64), "error analysis mismatch"),
    ],
)
def test_recomputed_checksums_cannot_certify_inconsistent_derived_manifests(bundle, artifact_name, mutate, expected):
    artifact = bundle.root / artifact_name
    value = json.loads(artifact.read_text(encoding="utf-8"))
    mutate(value)
    artifact.write_text(canonical_json(value) + "\n", encoding="utf-8")
    _rewrite_checksum(bundle.checksums, artifact)

    result = verify_report_bundle(bundle.root)
    assert result.valid is False
    assert expected in result.errors


@pytest.mark.parametrize("artifact_name", ["report.json", "report.md", "model_manifest.json"])
def test_malformed_utf8_artifact_fails_verifier_and_cli(bundle, artifact_name, capsys):
    artifact = bundle.root / artifact_name
    artifact.write_bytes(b"\xff\xfe")
    _rewrite_checksum(bundle.checksums, artifact)

    result = verify_report_bundle(bundle.root)
    assert result.valid is False
    assert verify_main([str(bundle.root)]) == 1
    assert json.loads(capsys.readouterr().out)["valid"] is False


def test_symlinked_checksum_manifest_is_unsafe(bundle, tmp_path):
    external = tmp_path / "external-checksums.sha256"
    external.write_bytes(bundle.checksums.read_bytes())
    bundle.checksums.unlink()
    bundle.checksums.symlink_to(external)

    result = verify_report_bundle(bundle.root)
    assert result.valid is False
    assert "unsafe bundle entry: checksums.sha256" in result.errors


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
