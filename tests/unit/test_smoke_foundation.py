from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def test_summary_validation_is_fail_closed_for_zero_chunks() -> None:
    from scripts.smoke_foundation import validate_foundation_summary

    summary = {
        "status": "ok",
        "synthetic_only": True,
        "mysql_tables": 7,
        "trade_records": 825,
        "months": 18,
        "corpus": {"sources": 74, "documents": 116, "chunks": 0},
        "metadata_required_completeness": 1.0,
        "quarantine_unexpected": 0,
    }
    with pytest.raises(ValueError, match="chunks"):
        validate_foundation_summary(summary)


def test_summary_validation_accepts_complete_expected_contract() -> None:
    from scripts.smoke_foundation import validate_foundation_summary

    summary = {
        "status": "ok",
        "synthetic_only": True,
        "mysql_tables": 7,
        "trade_records": 825,
        "months": 18,
        "corpus": {"sources": 74, "documents": 116, "chunks": 120},
        "metadata_required_completeness": 1.0,
        "quarantine_unexpected": 0,
    }
    assert validate_foundation_summary(summary) is summary


def test_failure_payload_contains_only_stage_and_exception_class() -> None:
    from scripts.smoke_foundation import failure_payload

    payload = failure_payload("mysql", RuntimeError("password=top-secret /Users/private/app"))
    assert payload == {"error": "RuntimeError", "stage": "mysql", "status": "error"}
    assert "top-secret" not in str(payload)
    assert "/Users" not in str(payload)


def test_summary_validator_rejects_missing_quarantine_field() -> None:
    from scripts.smoke_foundation import validate_foundation_summary

    with pytest.raises(ValueError, match="quarantine_unexpected"):
        validate_foundation_summary({"status": "ok"})


@pytest.mark.parametrize("counts", [(1, 116, 120), (74, 119, 120), (74, 116, 119), (74, 116, 121), (73, 116, 120)])
def test_summary_validator_requires_exact_corpus_counts(counts: tuple[int, int, int]) -> None:
    from scripts.smoke_foundation import validate_foundation_summary

    summary = {
        "status": "ok", "synthetic_only": True, "mysql_tables": 7,
        "trade_records": 825, "months": 18,
        "corpus": {"sources": counts[0], "documents": counts[1], "chunks": counts[2]},
        "metadata_required_completeness": 1.0, "quarantine_unexpected": 0,
    }
    with pytest.raises(ValueError, match="corpus"):
        validate_foundation_summary(summary)


@pytest.mark.parametrize("unsafe_value", ["/Users/private/secret.txt", r"C:\\private\\secret.txt", r"\\\\server\\share\\secret.txt"])
def test_cli_unknown_argument_is_redacted_without_usage_or_echo(unsafe_value: str) -> None:
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "scripts.smoke_foundation", f"--unknown={unsafe_value}"],
        cwd=repo, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert json.loads(result.stderr) == {"error": "ArgumentError", "stage": "arguments", "status": "error"}
    assert unsafe_value not in result.stderr
    assert "usage:" not in result.stderr


def test_corpus_resource_root_is_independent_of_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scripts.smoke_foundation import _repo_root, _resource_path

    monkeypatch.chdir(tmp_path)
    repo = Path(__file__).resolve().parents[2]
    assert _repo_root() == repo
    assert _resource_path("data/sources/trade_intel_demo.yaml") == repo / "data/sources/trade_intel_demo.yaml"


def test_corpus_temp_directory_is_removed_after_success_and_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import scripts.smoke_foundation as smoke

    original = tempfile.TemporaryDirectory

    def factory(*, prefix: str):
        return original(prefix=prefix, dir=tmp_path)

    monkeypatch.setattr(smoke.tempfile, "TemporaryDirectory", factory)
    monkeypatch.setattr(smoke, "_run_corpus_and_ingestion", lambda root: {"ok": root.exists()})
    assert smoke._run_corpus_in_temp() == {"ok": True}
    assert list(tmp_path.glob("trade-foundation-smoke-*")) == []

    def fail(_root: Path):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(smoke, "_run_corpus_and_ingestion", fail)
    with pytest.raises(RuntimeError):
        smoke._run_corpus_in_temp()
    assert list(tmp_path.glob("trade-foundation-smoke-*")) == []
