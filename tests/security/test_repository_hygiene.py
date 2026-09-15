"""Final repository boundaries for the synthetic trade-intelligence product."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

import scripts.verify_repository as repository_audit


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_REPORTS = REPO_ROOT / "data/eval/trade_intel/reports_public"


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def _repository_paths() -> set[str]:
    inventory = repository_audit.repository_inventory(root=REPO_ROOT)
    assert inventory.error is None
    return inventory.paths


def test_repository_has_no_legacy_ecommerce_runtime() -> None:
    forbidden = ["rag_core/engineering", "demo/ecommerce_seed", "engineering_api.py"]
    assert not [path for path in forbidden if (REPO_ROOT / path).exists()]


def test_readme_never_claims_production_or_real_customer_data() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    forbidden = ["生产已部署", "真实客户数据", "线上转化提升"]
    assert not any(term in readme for term in forbidden)


def test_task_10_documents_and_auditor_exist() -> None:
    required = [
        "README.md",
        "docs/architecture.md",
        "docs/data-contract.md",
        "docs/evaluation.md",
        "docs/operations.md",
        "docs/completion-audit.md",
        "scripts/verify_repository.py",
    ]
    assert not [path for path in required if not (REPO_ROOT / path).is_file()]


@pytest.mark.skipif(
    repository_audit.is_packaged_source(),
    reason="host-only ignored private inputs and lock are intentionally absent from image",
)
def test_private_trade_evaluation_inputs_and_lock_are_ignored_and_untracked() -> None:
    private_root = REPO_ROOT / "data/eval/private/trade_intel"
    lock = private_root / "holdout_consumption.json"
    assert lock.is_file()
    assert _git("check-ignore", "-q", str(lock.relative_to(REPO_ROOT))).returncode == 0
    tracked = set(_git("ls-files").stdout.splitlines())
    assert not any(path.startswith("data/eval/private/") for path in tracked)
    assert str(lock.relative_to(REPO_ROOT)) not in tracked


def test_public_holdout_contains_no_per_query_or_private_label_payload() -> None:
    holdout_bundles = []
    for path in PUBLIC_REPORTS.glob("report-*/report.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dataset_role") == "holdout":
            holdout_bundles.append(path.parent)
    assert holdout_bundles
    forbidden_names = {"per_query.jsonl", "holdout_private.jsonl", "references_private.jsonl"}
    forbidden_keys = {
        "question",
        "reference_claim",
        "reference_evidence",
        "decision_label",
        "decision_text",
    }
    for bundle in holdout_bundles:
        assert not (forbidden_names & {path.name for path in bundle.iterdir()})
        for artifact in bundle.glob("*.json"):
            assert not (forbidden_keys & set(json.loads(artifact.read_text(encoding="utf-8"))))


def test_superpowers_state_is_not_tracked() -> None:
    assert not [
        path for path in _repository_paths()
        if path == ".superpowers" or path.startswith(".superpowers/")
    ]


def test_machine_readable_repository_audit_passes() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.verify_repository"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["errors"] == []


def test_completion_matrix_has_every_stable_row_with_concrete_evidence() -> None:
    matrix = (REPO_ROOT / "docs/completion-audit.md").read_text(encoding="utf-8")
    tracked = _repository_paths()
    assert repository_audit.validate_completion_matrix(matrix, tracked, root=REPO_ROOT) == ()


def test_completion_matrix_validator_rejects_missing_and_indirect_rows() -> None:
    matrix = (REPO_ROOT / "docs/completion-audit.md").read_text(encoding="utf-8")
    tracked = _repository_paths()
    without_first = "\n".join(
        line for line in matrix.splitlines() if not line.startswith("| REQ-001 |")
    )
    assert any("REQ-001" in error for error in repository_audit.validate_completion_matrix(
        without_first, tracked, root=REPO_ROOT
    ))

    indirect = "\n".join(
        line.replace("`trade_agent/data/demo_generator.py`", "synthetic corpus generator")
        if line.startswith("| REQ-001 |") else line
        for line in matrix.splitlines()
    )
    assert any("REQ-001" in error for error in repository_audit.validate_completion_matrix(
        indirect, tracked, root=REPO_ROOT
    ))

    nonexistent = "\n".join(
        line.replace(
            "`demo/trade_intel_seed/manifests/corpus_manifest.json`",
            "`demo/trade_intel_seed/manifests/corpus_manifest.json`; `docs/not-evidence.md`",
            1,
        )
        if line.startswith("| REQ-001 |") else line
        for line in matrix.splitlines()
    )
    assert any("does not exist" in error for error in repository_audit.validate_completion_matrix(
        nonexistent, tracked, root=REPO_ROOT
    ))


def test_tracked_text_scanner_rejects_host_paths_and_secret_shapes() -> None:
    findings = repository_audit.scan_tracked_texts((
        ("docs/unsafe.md", b"cache=/" + b"Users/alice/private/model"),
        ("config/unsafe.txt", b"AWS=AK" + b"IA1234567890ABCDEF"),
    ))
    assert "docs/unsafe.md:1:host_absolute_path" in findings
    assert "config/unsafe.txt:1:secret_shaped_value" in findings


def test_tracked_text_scanner_allowlists_only_exact_reviewed_test_sentinels() -> None:
    allowed = repository_audit.scan_tracked_texts((
        ("tests/unit/test_evidence_models.py", b'        "/' + b'Users/private",'),
    ))
    changed = repository_audit.scan_tracked_texts((
        ("tests/unit/test_evidence_models.py", b'        "/' + b'Users/reviewer/private",'),
    ))
    assert allowed == ()
    assert changed == ("tests/unit/test_evidence_models.py:1:host_absolute_path",)


def test_packaged_inventory_does_not_require_git_and_excludes_runtime_files(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("TRADE_AGENT_PACKAGED_SOURCE", "1")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/evidence.md").write_text("evidence", encoding="utf-8")
    (tmp_path / "data/indexes").mkdir(parents=True)
    (tmp_path / "data/indexes/runtime.json").write_text("runtime", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/__pycache__").mkdir()
    (tmp_path / "scripts/check.py").write_text("pass", encoding="utf-8")
    (tmp_path / "scripts/__pycache__/check.pyc").write_bytes(b"cache")

    inventory = repository_audit.repository_inventory(root=tmp_path)

    assert inventory.source == "packaged"
    assert inventory.error is None
    assert inventory.paths == {"docs/evidence.md", "scripts/check.py"}


def test_packaged_repository_audit_uses_source_inventory_without_private_files(
    monkeypatch,
) -> None:
    host_inventory = repository_audit.repository_inventory(root=REPO_ROOT)
    assert host_inventory.error is None
    monkeypatch.setenv("TRADE_AGENT_PACKAGED_SOURCE", "1")
    packaged = repository_audit.RepositoryInventory(
        source="packaged", paths=host_inventory.paths
    )
    monkeypatch.setattr(repository_audit, "repository_inventory", lambda: packaged)
    monkeypatch.setattr(
        repository_audit,
        "_git",
        lambda *args: pytest.fail(f"packaged audit invoked Git: {args}"),
    )

    checks = repository_audit.audit_repository()

    assert all(check.passed for check in checks), [
        check.error for check in checks if not check.passed
    ]
    private_check = next(
        check for check in checks if check.check_id == "private-holdout-and-lock-ignored"
    )
    assert "packaged source excludes ignored private inputs and lock" in private_check.evidence
