"""Final repository boundaries for the synthetic trade-intelligence product."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_REPORTS = REPO_ROOT / "data/eval/trade_intel/reports_public"


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


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
        path for path in _git("ls-files").stdout.splitlines()
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
