"""Deterministically audit committed trade-agent evidence and truth boundaries."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable

from trade_agent.data.manifest import canonical_json
from trade_agent.evaluation.cycle import verify_cycle_bundle
from trade_agent.evaluation.holdout import verify_snapshot


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data/eval/trade_intel/reports_public"
CYCLE = REPORTS / "cycle-d2f1d18c65a549e2a191a989fd734236"
HOLDOUT_BUNDLE_NAME = (
    "report-7cd261cf288521093c50d1083afec460ea2bb2c624cbea16e46abdfe8d70daf8-"
    "local-build-a3137d12491fb54f1a058fefa47cc96e-full_rerank-"
    "20260910T214832Z-7bb1aa43"
)
HOLDOUT_BUNDLE = REPORTS / HOLDOUT_BUNDLE_NAME
HOLDOUT_LOCK = ROOT / "data/eval/private/trade_intel/holdout_consumption.json"

REQUIRED_TRACKED = (
    "README.md",
    "docs/architecture.md",
    "docs/data-contract.md",
    "docs/evaluation.md",
    "docs/operations.md",
    "docs/completion-audit.md",
    "scripts/verify_repository.py",
    "tests/security/test_repository_hygiene.py",
    "trade_agent/cli.py",
    "docker-compose.yml",
    ".env.example",
    "db/migrations/001_schema.sql",
    "db/init/010_users.sh",
    "trade_agent/config/schema_registry.yaml",
    "trade_agent/config/retrieval_profiles.yaml",
    "trade_agent/data/pipeline.py",
    "trade_agent/data/chunkers.py",
    "trade_agent/index/embeddings.py",
    "trade_agent/index/milvus_store.py",
    "trade_agent/index/builder.py",
    "trade_agent/retrieval/bm25.py",
    "trade_agent/retrieval/fusion.py",
    "trade_agent/retrieval/reranker.py",
    "trade_agent/retrieval/service.py",
    "trade_agent/db/registry.py",
    "trade_agent/db/sql_validator.py",
    "trade_agent/db/sql_executor.py",
    "trade_agent/evidence/validator.py",
    "trade_agent/evidence/claim_guard.py",
    "trade_agent/agents/graph.py",
    "trade_agent/agents/checkpoint.py",
    "trade_agent/api/app.py",
    "trade_agent/evaluation/holdout.py",
    "trade_agent/evaluation/leakage.py",
    "data/eval/trade_intel/dev_public.jsonl",
    "data/eval/trade_intel/references_dev.jsonl",
    "data/eval/trade_intel/holdout_snapshot.json",
    f"data/eval/trade_intel/reports_public/{HOLDOUT_BUNDLE_NAME}/report.json",
    "data/eval/trade_intel/reports_public/cycle-d2f1d18c65a549e2a191a989fd734236/paired.json",
)
DELIVERABLE_SCAN = (
    "README.md",
    "docs/architecture.md",
    "docs/data-contract.md",
    "docs/evaluation.md",
    "docs/operations.md",
    "docs/completion-audit.md",
    "scripts/verify_repository.py",
    "tests/security/test_repository_hygiene.py",
    "trade_agent/cli.py",
)
LEGACY_RUNTIME = ("rag_core/engineering", "demo/ecommerce_seed", "engineering_api.py")
PRIVATE_REQUIRED = (
    "data/eval/private/trade_intel/holdout_private.jsonl",
    "data/eval/private/trade_intel/references_private.jsonl",
    "data/eval/private/trade_intel/holdout_preflight.json",
    "data/eval/private/trade_intel/holdout_consumption.json",
)
README_PROHIBITED = ("生产已部署", "真实客户数据", "线上转化提升")
HOST_PATH = re.compile(r"(?:/(?:Users|home)/[^/\s]+|[A-Za-z]:\\Users\\)")
SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?:^|[^A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
PUBLIC_HOLDOUT_FORBIDDEN_FILES = {
    "per_query.jsonl",
    "holdout_private.jsonl",
    "references_private.jsonl",
    "holdout_preflight.json",
    "holdout_consumption.json",
}
PUBLIC_HOLDOUT_FORBIDDEN_KEYS = {
    "question",
    "case_id",
    "claim_id",
    "reference_claim",
    "reference_evidence",
    "decision_label",
    "decision_text",
    "source_excerpt",
}


@dataclass(frozen=True)
class Check:
    check_id: str
    passed: bool
    evidence: tuple[str, ...]
    error: str | None = None

    def as_json(self) -> dict[str, object]:
        return {
            "id": self.check_id,
            "status": "passed" if self.passed else "failed",
            "evidence": list(self.evidence),
            "error": self.error,
        }


def _command(arguments: Iterable[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(arguments), cwd=ROOT, check=False, capture_output=True, text=True
    )


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return _command(("git", "-C", str(ROOT), *arguments))


def _nested_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_nested_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_nested_keys(item) for item in value), set())
    return set()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.relative_to(ROOT)} must contain a JSON object")
    return value


def _close(actual: object, expected: float) -> bool:
    return type(actual) in (int, float) and math.isclose(float(actual), expected)


def audit_repository() -> tuple[Check, ...]:
    checks: list[Check] = []

    def record(check_id: str, passed: bool, *evidence: str, error: str | None = None) -> None:
        checks.append(Check(check_id, passed, tuple(evidence), error if not passed else None))

    tracked_result = _git("ls-files")
    tracked = set(tracked_result.stdout.splitlines()) if tracked_result.returncode == 0 else set()
    record(
        "git-index-readable",
        tracked_result.returncode == 0,
        "git ls-files",
        error="unable to read Git index" if tracked_result.returncode else None,
    )

    missing = sorted(path for path in REQUIRED_TRACKED if path not in tracked or not (ROOT / path).is_file())
    record(
        "required-committed-evidence",
        not missing,
        *REQUIRED_TRACKED,
        error=f"missing tracked evidence: {', '.join(missing)}" if missing else None,
    )

    legacy = sorted(path for path in LEGACY_RUNTIME if (ROOT / path).exists())
    record(
        "legacy-ecommerce-runtime-absent",
        not legacy,
        *LEGACY_RUNTIME,
        error=f"legacy runtime paths remain: {', '.join(legacy)}" if legacy else None,
    )

    superpowers = sorted(path for path in tracked if path == ".superpowers" or path.startswith(".superpowers/"))
    record(
        "superpowers-untracked",
        not superpowers,
        "git ls-files .superpowers",
        error=f"tracked orchestration files: {', '.join(superpowers)}" if superpowers else None,
    )

    private_tracked = sorted(path for path in tracked if path.startswith("data/eval/private/"))
    private_missing = sorted(path for path in PRIVATE_REQUIRED if not (ROOT / path).is_file())
    private_unignored = sorted(
        path
        for path in PRIVATE_REQUIRED
        if _git("check-ignore", "-q", path).returncode != 0
    )
    private_ok = not private_tracked and not private_missing and not private_unignored
    private_errors = []
    if private_tracked:
        private_errors.append("tracked private paths: " + ", ".join(private_tracked))
    if private_missing:
        private_errors.append("missing local private control artifacts: " + ", ".join(private_missing))
    if private_unignored:
        private_errors.append("private paths not ignored: " + ", ".join(private_unignored))
    record(
        "private-holdout-and-lock-ignored",
        private_ok,
        *PRIVATE_REQUIRED,
        error="; ".join(private_errors) if private_errors else None,
    )

    try:
        dev_rows = [
            json.loads(line)
            for line in (ROOT / "data/eval/trade_intel/dev_public.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        dev_ok = len(dev_rows) >= 36 and all(
            row.get("dataset_role") == "development" and row.get("visibility") == "public"
            for row in dev_rows
        )
        record(
            "development-dataset-size-and-role",
            dev_ok,
            "data/eval/trade_intel/dev_public.jsonl",
            f"case_count={len(dev_rows)}",
            error=None if dev_ok else "development dataset must contain at least 36 public development cases",
        )
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError) as exc:
        record("development-dataset-size-and-role", False, "data/eval/trade_intel/dev_public.jsonl", error=str(exc))

    cycle_result = verify_cycle_bundle(CYCLE)
    record(
        "public-development-cycle-verifies",
        cycle_result.valid,
        str(CYCLE.relative_to(ROOT)),
        error="; ".join(cycle_result.errors) if cycle_result.errors else None,
    )
    holdout_result = verify_snapshot(HOLDOUT_BUNDLE)
    record(
        "first-public-holdout-verifies",
        holdout_result.valid,
        str(HOLDOUT_BUNDLE.relative_to(ROOT)),
        "data/eval/trade_intel/holdout_snapshot.json",
        error="; ".join(holdout_result.errors) if holdout_result.errors else None,
    )

    try:
        cycle = _json(CYCLE / "paired.json")
        arm = cycle["arms"]["full_rerank"]  # type: ignore[index]
        precision = arm["paired_deltas"]["context_precision"]  # type: ignore[index]
        recall = arm["paired_deltas"]["recall_at_10"]  # type: ignore[index]
        regressions = arm["regressions"]  # type: ignore[index]
        cycle_metrics_ok = (
            _close(precision["baseline_mean"], 0.07523148148148148)
            and _close(precision["candidate_mean"], 0.19490740740740742)
            and _close(recall["baseline_mean"], 0.4722222222222222)
            and _close(recall["candidate_mean"], 0.4722222222222222)
            and regressions["recall_at_10"] == []
            and regressions["context_precision"] == []
        )
        record(
            "development-measured-facts",
            cycle_metrics_ok,
            str((CYCLE / "paired.json").relative_to(ROOT)),
            "full_rerank precision 0.07523148148148148 -> 0.19490740740740742",
            "full_rerank recall 0.4722222222222222 -> 0.4722222222222222",
            error=None if cycle_metrics_ok else "development metrics differ from the published measured facts",
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        record("development-measured-facts", False, str((CYCLE / "paired.json").relative_to(ROOT)), error=str(exc))

    try:
        report = _json(HOLDOUT_BUNDLE / "report.json")
        manifest = _json(HOLDOUT_BUNDLE / "manifest.json")
        aggregate = report["aggregate"]  # type: ignore[index]
        retrieval = aggregate["retrieval"]  # type: ignore[index]
        judge = aggregate["judge"]  # type: ignore[index]
        holdout_metrics_ok = (
            report.get("dataset_role") == "holdout"
            and aggregate["case_count"] == 44  # type: ignore[index]
            and aggregate["status_counts"] == {"completed": 44}  # type: ignore[index]
            and _close(retrieval["recall_at_10"]["value"], 0.5833333333333334)  # type: ignore[index]
            and _close(retrieval["context_precision"]["value"], 0.20601851851851852)  # type: ignore[index]
            and judge["status"] == "judge_not_run"  # type: ignore[index]
            and judge["scores"] == {"faithfulness": None, "relevance": None}  # type: ignore[index]
            and manifest["adapter"]["backend"] == "local-cpu-baseline"  # type: ignore[index]
            and manifest["adapter"]["generation_available"] is False  # type: ignore[index]
            and manifest["holdout_freeze"]["generation_status"] == "not_run"  # type: ignore[index]
        )
        record(
            "holdout-measured-facts-and-qualification",
            holdout_metrics_ok,
            str((HOLDOUT_BUNDLE / "report.json").relative_to(ROOT)),
            "local-cpu-baseline; generation not_run; judge_not_run/null",
            error=None if holdout_metrics_ok else "holdout metrics or local-only qualification differ from published facts",
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        record("holdout-measured-facts-and-qualification", False, str((HOLDOUT_BUNDLE / "report.json").relative_to(ROOT)), error=str(exc))

    public_errors: list[str] = []
    holdout_directories: list[Path] = []
    for report_path in sorted(REPORTS.glob("report-*/report.json")):
        try:
            if _json(report_path).get("dataset_role") == "holdout":
                holdout_directories.append(report_path.parent)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            public_errors.append(f"invalid public report {report_path.parent.name}: {exc}")
    if not holdout_directories:
        public_errors.append("no public holdout aggregate found")
    for directory in holdout_directories:
        exposed_files = PUBLIC_HOLDOUT_FORBIDDEN_FILES & {path.name for path in directory.iterdir()}
        if exposed_files:
            public_errors.append(f"{directory.name} exposes {sorted(exposed_files)}")
        for artifact in directory.glob("*.json"):
            try:
                exposed_keys = PUBLIC_HOLDOUT_FORBIDDEN_KEYS & _nested_keys(_json(artifact))
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                public_errors.append(f"cannot inspect {directory.name}/{artifact.name}: {exc}")
                continue
            if exposed_keys:
                public_errors.append(f"{directory.name}/{artifact.name} exposes {sorted(exposed_keys)}")
    record(
        "public-holdout-is-aggregate-only",
        not public_errors,
        *(str(path.relative_to(ROOT)) for path in holdout_directories),
        error="; ".join(public_errors) if public_errors else None,
    )

    leakage = _command(
        (
            sys.executable,
            "-m",
            "scripts.validate_trade_eval",
            "--dev",
            "data/eval/trade_intel",
            "--holdout",
            "data/eval/private/trade_intel",
            "--require-zero-leakage",
        )
    )
    leakage_ok = leakage.returncode == 0
    if leakage_ok:
        try:
            leakage_ok = json.loads(leakage.stdout).get("passed") is True
        except (json.JSONDecodeError, AttributeError):
            leakage_ok = False
    record(
        "evaluation-leakage-gate",
        leakage_ok,
        "scripts.validate_trade_eval",
        "question/reference/index separation",
        error=None if leakage_ok else "public/private contamination validation failed",
    )

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    prohibited = sorted(term for term in README_PROHIBITED if term in readme)
    qualification_ok = all(
        phrase in readme
        for phrase in (
            "synthetic, non-production",
            "local CPU hash-token cosine",
            "do **not** establish real Milvus/BGE acceptance",
            "Generation was unavailable",
            "judge_not_run",
        )
    )
    record(
        "readme-truth-boundary",
        not prohibited and qualification_ok,
        "README.md",
        error=(
            "README contains prohibited claims or lacks the local synthetic evaluation qualification"
            if prohibited or not qualification_ok
            else None
        ),
    )

    audit_doc = (ROOT / "docs/completion-audit.md").read_text(encoding="utf-8")
    live_terms = (
        "Controller-final live gate",
        "Milvus insert/search/filter/reconnect/cleanup",
        "Redis recovery",
        "no Task 10 Docker receipt",
    )
    record(
        "completion-audit-separates-live-gates",
        all(term in audit_doc for term in live_terms),
        "docs/completion-audit.md",
        error="completion audit does not enumerate controller-final live gates",
    )

    help_result = _command((sys.executable, "-m", "trade_agent.cli", "--help"))
    commands = (
        "bootstrap-demo",
        "db",
        "ingest",
        "index",
        "query",
        "eval",
        "smoke",
        "verify-report",
    )
    eval_result = _command(
        (
            sys.executable,
            "-m",
            "trade_agent.cli",
            "eval",
            "--dataset",
            "unused.jsonl",
            "--output",
            "unused-output",
            "--max-cases",
            "0",
        )
    )
    verify_result = _command(
        (
            sys.executable,
            "-m",
            "trade_agent.cli",
            "verify-report",
            "--latest",
            "data/eval/trade_intel/reports_public",
            "--kind",
            "holdout",
        )
    )
    cli_ok = (
        help_result.returncode == 0
        and all(command in help_result.stdout for command in commands)
        and eval_result.returncode == 2
        and "max-cases must be positive" in eval_result.stderr
        and verify_result.returncode == 0
        and '"valid": true' in verify_result.stdout
    )
    record(
        "cli-entry-points-and-delegation",
        cli_ok,
        "python -m trade_agent.cli --help",
        "trade-intel eval -> scripts.run_trade_eval",
        "trade-intel verify-report -> scripts.verify_trade_report",
        error="CLI surface or evaluation/report delegation is unavailable" if not cli_ok else None,
    )

    unsafe: list[str] = []
    for relative in DELIVERABLE_SCAN:
        if relative not in tracked:
            continue
        try:
            text = (ROOT / relative).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            unsafe.append(f"{relative}: unreadable ({exc})")
            continue
        if HOST_PATH.search(text):
            unsafe.append(f"{relative}: host absolute path")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            unsafe.append(f"{relative}: secret-shaped value")
    record(
        "tracked-deliverables-have-no-host-path-or-secret",
        not unsafe,
        *DELIVERABLE_SCAN,
        error="; ".join(unsafe) if unsafe else None,
    )

    return tuple(checks)


def main() -> int:
    try:
        checks = audit_repository()
    except Exception as exc:  # fail closed at the audit boundary
        payload = {
            "schema_version": "trade-repository-audit/v1",
            "valid": False,
            "checks": [],
            "errors": [f"audit_exception:{type(exc).__name__}"],
        }
        print(canonical_json(payload))
        return 1
    errors = [check.error for check in checks if not check.passed and check.error]
    payload = {
        "schema_version": "trade-repository-audit/v1",
        "valid": not errors,
        "checks": [check.as_json() for check in checks],
        "errors": errors,
    }
    print(canonical_json(payload))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
