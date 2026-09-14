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
LEGACY_RUNTIME = ("rag_core/engineering", "demo/ecommerce_seed", "engineering_api.py")
PRIVATE_REQUIRED = (
    "data/eval/private/trade_intel/holdout_private.jsonl",
    "data/eval/private/trade_intel/references_private.jsonl",
    "data/eval/private/trade_intel/holdout_preflight.json",
    "data/eval/private/trade_intel/holdout_consumption.json",
)
README_PROHIBITED = ("生产已部署", "真实客户数据", "线上转化提升")
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


@dataclass(frozen=True)
class MatrixExpectation:
    status: str
    code_paths: tuple[str, ...]
    test_paths: tuple[str, ...]
    artifact_paths: tuple[str, ...]


_HOLDOUT_REPORT = (
    f"data/eval/trade_intel/reports_public/{HOLDOUT_BUNDLE_NAME}/report.json"
)
_CYCLE_REPORT = (
    "data/eval/trade_intel/reports_public/"
    "cycle-d2f1d18c65a549e2a191a989fd734236/paired.json"
)
_CORPUS_MANIFEST = "demo/trade_intel_seed/manifests/corpus_manifest.json"
_STATUSES = {
    "repository_evidence_present",
    "committed_synthetic_evidence",
    "controller_final_live_gate",
}


def _matrix(
    status: str,
    code: tuple[str, ...],
    tests: tuple[str, ...],
    artifacts: tuple[str, ...],
) -> MatrixExpectation:
    return MatrixExpectation(status, code, tests, artifacts)


MATRIX_EXPECTATIONS = {
    "REQ-001": _matrix("repository_evidence_present", ("trade_agent/data/demo_generator.py",), ("tests/unit/test_demo_corpus.py",), (_CORPUS_MANIFEST,)),
    "REQ-002": _matrix("controller_final_live_gate", ("docker-compose.yml",), ("tests/contract/test_compose_contract.py",), ("scripts/smoke_foundation.py",)),
    "REQ-003": _matrix("controller_final_live_gate", ("db/migrations/001_schema.sql", "db/init/010_users.sh"), ("tests/integration/test_mysql_schema.py",), ("scripts/smoke_foundation.py",)),
    "REQ-004": _matrix("controller_final_live_gate", ("trade_agent/db/sql_validator.py", "trade_agent/db/sql_executor.py"), ("tests/integration/test_sql_execution.py",), ("trade_agent/config/schema_registry.yaml",)),
    "REQ-005": _matrix("repository_evidence_present", ("trade_agent/data/router.py", "trade_agent/data/pdf.py"), ("tests/integration/test_corpus_routing.py", "tests/integration/test_pdf_pipeline.py"), (_CORPUS_MANIFEST,)),
    "REQ-006": _matrix("repository_evidence_present", ("trade_agent/schemas/source.py", "trade_agent/data/chunkers.py"), ("tests/unit/test_source_schemas.py", "tests/unit/test_chunkers.py"), (_CORPUS_MANIFEST,)),
    "REQ-007": _matrix("controller_final_live_gate", ("trade_agent/index/embeddings.py", "trade_agent/index/bge_m3_artifact_manifest.json"), ("tests/model/test_real_bge_m3.py",), ("scripts/smoke_embeddings.py",)),
    "REQ-008": _matrix("controller_final_live_gate", ("trade_agent/index/milvus_store.py", "trade_agent/index/builder.py"), ("tests/milvus/test_milvus_roundtrip.py", "tests/milvus/test_full_index.py"), ("scripts/smoke_milvus_roundtrip.py",)),
    "REQ-009": _matrix("controller_final_live_gate", ("trade_agent/retrieval/service.py", "trade_agent/retrieval/reranker.py"), ("tests/integration/test_retrieval_service.py", "tests/model/test_real_bge_reranker.py"), (_CYCLE_REPORT, "scripts/smoke_milvus_roundtrip.py")),
    "REQ-010": _matrix("repository_evidence_present", ("trade_agent/entities/resolver.py", "trade_agent/entities/dedup.py"), ("tests/unit/test_entity_resolution.py", "tests/unit/test_deduplication.py"), (_CORPUS_MANIFEST,)),
    "REQ-011": _matrix("repository_evidence_present", ("trade_agent/entities/conflicts.py",), ("tests/unit/test_conflicts.py",), ("tests/integration/test_graph.py",)),
    "REQ-012": _matrix("repository_evidence_present", ("trade_agent/evidence/validator.py",), ("tests/unit/test_evidence_validator.py",), ("tests/integration/test_graph.py",)),
    "REQ-013": _matrix("repository_evidence_present", ("trade_agent/evidence/claim_guard.py",), ("tests/unit/test_claim_guard.py",), ("tests/integration/test_graph.py",)),
    "REQ-014": _matrix("controller_final_live_gate", ("trade_agent/agents/graph.py", "trade_agent/agents/nodes.py"), ("tests/integration/test_graph.py", "tests/e2e/test_query_workflow.py"), ("tests/e2e/test_query_workflow.py",)),
    "REQ-015": _matrix("controller_final_live_gate", ("trade_agent/agents/checkpoint.py",), ("tests/integration/test_redis_checkpoint.py",), ("scripts/smoke_foundation.py",)),
    "REQ-016": _matrix("repository_evidence_present", ("trade_agent/api/app.py", "trade_agent/cli.py"), ("tests/contract/test_api.py",), ("scripts/verify_trade_report.py",)),
    "REQ-017": _matrix("repository_evidence_present", ("trade_agent/errors.py", "trade_agent/config/settings.py"), ("tests/unit/test_settings.py", "tests/integration/test_graph.py"), ("docs/operations.md",)),
    "REQ-018": _matrix("repository_evidence_present", ("trade_agent/evaluation/generator.py",), ("tests/unit/test_eval_generator.py",), ("data/eval/trade_intel/dev_public.jsonl",)),
    "REQ-019": _matrix("committed_synthetic_evidence", (".gitignore", "trade_agent/evaluation/holdout.py"), ("tests/unit/test_holdout_consumption.py",), (_HOLDOUT_REPORT, "data/eval/private/trade_intel/holdout_private.jsonl", "data/eval/private/trade_intel/references_private.jsonl", "data/eval/private/trade_intel/holdout_consumption.json")),
    "REQ-020": _matrix("repository_evidence_present", ("trade_agent/evaluation/leakage.py", "trade_agent/evaluation/runner.py"), ("tests/unit/test_eval_leakage.py",), (_CORPUS_MANIFEST,)),
    "REQ-021": _matrix("committed_synthetic_evidence", ("trade_agent/evaluation/holdout.py",), ("tests/unit/test_holdout_consumption.py",), ("data/eval/trade_intel/holdout_snapshot.json", "data/eval/private/trade_intel/holdout_preflight.json", "data/eval/private/trade_intel/holdout_consumption.json")),
    "REQ-022": _matrix("committed_synthetic_evidence", ("trade_agent/evaluation/report.py", "trade_agent/evaluation/judge.py"), ("tests/contract/test_report_bundle.py", "tests/unit/test_judge.py"), (_HOLDOUT_REPORT,)),
    "REQ-023": _matrix("committed_synthetic_evidence", ("trade_agent/evaluation/optimizer.py", "trade_agent/evaluation/cycle.py"), ("tests/unit/test_trade_eval_cycle.py",), (_CYCLE_REPORT,)),
    "REQ-024": _matrix("committed_synthetic_evidence", ("trade_agent/evaluation/holdout.py", "trade_agent/evaluation/report.py"), ("tests/unit/test_holdout_consumption.py",), (_HOLDOUT_REPORT, "data/eval/trade_intel/holdout_snapshot.json")),
    "REQ-025": _matrix("repository_evidence_present", ("scripts/run_trade_eval.py",), ("tests/security/test_repository_hygiene.py",), (_CYCLE_REPORT, _HOLDOUT_REPORT)),
    "REQ-026": _matrix("repository_evidence_present", ("scripts/verify_repository.py", ".gitignore"), ("tests/security/test_repository_hygiene.py",), ("docs/completion-audit.md",)),
    "ACC-001": _matrix("controller_final_live_gate", ("docker-compose.yml",), ("tests/contract/test_compose_contract.py",), ("scripts/smoke_foundation.py",)),
    "ACC-002": _matrix("controller_final_live_gate", ("db/migrations/001_schema.sql", "db/init/010_users.sh"), ("tests/integration/test_mysql_schema.py",), ("scripts/smoke_foundation.py",)),
    "ACC-003": _matrix("controller_final_live_gate", ("trade_agent/data/router.py", "trade_agent/data/pdf.py"), ("tests/integration/test_corpus_routing.py", "tests/integration/test_pdf_pipeline.py"), ("scripts/smoke_foundation.py",)),
    "ACC-004": _matrix("repository_evidence_present", ("trade_agent/data/chunkers.py", "trade_agent/schemas/source.py"), ("tests/unit/test_chunkers.py", "tests/unit/test_source_schemas.py"), (_CORPUS_MANIFEST,)),
    "ACC-005": _matrix("controller_final_live_gate", ("trade_agent/index/milvus_store.py",), ("tests/milvus/test_milvus_roundtrip.py",), ("scripts/smoke_milvus_roundtrip.py",)),
    "ACC-006": _matrix("controller_final_live_gate", ("trade_agent/retrieval/service.py", "trade_agent/retrieval/reranker.py"), ("tests/integration/test_retrieval_service.py",), (_CYCLE_REPORT, "scripts/smoke_milvus_roundtrip.py")),
    "ACC-007": _matrix("controller_final_live_gate", ("trade_agent/agents/graph.py", "trade_agent/agents/checkpoint.py"), ("tests/integration/test_graph.py", "tests/integration/test_redis_checkpoint.py"), ("tests/e2e/test_query_workflow.py",)),
    "ACC-008": _matrix("repository_evidence_present", ("trade_agent/evidence/validator.py", "trade_agent/evidence/claim_guard.py"), ("tests/unit/test_evidence_validator.py", "tests/unit/test_claim_guard.py"), ("tests/integration/test_graph.py",)),
    "ACC-009": _matrix("committed_synthetic_evidence", ("trade_agent/evaluation/cycle.py", "trade_agent/evaluation/holdout.py"), ("tests/unit/test_trade_eval_cycle.py", "tests/unit/test_holdout_consumption.py"), (_CYCLE_REPORT, _HOLDOUT_REPORT)),
    "ACC-010": _matrix("controller_final_live_gate", ("scripts/verify_repository.py",), ("tests/security/test_repository_hygiene.py",), ("docs/operations.md",)),
    "ACC-011": _matrix("repository_evidence_present", ("README.md",), ("tests/security/test_repository_hygiene.py",), ("docs/evaluation.md",)),
    "ACC-012": _matrix("controller_final_live_gate", ("scripts/verify_repository.py",), ("tests/security/test_repository_hygiene.py",), ("docs/completion-audit.md",)),
}


_BACKTICK_PATH = re.compile(r"`([^`]+)`")
_MATRIX_ROW = re.compile(r"^\| ((?:REQ|ACC)-\d{3}) \|")
_HOST_PATTERNS = (
    re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._?=&%-]+)*"),
    re.compile(r"/private/var/folders/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*"),
    re.compile(r"[A-Za-z]:\\{1,2}Users\\{1,2}[A-Za-z0-9._-]+(?:\\{1,2}[A-Za-z0-9._-]+)*"),
)
_SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?:^|[^A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def _fixture_posix_path(*parts: str) -> str:
    """Build intentional host-path fixtures without embedding one in this file."""
    return "/" + "/".join(parts)


def _fixture_windows_path(*parts: str) -> str:
    """Build the doubled-backslash fixture used by the manifest tests."""
    return "C:" + ("\\" * 2) + ("\\" * 2).join(parts)


_HOST_SENTINEL_ALLOWLIST = {
    "tests/unit/test_evidence_models.py": {
        _fixture_posix_path("Users", "private"),
        _fixture_posix_path("Users", "private", "corpus.json"),
    },
    "tests/unit/test_manifest.py": {
        _fixture_posix_path("Users", "alice", "project", "input.pdf"),
        _fixture_posix_path("home", "alice", "input.pdf"),
        _fixture_posix_path("private", "var", "folders", "aa", "bb", "input.pdf"),
        _fixture_windows_path("Users", "alice", "input.pdf"),
    },
    "tests/unit/test_smoke_embeddings.py": {
        _fixture_posix_path("Users", "private", "model-cache?token=secret"),
    },
    "tests/unit/test_smoke_foundation.py": {
        _fixture_posix_path("Users", "private", "app"),
        _fixture_posix_path("Users", "private", "secret.txt"),
    },
}


def _cell_paths(cell: str) -> set[str]:
    return set(_BACKTICK_PATH.findall(cell))


def validate_completion_matrix(
    text: str, tracked: set[str], *, root: Path = ROOT
) -> tuple[str, ...]:
    """Validate stable requirement rows and exact evidence categories."""
    errors: list[str] = []
    rows: dict[str, list[str]] = {}
    for line in text.splitlines():
        match = _MATRIX_ROW.match(line)
        if not match:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        row_id = match.group(1)
        if row_id in rows:
            errors.append(f"{row_id}: duplicate matrix row")
        elif len(cells) != 6:
            errors.append(f"{row_id}: expected six matrix columns")
        else:
            rows[row_id] = cells
    expected_ids = set(MATRIX_EXPECTATIONS)
    for row_id in sorted(expected_ids - set(rows)):
        errors.append(f"{row_id}: missing matrix row")
    for row_id in sorted(set(rows) - expected_ids):
        errors.append(f"{row_id}: unexpected matrix row")
    for row_id in sorted(expected_ids & set(rows)):
        cells = rows[row_id]
        expected = MATRIX_EXPECTATIONS[row_id]
        status = cells[5].strip("`")
        if status not in _STATUSES or status != expected.status:
            errors.append(f"{row_id}: status must be {expected.status}")
        categories = (
            ("code", _cell_paths(cells[2]), set(expected.code_paths)),
            ("test", _cell_paths(cells[3]), set(expected.test_paths)),
            ("artifact", _cell_paths(cells[4]), set(expected.artifact_paths)),
        )
        for category, actual, required in categories:
            if not actual:
                errors.append(f"{row_id}: {category} evidence must contain exact paths")
                continue
            missing = sorted(required - actual)
            if missing:
                errors.append(f"{row_id}: missing required {category} paths: {', '.join(missing)}")
            for path in sorted(actual):
                candidate = Path(path)
                if candidate.is_absolute() or ".." in candidate.parts:
                    errors.append(f"{row_id}: unsafe evidence path {path}")
                elif not (root / candidate).exists():
                    errors.append(f"{row_id}: evidence path does not exist: {path}")
                elif not path.startswith("data/eval/private/") and path not in tracked:
                    errors.append(f"{row_id}: evidence path is not tracked: {path}")
            if category == "test" and any(not path.startswith("tests/") for path in actual):
                errors.append(f"{row_id}: test evidence must use tests/ paths")
    return tuple(errors)


def scan_tracked_texts(files: Iterable[tuple[str, bytes]]) -> tuple[str, ...]:
    """Scan every supplied tracked blob, allowing only exact reviewed sentinels."""
    findings: set[str] = set()
    for path, content in files:
        text = content.decode("utf-8", errors="replace")
        allowed_hosts = _HOST_SENTINEL_ALLOWLIST.get(path, set())
        for line_number, line in enumerate(text.splitlines(), 1):
            if any(
                match.group(0) not in allowed_hosts
                for pattern in _HOST_PATTERNS
                for match in pattern.finditer(line)
            ):
                findings.add(f"{path}:{line_number}:host_absolute_path")
            if any(pattern.search(line) for pattern in _SECRET_PATTERNS):
                findings.add(f"{path}:{line_number}:secret_shaped_value")
    return tuple(sorted(findings))


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
    matrix_errors = validate_completion_matrix(audit_doc, tracked)
    record(
        "completion-matrix-exact-evidence",
        not matrix_errors,
        "docs/completion-audit.md",
        f"stable_rows={len(MATRIX_EXPECTATIONS)}",
        error="; ".join(matrix_errors) if matrix_errors else None,
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

    blobs: list[tuple[str, bytes]] = []
    unreadable: list[str] = []
    for relative in sorted(tracked):
        try:
            blobs.append((relative, (ROOT / relative).read_bytes()))
        except OSError:
            unreadable.append(relative)
    unsafe = list(scan_tracked_texts(blobs))
    unsafe.extend(f"{path}:unreadable" for path in unreadable)
    record(
        "all-tracked-files-have-no-host-path-or-secret",
        not unsafe,
        f"tracked_files_scanned={len(blobs)}",
        "fixture allowlist: exact reviewed host-path sentinels only",
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
