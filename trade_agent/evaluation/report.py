"""Append-only, locally verifiable evidence bundles for trade evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any, Callable, Mapping
from uuid import uuid4

from trade_agent.data.manifest import canonical_json
from trade_agent.evaluation.error_analysis import ErrorAnalyzer
from trade_agent.evaluation.models import RunManifest
from trade_agent.evaluation.runner import EvaluationRun


_CHECKSUM_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")
_SAFE_PART = re.compile(r"[^A-Za-z0-9._-]+")
_ROLES = {"development", "holdout"}
_COMMON_FILES = {
    "manifest.json",
    "aggregate.json",
    "environment_manifest.json",
    "model_manifest.json",
    "config_manifest.json",
    "error_analysis.json",
    "report.json",
    "report.md",
}
_HOLDOUT_MARKDOWN_FORBIDDEN = re.compile(
    r"(?i)\b(?:question|reference[ _-]?(?:claim|evidence)|decision[ _-]?(?:label|text)|"
    r"source[ _-]?excerpt|per[ _-]?query|case[ _-]?id|claim[ _-]?id)\b"
)
_PRIVATE_AGGREGATE_FIELDS = {
    "question",
    "questions",
    "case_id",
    "claim_id",
    "claim_ids",
    "claim_text",
    "reference_claim",
    "reference_claims",
    "reference_evidence",
    "reference_evidence_ids",
    "decision_label",
    "decision_text",
    "source_excerpt",
    "source_excerpts",
    "per_query",
    "rows",
}


@dataclass(frozen=True)
class ReportBundle:
    root: Path
    manifest: Path
    aggregate: Path
    per_query: Path | None
    environment_manifest: Path
    model_manifest: Path
    config_manifest: Path
    report_json: Path
    report_markdown: Path
    checksums: Path
    error_analysis: Path


@dataclass(frozen=True)
class VerificationResult:
    path: Path
    valid: bool
    errors: tuple[str, ...]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_part(value: object, *, field: str) -> str:
    result = _SAFE_PART.sub("_", str(value)).strip("._-")
    if not result:
        raise ValueError(f"{field} cannot form a safe bundle identity")
    return result


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _write_json_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(value) + "\n")


def _safe_artifact(metadata: Mapping[str, Any], name: str, default: str) -> Path:
    value = Path(str(metadata.get("artifacts", {}).get(name, default)))
    if value.is_absolute() or len(value.parts) != 1 or ".." in value.parts:
        raise ValueError(f"{name} artifact must be a file directly inside the run directory")
    return value


def _contains_private_fields(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold().replace("-", "_").replace(" ", "_")
            if normalized in _PRIVATE_AGGREGATE_FIELDS or _contains_private_fields(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_private_fields(item) for item in value)
    return False


class ReportWriter:
    """Publish a frozen run once beneath a report root."""

    def __init__(
        self,
        *,
        now: Callable[[], str] = _utc_stamp,
        nonce: Callable[[], str] = lambda: uuid4().hex[:8],
    ) -> None:
        self._now = now
        self._nonce = nonce

    def write(self, run: EvaluationRun, root: Path) -> ReportBundle:
        root = Path(root)
        metadata = _read_json(run.path / "manifest.json")
        try:
            persisted = RunManifest.model_validate(metadata["run"])
            role = str(metadata["dataset_role"])
            profile = str(metadata["arm"])
            build_id = str(metadata["adapter"]["build_id"])
            case_count = int(metadata["case_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid Task 5 run manifest") from exc
        if persisted != run.manifest:
            raise ValueError("EvaluationRun does not match its persisted manifest")
        if role not in _ROLES:
            raise ValueError("dataset_role must be development or holdout")
        minimum = 36 if role == "development" else 15
        if case_count < minimum:
            raise ValueError(f"{role} report requires at least {minimum} cases")

        aggregate_artifact = _safe_artifact(metadata, "aggregate", "aggregate.json")
        per_query_artifact = _safe_artifact(metadata, "per_query", "per_query.jsonl")
        aggregate = _read_json(run.path / aggregate_artifact)
        if dict(aggregate) != dict(run.aggregate):
            raise ValueError("EvaluationRun aggregate does not match its persisted artifact")
        if int(aggregate.get("case_count", -1)) != case_count:
            raise ValueError("aggregate case_count does not match run manifest")
        if role == "holdout" and _contains_private_fields(aggregate):
            raise ValueError("holdout aggregate contains private fields")
        if not (run.path / per_query_artifact).is_file():
            raise ValueError("run per-query artifact is missing")

        created_at = _safe_part(self._now(), field="UTC timestamp")
        nonce = _safe_part(self._nonce(), field="nonce")
        code_sha = persisted.snapshot.code_hash
        bundle_id = "-".join(
            (
                "report",
                code_sha,
                _safe_part(build_id, field="build ID"),
                _safe_part(profile, field="profile"),
                created_at,
                nonce,
            )
        )
        root.mkdir(parents=True, exist_ok=True)
        self._refuse_prior_publication(root, persisted.run_id)
        destination = root / bundle_id
        if destination.exists():
            raise FileExistsError(destination)
        temporary = root / f".{bundle_id}.tmp-{uuid4().hex}"
        temporary.mkdir(exist_ok=False)
        try:
            shutil.copyfile(run.path / "manifest.json", temporary / "manifest.json")
            shutil.copyfile(run.path / aggregate_artifact, temporary / "aggregate.json")
            if role == "development":
                shutil.copyfile(run.path / per_query_artifact, temporary / "per_query.jsonl")

            adapter = metadata.get("adapter", {})
            environment_manifest = {
                "schema_version": "trade-report-environment/v1",
                "backend": adapter.get("backend"),
                "runtime": adapter.get("runtime", {}),
                "backend_statuses": persisted.backend_statuses,
            }
            model_manifest = {
                "schema_version": "trade-report-model/v1",
                "model_id": adapter.get("model_id"),
                "model_hash": persisted.snapshot.model_hash,
                "prompt_id": adapter.get("prompt_id"),
                "prompt_hash": persisted.snapshot.prompt_hash,
            }
            config_manifest = {
                "schema_version": "trade-report-config/v1",
                "build_id": build_id,
                "profile": metadata.get("profile", {"arm": profile}),
                "budget": metadata.get("budget", {}),
                "snapshot": persisted.snapshot.model_dump(mode="json"),
                "evaluator_id": metadata.get("evaluator_id"),
            }
            if role == "development":
                error_analysis: Mapping[str, Any] = {
                    "status": "available",
                    **ErrorAnalyzer().analyze((run,)).to_dict(),
                }
            else:
                error_analysis = {
                    "schema_version": "trade-error-analysis-placeholder/v1",
                    "dataset_role": "holdout",
                    "status": "not_applicable",
                    "reason": "private_holdout_is_never_used_for_development_error_analysis",
                    "analysis": None,
                }
            report = {
                "schema_version": "trade-evaluation-report/v1",
                "bundle_id": bundle_id,
                "source_run_id": persisted.run_id,
                "dataset_role": role,
                "created_at": created_at,
                "code_sha": code_sha,
                "build_id": build_id,
                "profile": profile,
                "aggregate": aggregate,
                "error_analysis_status": error_analysis["status"],
            }
            for name, value in (
                ("environment_manifest.json", environment_manifest),
                ("model_manifest.json", model_manifest),
                ("config_manifest.json", config_manifest),
                ("error_analysis.json", error_analysis),
                ("report.json", report),
            ):
                _write_json_new(temporary / name, value)
            (temporary / "report.md").write_text(self._markdown(report), encoding="utf-8")
            checksum_lines = [
                f"{sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
                for path in sorted(temporary.iterdir(), key=lambda item: item.name)
            ]
            (temporary / "checksums.sha256").write_text("".join(checksum_lines), encoding="utf-8")
            temporary.rename(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return ReportBundle(
            root=destination,
            manifest=destination / "manifest.json",
            aggregate=destination / "aggregate.json",
            per_query=destination / "per_query.jsonl" if role == "development" else None,
            environment_manifest=destination / "environment_manifest.json",
            model_manifest=destination / "model_manifest.json",
            config_manifest=destination / "config_manifest.json",
            report_json=destination / "report.json",
            report_markdown=destination / "report.md",
            checksums=destination / "checksums.sha256",
            error_analysis=destination / "error_analysis.json",
        )

    @staticmethod
    def _refuse_prior_publication(root: Path, run_id: str) -> None:
        for report_path in root.glob("*/report.json"):
            try:
                report = _read_json(report_path)
            except ValueError:
                continue
            if report.get("source_run_id") == run_id:
                raise FileExistsError(f"run {run_id} is already published at {report_path.parent}")

    @staticmethod
    def _markdown(report: Mapping[str, Any]) -> str:
        aggregate = canonical_json(report["aggregate"])
        return (
            "# Trade evaluation report\n\n"
            f"- Bundle: `{report['bundle_id']}`\n"
            f"- Dataset role: `{report['dataset_role']}`\n"
            f"- Source run: `{report['source_run_id']}`\n"
            f"- Build: `{report['build_id']}`\n"
            f"- Profile: `{report['profile']}`\n"
            f"- Code SHA-256: `{report['code_sha']}`\n"
            f"- Created (UTC): `{report['created_at']}`\n"
            f"- Error analysis: `{report['error_analysis_status']}`\n\n"
            "## Aggregate results\n\n"
            f"```json\n{aggregate}\n```\n"
        )


def verify_report_bundle(path: Path) -> VerificationResult:
    """Verify structure, checksums, contracts, identity, and holdout redaction."""

    path = Path(path)
    errors: list[str] = []
    if not path.is_dir() or path.is_symlink():
        return VerificationResult(path, False, ("bundle path is not a safe directory",))
    checksum_path = path / "checksums.sha256"
    entries: dict[str, str] = {}
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except UnicodeError:
        lines = []
        errors.append("invalid checksum encoding")
    except OSError:
        lines = []
        errors.append("missing file: checksums.sha256")
    for line_number, line in enumerate(lines, 1):
        match = _CHECKSUM_LINE.fullmatch(line)
        if not match:
            errors.append(f"malformed checksum line: {line_number}")
            continue
        digest, name = match.groups()
        item = PurePosixPath(name)
        if item.is_absolute() or len(item.parts) != 1 or name in {".", ".."} or "\\" in name:
            errors.append(f"unsafe checksum path: {name}")
            continue
        if name in entries:
            errors.append(f"duplicate checksum path: {name}")
            continue
        entries[name] = digest

    actual_names: set[str] = set()
    for item in path.iterdir():
        if item.name == "checksums.sha256":
            continue
        if item.is_symlink() or not item.is_file():
            errors.append(f"unsafe bundle entry: {item.name}")
            continue
        actual_names.add(item.name)
    for name in sorted(set(entries) - actual_names):
        errors.append(f"missing file: {name}")
    for name in sorted(actual_names - set(entries)):
        errors.append(f"unexpected file: {name}")
    for name in sorted(actual_names & set(entries)):
        if sha256((path / name).read_bytes()).hexdigest() != entries[name]:
            errors.append(f"checksum mismatch: {name}")

    manifest: Mapping[str, Any] | None = None
    report: Mapping[str, Any] | None = None
    try:
        manifest = _read_json(path / "manifest.json")
        report = _read_json(path / "report.json")
        role = str(manifest["dataset_role"])
        persisted = RunManifest.model_validate(manifest["run"])
        if role not in _ROLES:
            raise ValueError("invalid dataset role")
        expected = _COMMON_FILES | ({"per_query.jsonl"} if role == "development" else set())
        for name in sorted(expected - actual_names):
            message = f"missing file: {name}"
            if message not in errors:
                errors.append(message)
        for name in sorted(actual_names - expected):
            message = f"unexpected file: {name}"
            if message not in errors:
                errors.append(message)
        required_report_keys = {
            "schema_version", "bundle_id", "source_run_id", "dataset_role", "created_at",
            "code_sha", "build_id", "profile", "aggregate", "error_analysis_status",
        }
        if set(report) != required_report_keys:
            errors.append("report.json has an invalid public schema")
        if report.get("bundle_id") != path.name:
            errors.append("bundle directory does not match report identity")
        if report.get("source_run_id") != persisted.run_id:
            errors.append("source run identity mismatch")
        if report.get("dataset_role") != role:
            errors.append("dataset role mismatch")
        if report.get("code_sha") != persisted.snapshot.code_hash:
            errors.append("code SHA mismatch")
        aggregate = _read_json(path / "aggregate.json")
        if report.get("aggregate") != aggregate:
            errors.append("report aggregate does not match aggregate artifact")
        if role == "holdout":
            markdown = (path / "report.md").read_text(encoding="utf-8")
            if _contains_private_fields(report.get("aggregate")):
                errors.append("holdout redaction failure: report.json")
            if _HOLDOUT_MARKDOWN_FORBIDDEN.search(markdown):
                errors.append("holdout redaction failure: report.md")
            if "per_query.jsonl" in actual_names:
                errors.append("holdout redaction failure: per_query.jsonl")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        errors.append(f"invalid bundle metadata: {exc}")
    return VerificationResult(path, not errors, tuple(dict.fromkeys(errors)))
