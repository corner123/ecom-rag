"""Append-only, locally verifiable evidence bundles for trade evaluation runs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
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
_RESULT_STATUSES = {"completed", "refused", "failed"}
_JUDGE_STATUSES = {"judge_not_run", "judge_completed", "judge_failed"}
_RETRIEVAL_METRICS = ("recall_at_10", "context_precision", "context_recall", "reciprocal_rank")
_JUDGE_SCORES = ("faithfulness", "relevance")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_DEGRADATION_STATUS = re.compile(
    r"^[A-Za-z0-9._-]+:(?:degraded|unavailable|not_run|failed)$"
)
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
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
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


def _read_rows(path: Path, run_id: str, case_count: int) -> tuple[Mapping[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read per-query artifact {path}: {exc}") from exc
    rows: list[Mapping[str, Any]] = []
    case_ids: set[str] = set()
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
            result = row["result"]
            case_id = str(result["case_id"])
            if not isinstance(row, Mapping) or result.get("run_id") != run_id:
                raise ValueError("row identity mismatch")
            if case_id in case_ids:
                raise ValueError("duplicate case ID")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid per-query row {line_number}: {exc}") from exc
        case_ids.add(case_id)
        rows.append(row)
    if len(rows) != case_count:
        raise ValueError("per-query row count does not match run manifest")
    return tuple(rows)


def _finite_number(value: object, *, field: str, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _consistent(values: list[object], *, field: str) -> object:
    if not values or any(value != values[0] for value in values[1:]):
        raise ValueError(f"per-query judge {field} must be consistent")
    return values[0]


def _validate_judge_summary(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("judge summary must be an object")
    expected_keys = {
        "status", "status_counts", "scores", "errors", "coverage", "prompt_hash",
        "model_hash", "provider", "model", "temperature",
    }
    if set(value) != expected_keys:
        raise ValueError("judge summary has an invalid schema")
    status = value["status"]
    if status not in _JUDGE_STATUSES | {"mixed"}:
        raise ValueError("judge summary status is invalid")
    counts = value["status_counts"]
    if not isinstance(counts, Mapping) or not counts or any(
        key not in _JUDGE_STATUSES or type(count) is not int or count < 0
        for key, count in counts.items()
    ):
        raise ValueError("judge status counts are invalid")
    scores = value["scores"]
    if not isinstance(scores, Mapping) or set(scores) != set(_JUDGE_SCORES):
        raise ValueError("judge scores have an invalid schema")
    normalized_scores = {
        name: _finite_number(scores[name], field=f"judge {name}", nullable=True)
        for name in _JUDGE_SCORES
    }
    errors = value["errors"]
    if not isinstance(errors, list) or any(not isinstance(item, str) or not item for item in errors):
        raise ValueError("judge errors must be nonblank strings")
    coverage = _finite_number(value["coverage"], field="judge coverage")
    if coverage is None or not 0 <= coverage <= 1:
        raise ValueError("judge coverage must be in [0,1]")
    for field in ("prompt_hash", "model_hash"):
        if not isinstance(value[field], str) or not _HASH.fullmatch(value[field]):
            raise ValueError(f"judge {field} must be a SHA-256 digest")
    for field in ("provider", "model"):
        if value[field] is not None and (not isinstance(value[field], str) or not value[field]):
            raise ValueError(f"judge {field} must be a nonblank string or null")
    temperature = value["temperature"]
    if temperature is not None:
        temperature = _finite_number(temperature, field="judge temperature")
    if status in {"judge_not_run", "judge_failed"} and any(
        score is not None for score in normalized_scores.values()
    ):
        raise ValueError("an unscored judge status requires null scores")
    return {
        "status": status,
        "status_counts": dict(sorted(counts.items())),
        "scores": normalized_scores,
        "errors": list(errors),
        "coverage": coverage,
        "prompt_hash": value["prompt_hash"],
        "model_hash": value["model_hash"],
        "provider": value["provider"],
        "model": value["model"],
        "temperature": temperature,
    }


def _judge_summary(rows: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    score_values: dict[str, list[float]] = {name: [] for name in _JUDGE_SCORES}
    errors: set[str] = set()
    coverage: list[float] = []
    identities: dict[str, list[object]] = {
        name: [] for name in ("prompt_hash", "model_hash", "provider", "model", "temperature")
    }
    for row in rows:
        judge = row.get("judge")
        if not isinstance(judge, Mapping):
            raise ValueError("per-query judge payload must be an object")
        status = judge.get("status")
        if status not in _JUDGE_STATUSES:
            raise ValueError("per-query judge status is invalid")
        statuses[status] += 1
        scores = judge.get("scores")
        if status == "judge_completed":
            if not isinstance(scores, Mapping) or set(scores) != set(_JUDGE_SCORES):
                raise ValueError("completed judge requires faithfulness and relevance scores")
            for name in _JUDGE_SCORES:
                score = _finite_number(scores[name], field=f"judge {name}")
                if score is None or not 0 <= score <= 1:
                    raise ValueError(f"judge {name} must be in [0,1]")
                score_values[name].append(score)
        elif scores is not None:
            raise ValueError("not-run or failed judge requires null scores")
        judge_errors = judge.get("errors", ())
        if not isinstance(judge_errors, (list, tuple)) or any(
            not isinstance(item, str) or not item for item in judge_errors
        ):
            raise ValueError("per-query judge errors must be nonblank strings")
        errors.update(judge_errors)
        row_coverage = _finite_number(judge.get("coverage"), field="judge coverage")
        if row_coverage is None or not 0 <= row_coverage <= 1:
            raise ValueError("judge coverage must be in [0,1]")
        coverage.append(row_coverage)
        for name in identities:
            identities[name].append(judge.get(name))
    summary = {
        "status": next(iter(statuses)) if len(statuses) == 1 else "mixed",
        "status_counts": dict(sorted(statuses.items())),
        "scores": {
            name: sum(values) / len(values) if values else None
            for name, values in score_values.items()
        },
        "errors": sorted(errors),
        "coverage": sum(coverage) / len(coverage),
        **{
            name: _consistent(values, field=name)
            for name, values in identities.items()
        },
    }
    return _validate_judge_summary(summary)


def _status_counts(value: object, case_count: int) -> dict[str, int]:
    if not isinstance(value, Mapping) or any(
        key not in _RESULT_STATUSES or type(count) is not int or count < 0
        for key, count in value.items()
    ) or sum(value.values()) != case_count:
        raise ValueError("aggregate status_counts are invalid")
    return dict(sorted(value.items()))


def _retrieval_summary(value: object, case_count: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(_RETRIEVAL_METRICS):
        raise ValueError("aggregate retrieval metrics have an invalid schema")
    result: dict[str, Any] = {}
    for name in _RETRIEVAL_METRICS:
        metric = value[name]
        if not isinstance(metric, Mapping) or set(metric) != {"value", "scored_count", "total_count"}:
            raise ValueError(f"aggregate retrieval metric {name} has an invalid schema")
        scored = metric["scored_count"]
        total = metric["total_count"]
        if type(scored) is not int or type(total) is not int or not 0 <= scored <= total or total != case_count:
            raise ValueError(f"aggregate retrieval metric {name} has invalid counts")
        result[name] = {
            "value": _finite_number(metric["value"], field=name, nullable=True),
            "scored_count": scored,
            "total_count": total,
        }
    return result


def _holdout_aggregate(source: Mapping[str, Any], judge: Mapping[str, Any]) -> dict[str, Any]:
    case_count = source.get("case_count")
    if type(case_count) is not int or case_count < 15:
        raise ValueError("holdout aggregate requires at least 15 cases")
    degradation = source.get("degradation", [])
    if not isinstance(degradation, list):
        raise ValueError("holdout aggregate degradation statuses are invalid")
    public_degradation = [
        item for item in degradation
        if isinstance(item, str) and _DEGRADATION_STATUS.fullmatch(item)
    ]
    return {
        "schema_version": "trade-holdout-aggregate/v1",
        "case_count": case_count,
        "status_counts": _status_counts(source.get("status_counts"), case_count),
        "retrieval": _retrieval_summary(source.get("retrieval"), case_count),
        "judge": _validate_judge_summary(judge),
        "degradation": sorted(set(public_degradation)),
    }


def _published_aggregate(
    source: Mapping[str, Any], role: str, judge: Mapping[str, Any]
) -> dict[str, Any]:
    if role == "holdout":
        return _holdout_aggregate(source, judge)
    result = dict(source)
    result["judge"] = _validate_judge_summary(judge)
    return result


def _environment_manifest(metadata: Mapping[str, Any], persisted: RunManifest) -> dict[str, Any]:
    adapter = metadata["adapter"]
    return {
        "schema_version": "trade-report-environment/v1",
        "backend": adapter.get("backend"),
        "runtime": adapter.get("runtime", {}),
        "backend_statuses": persisted.backend_statuses,
    }


def _model_manifest(
    metadata: Mapping[str, Any], persisted: RunManifest, judge: Mapping[str, Any]
) -> dict[str, Any]:
    adapter = metadata["adapter"]
    return {
        "schema_version": "trade-report-model/v1",
        "model_id": adapter.get("model_id"),
        "model_hash": persisted.snapshot.model_hash,
        "prompt_id": adapter.get("prompt_id"),
        "prompt_hash": persisted.snapshot.prompt_hash,
        "judge": {
            name: judge[name]
            for name in ("provider", "model", "model_hash", "prompt_hash", "temperature")
        },
    }


def _config_manifest(metadata: Mapping[str, Any], persisted: RunManifest) -> dict[str, Any]:
    return {
        "schema_version": "trade-report-config/v1",
        "build_id": metadata["adapter"]["build_id"],
        "profile": metadata.get("profile", {"arm": metadata["arm"]}),
        "budget": metadata.get("budget", {}),
        "snapshot": persisted.snapshot.model_dump(mode="json"),
        "evaluator_id": metadata.get("evaluator_id"),
    }


def _error_analysis(role: str, run: EvaluationRun) -> dict[str, Any]:
    if role == "development":
        return {"status": "available", **ErrorAnalyzer().analyze((run,)).to_dict()}
    return {
        "schema_version": "trade-error-analysis-placeholder/v1",
        "dataset_role": "holdout",
        "status": "not_applicable",
        "reason": "private_holdout_is_never_used_for_development_error_analysis",
        "analysis": None,
    }


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
        source_aggregate = _read_json(run.path / aggregate_artifact)
        if dict(source_aggregate) != dict(run.aggregate):
            raise ValueError("EvaluationRun aggregate does not match its persisted artifact")
        if int(source_aggregate.get("case_count", -1)) != case_count:
            raise ValueError("aggregate case_count does not match run manifest")
        if not (run.path / per_query_artifact).is_file():
            raise ValueError("run per-query artifact is missing")
        rows = _read_rows(run.path / per_query_artifact, persisted.run_id, case_count)
        judge = _judge_summary(rows)
        source_coverage = _finite_number(
            source_aggregate.get("judge_coverage"), field="aggregate judge coverage"
        )
        if source_coverage is None or not math.isclose(source_coverage, judge["coverage"]):
            raise ValueError("aggregate judge coverage does not match per-query judge payloads")
        aggregate = _published_aggregate(source_aggregate, role, judge)

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
            _write_json_new(temporary / "aggregate.json", aggregate)
            if role == "development":
                shutil.copyfile(run.path / per_query_artifact, temporary / "per_query.jsonl")

            environment_manifest = _environment_manifest(metadata, persisted)
            model_manifest = _model_manifest(metadata, persisted, judge)
            config_manifest = _config_manifest(metadata, persisted)
            error_analysis = _error_analysis(role, run)
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
    if checksum_path.is_symlink():
        lines = []
        errors.append("unsafe bundle entry: checksums.sha256")
    else:
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
        case_count = manifest["case_count"]
        minimum = 36 if role == "development" else 15
        if type(case_count) is not int or case_count < minimum:
            errors.append(f"{role} report requires at least {minimum} cases")
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
        if aggregate.get("case_count") != case_count:
            errors.append("aggregate case_count does not match run manifest")
        if role == "development":
            rows = _read_rows(path / "per_query.jsonl", persisted.run_id, case_count)
            judge = _judge_summary(rows)
            source_coverage = _finite_number(
                aggregate.get("judge_coverage"), field="aggregate judge coverage"
            )
            if source_coverage is None or not math.isclose(source_coverage, judge["coverage"]):
                errors.append("aggregate judge coverage mismatch")
            expected_aggregate = _published_aggregate(aggregate, role, judge)
        else:
            judge = _validate_judge_summary(aggregate.get("judge"))
            expected_aggregate = _holdout_aggregate(aggregate, judge)
        if aggregate != expected_aggregate:
            errors.append(f"{role} aggregate schema mismatch")
        if sum(judge["status_counts"].values()) != case_count:
            errors.append("judge status counts do not match case_count")

        environment = _read_json(path / "environment_manifest.json")
        model = _read_json(path / "model_manifest.json")
        config = _read_json(path / "config_manifest.json")
        analysis = _read_json(path / "error_analysis.json")
        if environment != _environment_manifest(manifest, persisted):
            errors.append("environment manifest mismatch")
        if model != _model_manifest(manifest, persisted, judge):
            errors.append("model manifest mismatch")
        if config != _config_manifest(manifest, persisted):
            errors.append("config manifest mismatch")
        expected_analysis = _error_analysis(
            role, EvaluationRun(path=path, manifest=persisted, aggregate=aggregate)
        )
        if analysis != expected_analysis:
            errors.append("error analysis mismatch")
        if report.get("build_id") != manifest["adapter"]["build_id"]:
            errors.append("build identity mismatch")
        if report.get("profile") != manifest["arm"]:
            errors.append("profile identity mismatch")
        if report.get("error_analysis_status") != analysis.get("status"):
            errors.append("error analysis status mismatch")

        markdown = (path / "report.md").read_text(encoding="utf-8")
        if markdown != ReportWriter._markdown(report):
            errors.append("report Markdown mismatch")
        if role == "holdout":
            if report.get("aggregate") != _holdout_aggregate(aggregate, judge):
                errors.append("holdout redaction failure: report.json")
            if _HOLDOUT_MARKDOWN_FORBIDDEN.search(markdown):
                errors.append("holdout redaction failure: report.md")
            if "per_query.jsonl" in actual_names:
                errors.append("holdout redaction failure: per_query.jsonl")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        errors.append(f"invalid bundle metadata: {exc}")
    return VerificationResult(path, not errors, tuple(dict.fromkeys(errors)))
