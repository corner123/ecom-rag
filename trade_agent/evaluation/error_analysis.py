"""Development-only diagnosis of measured trade evaluation failures."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from trade_agent.evaluation.models import RunManifest
from trade_agent.evaluation.runner import EvaluationRun


ERROR_BUCKETS = (
    "missed_required_evidence",
    "keyword_miss",
    "filter_false_negative",
    "filter_false_positive",
    "stale_source",
    "wrong_source_prior",
    "dedupe_error",
    "conflict_missed",
    "reranker_regression",
    "unsupported_claim",
    "unsafe_decision",
    "backend_degraded",
)

_SUGGESTED_COMPONENTS = {
    "missed_required_evidence": "candidate_retrieval",
    "keyword_miss": "tokenizer_or_bm25",
    "filter_false_negative": "metadata_filter",
    "filter_false_positive": "metadata_filter",
    "stale_source": "source_ingestion",
    "wrong_source_prior": "fusion_weights",
    "dedupe_error": "deduplicator",
    "conflict_missed": "conflict_detector",
    "reranker_regression": "reranker",
    "unsupported_claim": "claim_guard",
    "unsafe_decision": "decision_policy",
    "backend_degraded": "runtime_backend",
}


class HoldoutPolicyError(ValueError):
    """Raised before private holdout results can enter development analysis."""


@dataclass(frozen=True)
class ErrorCase:
    run_id: str
    case_id: str
    bucket: str
    suggested_component: str
    observed_value: float | None
    comparison_run_id: str | None = None
    metric_delta: float | None = None
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ErrorAnalysis:
    cases: tuple[ErrorCase, ...]
    bucket_counts: Mapping[str, int]
    analyzed_run_ids: tuple[str, ...]
    dataset_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "trade-error-analysis/v1",
            "dataset_role": "development",
            "dataset_hash": self.dataset_hash,
            "analyzed_run_ids": list(self.analyzed_run_ids),
            "bucket_counts": dict(self.bucket_counts),
            "cases": [asdict(case) for case in self.cases],
        }


@dataclass(frozen=True)
class _RunData:
    run: EvaluationRun
    metadata: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]

    @property
    def components(self) -> frozenset[str]:
        return frozenset(str(item) for item in self.metadata.get("enabled_components", ()))


def reject_protected_holdout_path(path: Path) -> None:
    """Reject known holdout locations without touching the filesystem."""

    parts = tuple(part.casefold() for part in Path(path).parts)
    if any("holdout" in part for part in parts):
        raise HoldoutPolicyError(f"holdout input is forbidden for development optimization: {path}")
    protected = ("data", "eval", "private", "trade_intel")
    if any(parts[index : index + len(protected)] == protected for index in range(len(parts) - len(protected) + 1)):
        raise HoldoutPolicyError(f"private holdout path is forbidden for development optimization: {path}")


def read_development_metadata(run: EvaluationRun) -> Mapping[str, Any]:
    reject_protected_holdout_path(run.path)
    path = run.path / "manifest.json"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read run manifest {path}: {exc}") from exc
    if metadata.get("dataset_role") != "development":
        raise HoldoutPolicyError(f"holdout input is forbidden for development optimization: {run.path}")
    try:
        persisted = RunManifest.model_validate(metadata["run"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid Task 5 run manifest: {path}") from exc
    if persisted != run.manifest:
        raise ValueError(f"EvaluationRun does not match persisted manifest: {path}")
    return metadata


def read_per_query_rows(run: EvaluationRun, metadata: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    artifact = metadata.get("artifacts", {}).get("per_query", "per_query.jsonl")
    artifact_path = Path(str(artifact))
    if artifact_path.is_absolute() or ".." in artifact_path.parts:
        raise ValueError("per-query artifact must be a safe path inside the run directory")
    path = run.path / artifact_path
    rows: list[Mapping[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"line {line_number} is not an object")
            if value.get("result", {}).get("run_id") != run.manifest.run_id:
                raise ValueError(f"line {line_number} has a mismatched run_id")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read per-query artifact {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"run has no per-query rows: {path}")
    return tuple(rows)


def load_evaluation_runs(paths: Sequence[Path]) -> tuple[EvaluationRun, ...]:
    """Load Task 5 run directories, validating role before reading aggregates."""

    directories: list[Path] = []
    for value in paths:
        path = Path(value)
        reject_protected_holdout_path(path)
        if (path / "manifest.json").is_file():
            directories.append(path)
        elif path.is_dir():
            directories.extend(sorted(item.parent for item in path.rglob("manifest.json")))
        else:
            raise ValueError(f"run path does not exist: {path}")
    unique = tuple(dict.fromkeys(directories))
    if not unique:
        raise ValueError("no Task 5 run directories found")

    manifests: list[tuple[Path, Mapping[str, Any], RunManifest]] = []
    for path in unique:
        reject_protected_holdout_path(path)
        try:
            metadata = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read run manifest {path}: {exc}") from exc
        if metadata.get("dataset_role") != "development":
            raise HoldoutPolicyError(f"holdout input is forbidden for development optimization: {path}")
        manifests.append((path, metadata, RunManifest.model_validate(metadata["run"])))

    runs = []
    for path, _metadata, manifest in manifests:
        aggregate_path = path / "aggregate.json"
        try:
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read run aggregate {aggregate_path}: {exc}") from exc
        runs.append(EvaluationRun(path=path, manifest=manifest, aggregate=aggregate))
    return tuple(runs)


class ErrorAnalyzer:
    """Turn failed development rows into one reproducible actionable bucket each."""

    def analyze(self, runs: Sequence[EvaluationRun]) -> ErrorAnalysis:
        values = tuple(runs)
        if not values:
            raise ValueError("error analysis requires at least one development run")
        # Complete the metadata preflight for every run before reading any result row.
        metadata = tuple(read_development_metadata(run) for run in values)
        dataset_hashes = {run.manifest.snapshot.dataset_hash for run in values}
        if len(dataset_hashes) != 1:
            raise ValueError("paired error analysis requires one development dataset hash")
        data = tuple(
            _RunData(run, meta, read_per_query_rows(run, meta))
            for run, meta in zip(values, metadata, strict=True)
        )
        row_index = {
            (item.run.manifest.run_id, str(row["result"]["case_id"])): row
            for item in data
            for row in item.rows
        }
        failures: list[ErrorCase] = []
        for item in data:
            for row in item.rows:
                failure = self._classify(item, row, data, row_index)
                if failure is not None:
                    failures.append(failure)
        counts = Counter(case.bucket for case in failures)
        return ErrorAnalysis(
            cases=tuple(failures),
            bucket_counts=dict(sorted(counts.items())),
            analyzed_run_ids=tuple(item.run.manifest.run_id for item in data),
            dataset_hash=next(iter(dataset_hashes)),
        )

    def _classify(self, current: _RunData, row: Mapping[str, Any], all_runs: tuple[_RunData, ...],
                  row_index: Mapping[tuple[str, str], Mapping[str, Any]]) -> ErrorCase | None:
        run_id = current.run.manifest.run_id
        case_id = str(row["result"]["case_id"])
        retrieval = row.get("metrics", {}).get("retrieval")
        recall = _number(retrieval, "recall_at_10")
        precision = _number(retrieval, "context_precision")
        evidence = tuple(str(item) for item in row.get("trace", {}).get("degradation", ()))
        evidence += tuple(str(item) for item in row.get("errors", ()))
        statuses = row.get("trace", {}).get("backend_statuses", row["result"].get("backend_statuses", {}))
        if retrieval is None or any(value in {"degraded", "unavailable", "failed"} for value in statuses.values()):
            return self._case(run_id, case_id, "backend_degraded", recall, evidence=evidence)
        searchable = " ".join(evidence).casefold()
        if "source_stale" in searchable or "stale source" in searchable or "source:stale" in searchable:
            return self._case(run_id, case_id, "stale_source", recall, evidence=evidence)
        if "wrong_source_prior" in searchable or "wrong authority" in searchable:
            return self._case(run_id, case_id, "wrong_source_prior", recall, evidence=evidence)
        hit_ids = [
            str(hit.get("evidence_id", hit.get("chunk_id", "")))
            for hit in row.get("trace", {}).get("hits", ())
        ]
        if any(identifier and hit_ids.count(identifier) > 1 for identifier in set(hit_ids)):
            return self._case(run_id, case_id, "dedupe_error", recall, evidence=tuple(hit_ids))
        fusion = row.get("metrics", {}).get("fusion")
        if _explicit_failure(
            fusion,
            ("conflict_correct", "escalation_correct"),
            ("accuracy", "conflict_accuracy", "macro_f1", "escalation_correctness"),
        ):
            return self._case(run_id, case_id, "conflict_missed", recall)
        generation_outcome = row.get("generation_outcome")
        guard = generation_outcome.get("guard") if isinstance(generation_outcome, Mapping) else None
        answer = generation_outcome.get("answer") if isinstance(generation_outcome, Mapping) else None
        bypassed_guard = (
            isinstance(guard, Mapping)
            and guard.get("accepted") is False
            and isinstance(answer, Mapping)
            and not answer.get("refusal_reason")
            and row["result"].get("status") != "refused"
        )
        if bypassed_guard:
            return self._case(run_id, case_id, "unsafe_decision", None)
        generation = row.get("metrics", {}).get("generation")
        faithfulness = _nested_number(generation, "faithfulness", "faithfulness")
        unsupported = tuple(
            str(value)
            for value in (generation or {}).get("faithfulness", {}).get("unsupported_factual_claim_ids", ())
        )
        if unsupported or (faithfulness is not None and faithfulness < 1.0):
            return self._case(run_id, case_id, "unsupported_claim", faithfulness, evidence=unsupported)
        business = row.get("metrics", {}).get("business")
        if _explicit_failure(business, ("safe_decision", "decision_allowed"), ("safety_rate",)):
            return self._case(run_id, case_id, "unsafe_decision", None)

        if "reranker" in current.components:
            paired = self._best_predecessor(current, case_id, all_runs, row_index, "reranker")
            regression = _regression(row, paired[1]) if paired else None
            if regression is not None:
                return self._case(run_id, case_id, "reranker_regression", regression[0],
                                  comparison_run_id=paired[0], metric_delta=regression[1])
        if "filter" in current.components:
            paired = self._best_predecessor(current, case_id, all_runs, row_index, "filter")
            regression = _regression(row, paired[1]) if paired else None
            if regression is not None:
                return self._case(run_id, case_id, "filter_false_negative", regression[0],
                                  comparison_run_id=paired[0], metric_delta=regression[1])
            noise = _number(retrieval, "retrieval_noise_count")
            if noise is not None and noise > 0:
                return self._case(run_id, case_id, "filter_false_positive", precision)
        if current.components == {"dense"} and recall is not None and recall < 1.0:
            bm25 = next((item for item in all_runs if item.components == {"bm25"}), None)
            paired_row = row_index.get((bm25.run.manifest.run_id, case_id)) if bm25 else None
            paired_recall = _retrieval_number(paired_row, "recall_at_10")
            if paired_recall is not None and paired_recall > recall:
                return self._case(run_id, case_id, "keyword_miss", recall,
                                  comparison_run_id=bm25.run.manifest.run_id,
                                  metric_delta=paired_recall - recall)
        if recall == 1.0 and precision is not None and precision < 1.0:
            return self._case(run_id, case_id, "wrong_source_prior", precision)
        if recall is not None and recall < 1.0:
            return self._case(run_id, case_id, "missed_required_evidence", recall)
        return None

    @staticmethod
    def _best_predecessor(current: _RunData, case_id: str, all_runs: tuple[_RunData, ...],
                          row_index: Mapping[tuple[str, str], Mapping[str, Any]], component: str):
        expected = current.components - {component}
        candidates = [item for item in all_runs if item.components == expected]
        if not candidates:
            return None
        predecessor = candidates[0]
        row = row_index.get((predecessor.run.manifest.run_id, case_id))
        return None if row is None else (predecessor.run.manifest.run_id, row)

    @staticmethod
    def _case(run_id: str, case_id: str, bucket: str, observed: float | None, *,
              comparison_run_id: str | None = None, metric_delta: float | None = None,
              evidence: tuple[str, ...] = ()) -> ErrorCase:
        return ErrorCase(run_id, case_id, bucket, _SUGGESTED_COMPONENTS[bucket], observed,
                         comparison_run_id, metric_delta, evidence)


def _number(value: Any, key: str) -> float | None:
    if not isinstance(value, Mapping):
        return None
    item = value.get(key)
    return float(item) if isinstance(item, (int, float)) and not isinstance(item, bool) else None


def _nested_number(value: Any, outer: str, inner: str) -> float | None:
    return _number(value.get(outer), inner) if isinstance(value, Mapping) else None


def _retrieval_number(row: Mapping[str, Any] | None, key: str) -> float | None:
    if row is None:
        return None
    return _number(row.get("metrics", {}).get("retrieval"), key)


def _regression(current: Mapping[str, Any], predecessor: Mapping[str, Any]) -> tuple[float, float] | None:
    for metric in ("recall_at_10", "context_precision"):
        observed = _retrieval_number(current, metric)
        previous = _retrieval_number(predecessor, metric)
        if observed is not None and previous is not None and observed < previous:
            return observed, previous - observed
    return None


def _explicit_failure(value: Any, booleans: tuple[str, ...], rates: tuple[str, ...]) -> bool:
    if not isinstance(value, Mapping):
        return False
    return any(value.get(key) is False for key in booleans) or any(
        _number(value, key) is not None and _number(value, key) < 1.0 for key in rates
    )
