"""Preregistered, development-only selection over paired evaluation runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

from trade_agent.evaluation.error_analysis import read_development_metadata, read_per_query_rows
from trade_agent.evaluation.runner import EvaluationRun


@dataclass(frozen=True)
class OptimizationObjective:
    baseline_run_id: str | None = None
    max_mean_latency_increase_ms: float = 50.0
    max_refusal_rate_increase: float = 0.0

    def __post_init__(self) -> None:
        if self.baseline_run_id is not None and not self.baseline_run_id.strip():
            raise ValueError("baseline_run_id must be nonblank")
        if not math.isfinite(self.max_mean_latency_increase_ms) or self.max_mean_latency_increase_ms < 0:
            raise ValueError("max_mean_latency_increase_ms must be finite and nonnegative")
        if not math.isfinite(self.max_refusal_rate_increase) or not 0 <= self.max_refusal_rate_increase <= 1:
            raise ValueError("max_refusal_rate_increase must be between zero and one")


@dataclass(frozen=True)
class PairedDelta:
    baseline_mean: float | None
    candidate_mean: float | None
    delta: float | None
    paired_count: int


@dataclass(frozen=True)
class OptimizationDecision:
    baseline_run_id: str
    selected_run_id: str
    accepted_change: bool
    objective_priority: tuple[str, ...]
    selected_metrics: Mapping[str, float | None]
    paired_deltas: Mapping[str, PairedDelta]
    rejected_candidates: Mapping[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "trade-optimization-decision/v1",
            "dataset_role": "development",
            "baseline_run_id": self.baseline_run_id,
            "selected_run_id": self.selected_run_id,
            "accepted_change": self.accepted_change,
            "objective_priority": list(self.objective_priority),
            "selected_metrics": dict(self.selected_metrics),
            "paired_deltas": {name: asdict(value) for name, value in self.paired_deltas.items()},
            "rejected_candidates": {name: list(reasons) for name, reasons in self.rejected_candidates.items()},
        }


class DevelopmentOptimizer:
    """Select one bounded change using rule metrics from development rows only."""

    priority = ("recall_at_10", "context_precision", "faithfulness")

    def select(
        self,
        candidates: Sequence[EvaluationRun],
        objective: OptimizationObjective | Mapping[str, Any],
    ) -> OptimizationDecision:
        runs = tuple(candidates)
        if not runs:
            raise ValueError("optimizer requires at least one development run")
        if isinstance(objective, Mapping):
            objective = OptimizationObjective(**objective)
        if not isinstance(objective, OptimizationObjective):
            raise TypeError("objective must be OptimizationObjective or a compatible mapping")

        # Role-check every input before opening any result artifact.
        metadata = tuple(read_development_metadata(run) for run in runs)
        run_ids = [run.manifest.run_id for run in runs]
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("candidate run IDs must be unique")
        baseline_id = objective.baseline_run_id or run_ids[0]
        if baseline_id not in run_ids:
            raise ValueError("baseline_run_id must identify one candidate run")
        dataset_hashes = {run.manifest.snapshot.dataset_hash for run in runs}
        if len(dataset_hashes) != 1:
            raise ValueError("optimizer candidates must use one development dataset hash")

        rows = {
            run.manifest.run_id: _index_rows(read_per_query_rows(run, meta))
            for run, meta in zip(runs, metadata, strict=True)
        }
        baseline = rows[baseline_id]
        baseline_cases = set(baseline)
        rejected: dict[str, tuple[str, ...]] = {}
        summaries: dict[str, tuple[dict[str, PairedDelta], dict[str, float | None]]] = {}
        eligible = [baseline_id]
        for run_id in run_ids:
            if run_id == baseline_id:
                continue
            current = rows[run_id]
            if set(current) != baseline_cases:
                rejected[run_id] = ("unpaired_case_set",)
                continue
            deltas = _paired_deltas(baseline, current)
            summaries[run_id] = (deltas, _candidate_metrics(current))
            reasons = []
            latency = deltas["latency_ms"].delta
            refusal = deltas["refusal_rate"].delta
            if latency is None or latency > objective.max_mean_latency_increase_ms:
                reasons.append("latency_constraint")
            if refusal is None or refusal > objective.max_refusal_rate_increase:
                reasons.append("refusal_constraint")
            if reasons:
                rejected[run_id] = tuple(reasons)
            else:
                eligible.append(run_id)

        baseline_metrics = _candidate_metrics(baseline)
        scores = {baseline_id: _score(baseline_metrics)}
        scores.update({run_id: _score(summaries[run_id][1]) for run_id in eligible if run_id != baseline_id})
        selected_id = max(eligible, key=lambda run_id: (scores[run_id], run_id == baseline_id))
        if selected_id == baseline_id:
            selected_metrics = baseline_metrics
            paired = _paired_deltas(baseline, baseline)
        else:
            paired, selected_metrics = summaries[selected_id]
        return OptimizationDecision(
            baseline_run_id=baseline_id,
            selected_run_id=selected_id,
            accepted_change=selected_id != baseline_id,
            objective_priority=self.priority,
            selected_metrics=selected_metrics,
            paired_deltas=paired,
            rejected_candidates=dict(sorted(rejected.items())),
        )


def _index_rows(rows: tuple[Mapping[str, Any], ...]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        case_id = str(row["result"]["case_id"])
        if case_id in result:
            raise ValueError(f"duplicate case ID in run: {case_id}")
        result[case_id] = row
    return result


def _paired_deltas(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
) -> dict[str, PairedDelta]:
    result = {}
    for name in (*DevelopmentOptimizer.priority, "latency_ms", "refusal_rate"):
        pairs = [
            (_value(baseline[case_id], name), _value(candidate[case_id], name))
            for case_id in sorted(baseline)
        ]
        valid = [(left, right) for left, right in pairs if left is not None and right is not None]
        before = _mean(left for left, _right in valid)
        after = _mean(right for _left, right in valid)
        result[name] = PairedDelta(before, after, None if before is None else after - before, len(valid))
    return result


def _candidate_metrics(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, float | None]:
    return {name: _mean(_value(row, name) for row in rows.values()) for name in DevelopmentOptimizer.priority}


def _value(row: Mapping[str, Any], name: str) -> float | None:
    if name == "latency_ms":
        value = row.get("result", {}).get("latency_ms")
    elif name == "refusal_rate":
        return 1.0 if row.get("result", {}).get("status") == "refused" else 0.0
    elif name == "faithfulness":
        value = row.get("metrics", {}).get("generation")
        value = value.get("faithfulness", {}).get("faithfulness") if isinstance(value, Mapping) else None
    else:
        value = row.get("metrics", {}).get("retrieval")
        value = value.get(name) if isinstance(value, Mapping) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _mean(values) -> float | None:
    measured = [float(value) for value in values if value is not None]
    return sum(measured) / len(measured) if measured else None


def _score(metrics: Mapping[str, float | None]) -> tuple[float, float, float]:
    return tuple(-math.inf if metrics[name] is None else metrics[name] for name in DevelopmentOptimizer.priority)
