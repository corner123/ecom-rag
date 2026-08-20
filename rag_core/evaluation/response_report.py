"""Auditable aggregation and comparison for generated-answer experiments.

The module is deliberately independent from RAGAS and model clients.  It
consumes persisted per-question records, which makes aggregation repeatable
without paying for another generation or judge run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any


RESPONSE_METRICS = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
)
DETERMINISTIC_RETRIEVAL_PRIMARY_METRICS = (
    "hit_at_k",
    "required_claim_recall_at_k",
    "mrr",
    "source_precision_at_k",
)
DETERMINISTIC_RETRIEVAL_DIAGNOSTIC_METRICS = (
    "source_option_recall",
)
DETERMINISTIC_RETRIEVAL_METRICS = (
    *DETERMINISTIC_RETRIEVAL_PRIMARY_METRICS,
    *DETERMINISTIC_RETRIEVAL_DIAGNOSTIC_METRICS,
)
LATENCY_SCOPES = ("retrieval_latency_ms", "generation_latency_ms", "total_latency_ms")
GENERATION_TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
)
RESPONSE_ARTIFACT_SCHEMA = "engineering-response-experiment/v1"
RESPONSE_ARTIFACT_STAGE = "judged"
_PAIRED_IDENTITY_FIELDS = (
    "dataset_file_sha256",
    "dataset_canonical_sha256",
    "manifest_build_id",
    "index_catalog_sha256",
    "top_k",
)


class ResponseReportValidationError(ValueError):
    """Raised when persisted experiments are not safe to compare."""


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_judged_response_artifact(path: str | Path) -> dict[str, Any]:
    """Load one complete response artifact without invoking retrieval or models."""

    candidate = Path(path)
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResponseReportValidationError(
            f"response artifact does not exist: {candidate}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ResponseReportValidationError(
            f"response artifact is not readable JSON: {candidate}"
        ) from exc
    if not isinstance(payload, dict):
        raise ResponseReportValidationError("response artifact must be a JSON object")
    if payload.get("schema_version") != RESPONSE_ARTIFACT_SCHEMA:
        raise ResponseReportValidationError(
            f"unsupported response artifact schema: {payload.get('schema_version')!r}"
        )
    if payload.get("stage") != RESPONSE_ARTIFACT_STAGE:
        raise ResponseReportValidationError(
            "response artifact must have stage='judged' before aggregation"
        )
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ResponseReportValidationError("response artifact metadata must be an object")
    for field in _PAIRED_IDENTITY_FIELDS:
        value = metadata.get(field)
        if field == "top_k":
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 50:
                raise ResponseReportValidationError(
                    "response artifact metadata.top_k must be an integer from 1 to 50"
                )
        elif not isinstance(value, str) or not value.strip():
            raise ResponseReportValidationError(
                f"response artifact metadata.{field} must be a non-empty string"
            )
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ResponseReportValidationError(
            "judged response artifact records must be a non-empty array"
        )
    ids: list[str] = []
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise ResponseReportValidationError(
                f"response artifact record {position} must be an object"
            )
        sample_id = str(record.get("id") or "").strip()
        if not sample_id:
            raise ResponseReportValidationError(
                f"response artifact record {position} requires a non-empty id"
            )
        if not isinstance(record.get("answerable"), bool):
            raise ResponseReportValidationError(
                f"response artifact record {sample_id!r} requires a boolean answerable label"
            )
        if not isinstance(record.get("metrics"), dict):
            raise ResponseReportValidationError(
                f"response artifact record {sample_id!r} requires a metrics object"
            )
        ids.append(sample_id)
    if len(set(ids)) != len(ids):
        raise ResponseReportValidationError(
            "response artifact requires unique non-empty record ids"
        )
    payload["_artifact_sha256"] = _artifact_sha256(candidate)
    return payload


def _paired_artifact_inputs(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_meta = baseline["metadata"]
    candidate_meta = candidate["metadata"]
    for field in _PAIRED_IDENTITY_FIELDS:
        if baseline_meta[field] != candidate_meta[field]:
            raise ResponseReportValidationError(
                f"baseline and candidate metadata.{field} must match"
            )

    baseline_records = {str(record["id"]): record for record in baseline["records"]}
    candidate_records = {str(record["id"]): record for record in candidate["records"]}
    if set(baseline_records) != set(candidate_records):
        raise ResponseReportValidationError(
            "baseline and candidate must contain exactly the same question ids"
        )
    for sample_id in sorted(baseline_records):
        before = baseline_records[sample_id]
        after = candidate_records[sample_id]
        if before.get("question") != after.get("question"):
            raise ResponseReportValidationError(
                f"question text differs for paired id {sample_id!r}"
            )
        if before["answerable"] != after["answerable"]:
            raise ResponseReportValidationError(
                f"answerable label differs for paired id {sample_id!r}"
            )
    return {
        "dataset_file_sha256": baseline_meta["dataset_file_sha256"],
        "dataset_canonical_sha256": baseline_meta["dataset_canonical_sha256"],
        "manifest_build_id": baseline_meta["manifest_build_id"],
        "index_catalog_sha256": baseline_meta["index_catalog_sha256"],
        "top_k": baseline_meta["top_k"],
        "question_count": len(baseline_records),
        "baseline_artifact_sha256": baseline["_artifact_sha256"],
        "candidate_artifact_sha256": candidate["_artifact_sha256"],
    }


def build_response_comparison_from_artifacts(
    *,
    baseline_path: str | Path,
    candidate_path: str | Path,
    baseline_name: str,
    candidate_name: str,
    user_metadata: Mapping[str, Any] | None = None,
    minimum_judge_coverage: float = 0.95,
) -> dict[str, Any]:
    """Validate and aggregate two persisted judged artifacts."""

    if not baseline_name.strip() or not candidate_name.strip():
        raise ResponseReportValidationError("baseline and candidate names are required")
    if baseline_name == candidate_name:
        raise ResponseReportValidationError(
            "baseline and candidate names must be different"
        )
    if not 0.0 <= minimum_judge_coverage <= 1.0:
        raise ResponseReportValidationError(
            "minimum judge coverage must be between 0 and 1"
        )
    baseline = load_judged_response_artifact(baseline_path)
    candidate = load_judged_response_artifact(candidate_path)
    comparison_inputs = _paired_artifact_inputs(baseline, candidate)
    metadata: dict[str, Any] = {"comparison_inputs": comparison_inputs}
    if user_metadata is not None:
        if not isinstance(user_metadata, Mapping):
            raise ResponseReportValidationError("user metadata must be a JSON object")
        metadata["user"] = dict(user_metadata)
    return build_response_comparison_report(
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        baseline_records=baseline["records"],
        candidate_records=candidate["records"],
        metadata=metadata,
        minimum_judge_coverage=minimum_judge_coverage,
    )


def load_report_metadata(value: str | Path | None) -> dict[str, Any] | None:
    """Parse ``--metadata-json`` as an existing file or inline JSON object."""

    if value is None:
        return None
    raw = str(value)
    candidate = Path(raw)
    try:
        text = candidate.read_text(encoding="utf-8") if candidate.is_file() else raw
        payload = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        raise ResponseReportValidationError(
            "--metadata-json must be an existing JSON file or an inline JSON object"
        ) from exc
    if not isinstance(payload, dict):
        raise ResponseReportValidationError("--metadata-json must contain a JSON object")
    return payload


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def aggregate_response_records(
    records: Sequence[Mapping[str, Any]],
    *,
    minimum_judge_coverage: float = 0.95,
) -> dict[str, Any]:
    """Aggregate one strategy while preserving judge failures as missing.

    RAGAS response metrics are eligible only for answerable, successfully
    generated responses.  A provider or judge failure never becomes a zero.
    Refusal metrics are computed over all answerability labels.
    """

    if not records:
        raise ValueError("at least one response-evaluation record is required")
    ids = [str(record.get("id") or "").strip() for record in records]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("response records require unique non-empty ids")

    answerable = [record for record in records if bool(record.get("answerable"))]
    unanswerable = [record for record in records if not bool(record.get("answerable"))]
    refusal_tp = sum(bool(record.get("refused")) for record in unanswerable)
    refusal_fp = sum(bool(record.get("refused")) for record in answerable)
    refusal_fn = len(unanswerable) - refusal_tp
    refusal_precision = (
        refusal_tp / (refusal_tp + refusal_fp) if refusal_tp + refusal_fp else 0.0
    )
    refusal_recall = refusal_tp / (refusal_tp + refusal_fn) if unanswerable else 0.0
    refusal_f1 = (
        2 * refusal_precision * refusal_recall / (refusal_precision + refusal_recall)
        if refusal_precision + refusal_recall
        else 0.0
    )

    eligible = [
        record
        for record in answerable
        if bool(record.get("generation_succeeded")) and not bool(record.get("refused"))
    ]
    answerable_response_coverage = (
        len(eligible) / len(answerable) if answerable else 0.0
    )
    metric_summary: dict[str, Any] = {}
    coverage_values: list[float] = []
    for metric in RESPONSE_METRICS:
        values: list[float] = []
        for record in eligible:
            raw = (record.get("metrics") or {}).get(metric)
            if raw is not None:
                values.append(float(raw))
        coverage = len(values) / len(eligible) if eligible else 0.0
        coverage_values.append(coverage)
        metric_summary[metric] = {
            "mean": mean(values) if values else None,
            "count": len(values),
            "eligible": len(eligible),
            "coverage": coverage,
        }

    retrieval_metric_summary: dict[str, Any] = {}
    retrieval_coverage_values: list[float] = []
    for metric in DETERMINISTIC_RETRIEVAL_METRICS:
        values = [
            float(raw)
            for record in answerable
            if (
                raw := (record.get("deterministic_retrieval_metrics") or {}).get(
                    metric
                )
            )
            is not None
        ]
        coverage = len(values) / len(answerable) if answerable else 0.0
        retrieval_coverage_values.append(coverage)
        retrieval_metric_summary[metric] = {
            "mean": mean(values) if values else None,
            "count": len(values),
            "eligible": len(answerable),
            "coverage": coverage,
        }

    latency_summary: dict[str, Any] = {}
    for scope in LATENCY_SCOPES:
        values = [
            float(record[scope])
            for record in records
            if record.get(scope) is not None
        ]
        latency_summary[scope] = {
            "mean": mean(values) if values else None,
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "p99_exploratory": _percentile(values, 0.99),
            "count": len(values),
        }

    attempted_generation = [
        record
        for record in records
        if bool((record.get("generation_status") or {}).get("attempted"))
    ]
    usage_records = [
        record
        for record in attempted_generation
        if isinstance(record.get("generation_usage"), Mapping)
        and bool(record.get("generation_usage"))
    ]
    token_summary: dict[str, Any] = {}
    for field in GENERATION_TOKEN_FIELDS:
        values = [
            int(value)
            for record in usage_records
            if type(value := (record.get("generation_usage") or {}).get(field)) is int
            and value >= 0
        ]
        token_summary[field] = {
            "total": sum(values) if values else None,
            "mean_per_reported_call": mean(values) if values else None,
            "count": len(values),
        }
    generation_usage_summary = {
        "attempted_call_count": len(attempted_generation),
        "reported_usage_count": len(usage_records),
        "coverage": (
            len(usage_records) / len(attempted_generation)
            if attempted_generation
            else 0.0
        ),
        "tokens": token_summary,
    }

    judge_coverage = min(coverage_values) if coverage_values else 0.0
    deterministic_retrieval_coverage = (
        min(retrieval_coverage_values) if retrieval_coverage_values else 0.0
    )
    failure_codes: dict[str, int] = {}
    retrieval_failure_codes: dict[str, int] = {}
    infrastructure_failure_codes: dict[str, int] = {}
    judge_metric_slots = len(eligible) * len(RESPONSE_METRICS)
    judge_executed_slots = 0
    for record in records:
        for error in record.get("metric_errors") or ():
            code = str(error.get("code") or "unknown_judge_error")
            failure_codes[code] = failure_codes.get(code, 0) + 1
        failure = record.get("failure")
        if isinstance(failure, Mapping):
            code = str(failure.get("code") or "unknown_run_failure")
            infrastructure_failure_codes[code] = (
                infrastructure_failure_codes.get(code, 0) + 1
            )
        retrieval_error = record.get("deterministic_retrieval_metric_error")
        if isinstance(retrieval_error, Mapping):
            code = str(
                retrieval_error.get("code") or "unknown_retrieval_metric_error"
            )
            retrieval_failure_codes[code] = retrieval_failure_codes.get(code, 0) + 1

    for record in eligible:
        attempts = record.get("judge_attempts") or ()
        attempted_metrics = {
            str(attempt.get("metric"))
            for attempt in attempts
            if isinstance(attempt, Mapping) and attempt.get("metric")
        }
        current_errors = record.get("metric_errors") or ()
        historical_errors = record.get("error_history") or ()
        for metric in RESPONSE_METRICS:
            metrics = record.get("metrics") or {}
            if metrics.get(metric) is not None or metric in attempted_metrics:
                judge_executed_slots += 1
                continue
            # v1 artifacts written before per-metric attempts are inferred only
            # when an explicit Judge error exists; null alone is not execution.
            errors = (*current_errors, *historical_errors)
            if any(
                isinstance(error, Mapping)
                and (
                    error.get("metric") == metric
                    or (
                        not error.get("metric")
                        and str(error.get("stage") or "").endswith("judge")
                    )
                    or (
                        not error.get("metric")
                        and error.get("code") == "judge_timeout"
                    )
                )
                for error in errors
            ):
                judge_executed_slots += 1

    run_completed = 0
    for record in records:
        failure = record.get("failure")
        semantic_outcome = (
            bool(record.get("refused"))
            or bool(record.get("generation_succeeded"))
            or isinstance(record.get("generation_status"), Mapping)
        )
        if not failure and semantic_outcome:
            run_completed += 1
    run_execution_coverage = run_completed / len(records)
    judge_execution_coverage = (
        judge_executed_slots / judge_metric_slots if judge_metric_slots else 1.0
    )
    artifact_complete = (
        run_execution_coverage == 1.0
        and judge_execution_coverage == 1.0
        and deterministic_retrieval_coverage == 1.0
    )
    response_metrics_publishable = (
        artifact_complete and judge_coverage >= minimum_judge_coverage
    )

    return {
        "question_count": len(records),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "generation_success_rate": (
            sum(bool(record.get("generation_succeeded")) for record in answerable)
            / len(answerable)
            if answerable
            else 0.0
        ),
        "answerable_response_coverage": answerable_response_coverage,
        "answerable_refusal_rate": (
            refusal_fp / len(answerable) if answerable else 0.0
        ),
        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,
        "refusal_f1": refusal_f1,
        "judge_coverage": judge_coverage,
        "judge_score_coverage": judge_coverage,
        "judge_execution_coverage": judge_execution_coverage,
        "run_execution_coverage": run_execution_coverage,
        "artifact_complete": artifact_complete,
        "response_metrics_publishable": response_metrics_publishable,
        "deterministic_retrieval_coverage": deterministic_retrieval_coverage,
        "minimum_judge_coverage": minimum_judge_coverage,
        "minimum_deterministic_retrieval_coverage": 1.0,
        # Compatibility alias retained for existing report consumers.
        "publishable": response_metrics_publishable,
        "metrics": metric_summary,
        "deterministic_retrieval_metrics": retrieval_metric_summary,
        "latency": latency_summary,
        "generation_usage": generation_usage_summary,
        "metric_failure_codes": failure_codes,
        "deterministic_retrieval_metric_failure_codes": retrieval_failure_codes,
        "infrastructure_failure_codes": infrastructure_failure_codes,
    }


def _paired_bootstrap(
    deltas: Sequence[float], *, repeats: int = 2_000, seed: int = 20260813
) -> tuple[float | None, float | None]:
    if not deltas:
        return None, None
    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choice(deltas) for _ in deltas) for _ in range(repeats)
    )
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def compare_response_records(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    bootstrap_repeats: int = 2_000,
    seed: int = 20260813,
) -> dict[str, Any]:
    """Return paired candidate-minus-baseline metric deltas and 95% CIs."""

    baseline_by_id = {str(record["id"]): record for record in baseline}
    candidate_by_id = {str(record["id"]): record for record in candidate}
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("baseline and candidate must contain the same question ids")
    comparison: dict[str, Any] = {}
    metric_fields = (
        *((metric, "metrics") for metric in RESPONSE_METRICS),
        *(
            (metric, "deterministic_retrieval_metrics")
            for metric in DETERMINISTIC_RETRIEVAL_METRICS
        ),
    )
    for metric, field_name in metric_fields:
        paired: list[float] = []
        for sample_id in sorted(baseline_by_id):
            before = (baseline_by_id[sample_id].get(field_name) or {}).get(metric)
            after = (candidate_by_id[sample_id].get(field_name) or {}).get(metric)
            if before is not None and after is not None:
                paired.append(float(after) - float(before))
        low, high = _paired_bootstrap(
            paired, repeats=bootstrap_repeats, seed=seed
        )
        comparison[metric] = {
            "mean_delta": mean(paired) if paired else None,
            "ci95_low": low,
            "ci95_high": high,
            "paired_count": len(paired),
        }
    for scope in LATENCY_SCOPES:
        paired = [
            float(candidate_by_id[sample_id][scope])
            - float(baseline_by_id[sample_id][scope])
            for sample_id in sorted(baseline_by_id)
            if baseline_by_id[sample_id].get(scope) is not None
            and candidate_by_id[sample_id].get(scope) is not None
        ]
        low, high = _paired_bootstrap(
            paired, repeats=bootstrap_repeats, seed=seed
        )
        comparison[scope] = {
            "mean_delta": mean(paired) if paired else None,
            "ci95_low": low,
            "ci95_high": high,
            "paired_count": len(paired),
        }
    return comparison


def build_response_comparison_report(
    *,
    baseline_name: str,
    candidate_name: str,
    baseline_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    minimum_judge_coverage: float = 0.95,
) -> dict[str, Any]:
    systems = {
        baseline_name: aggregate_response_records(
            baseline_records, minimum_judge_coverage=minimum_judge_coverage
        ),
        candidate_name: aggregate_response_records(
            candidate_records, minimum_judge_coverage=minimum_judge_coverage
        ),
    }
    return {
        "schema_version": "engineering-response-comparison/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline_name,
        "candidate": candidate_name,
        "metadata": dict(metadata),
        "systems": systems,
        "paired_deltas": compare_response_records(
            baseline_records, candidate_records
        ),
        "artifact_complete": all(
            system["artifact_complete"] for system in systems.values()
        ),
        "response_metrics_publishable": all(
            system["response_metrics_publishable"] for system in systems.values()
        ),
        "publishable": all(
            system["response_metrics_publishable"] for system in systems.values()
        ),
        "interpretation_boundaries": [
            "Deterministic retrieval metrics use reviewed evidence sources and include answerable questions only.",
            "source_option_recall is diagnostic coverage of alternative acceptable locators; required_claim_recall_at_k is the primary completeness metric.",
            "RAGAS scores are model-judged estimates, not absolute ground truth.",
            "Using a DeepSeek-family judge for DeepSeek-family answers can introduce self-evaluation bias.",
            "P99 is exploratory on small offline samples and is not a production SLA.",
            "A private holdout may be used only once after the candidate is frozen.",
        ],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def render_response_markdown(report: Mapping[str, Any]) -> str:
    baseline = str(report["baseline"])
    candidate = str(report["candidate"])
    lines = [
        "# Engineering RAG response evaluation",
        "",
        f"- Baseline: `{baseline}`",
        f"- Candidate: `{candidate}`",
        f"- Publishable: `{str(bool(report['publishable'])).lower()}`",
        "",
        "## Deterministic retrieval (answerable questions only)",
        "",
        "| Metric | Baseline | Candidate | Delta | 95% paired bootstrap CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric in DETERMINISTIC_RETRIEVAL_METRICS:
        before = report["systems"][baseline]["deterministic_retrieval_metrics"][
            metric
        ]["mean"]
        after = report["systems"][candidate]["deterministic_retrieval_metrics"][
            metric
        ]["mean"]
        delta = report["paired_deltas"][metric]
        ci = f"[{_fmt(delta['ci95_low'])}, {_fmt(delta['ci95_high'])}]"
        lines.append(
            f"| {metric} | {_fmt(before)} | {_fmt(after)} | "
            f"{_fmt(delta['mean_delta'])} | {ci} |"
        )
    lines.extend(
        [
            "",
            "## RAGAS response and context metrics",
            "",
            "| Metric | Baseline | Candidate | Delta | 95% paired bootstrap CI |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric in RESPONSE_METRICS:
        before = report["systems"][baseline]["metrics"][metric]["mean"]
        after = report["systems"][candidate]["metrics"][metric]["mean"]
        delta = report["paired_deltas"][metric]
        ci = f"[{_fmt(delta['ci95_low'])}, {_fmt(delta['ci95_high'])}]"
        lines.append(
            f"| {metric} | {_fmt(before)} | {_fmt(after)} | "
            f"{_fmt(delta['mean_delta'])} | {ci} |"
        )
    lines.extend(["", "## Safety and coverage", ""])
    for name in (baseline, candidate):
        system = report["systems"][name]
        lines.append(
            f"- `{name}`: artifact complete={str(bool(system['artifact_complete'])).lower()}, "
            f"run execution coverage={_fmt(system['run_execution_coverage'])}, "
            f"judge execution coverage={_fmt(system['judge_execution_coverage'])}, "
            f"judge score coverage={_fmt(system['judge_score_coverage'])}, "
            f"answerable response coverage={_fmt(system['answerable_response_coverage'])}, "
            f"deterministic retrieval coverage={_fmt(system['deterministic_retrieval_coverage'])}, "
            f"generation success={_fmt(system['generation_success_rate'])}, "
            f"false refusal={_fmt(system['answerable_refusal_rate'])}, "
            f"refusal F1={_fmt(system['refusal_f1'])}."
        )
    lines.extend(["", "## Generation token usage", ""])
    for name in (baseline, candidate):
        usage = report["systems"][name]["generation_usage"]
        tokens = usage["tokens"]
        lines.append(
            f"- `{name}`: attempted calls={usage['attempted_call_count']}, "
            f"usage reported={usage['reported_usage_count']} "
            f"(coverage={_fmt(usage['coverage'])}), "
            f"prompt={_fmt(tokens['prompt_tokens']['total'], 0)}, "
            f"completion={_fmt(tokens['completion_tokens']['total'], 0)}, "
            f"total={_fmt(tokens['total_tokens']['total'], 0)}, "
            f"cache hit={_fmt(tokens['prompt_cache_hit_tokens']['total'], 0)}, "
            f"cache miss={_fmt(tokens['prompt_cache_miss_tokens']['total'], 0)}."
        )
    lines.extend(["", "## Interpretation boundaries", ""])
    lines.extend(
        f"- {boundary}" for boundary in report["interpretation_boundaries"]
    )
    lines.append("")
    return "\n".join(lines)


def write_response_report(
    report: Mapping[str, Any], output_prefix: str | Path, *, plot: bool = True
) -> tuple[Path, Path, Path | None]:
    prefix = Path(output_prefix)
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    markdown_path = prefix.with_suffix(".md")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_response_markdown(report), encoding="utf-8")
    plot_path = prefix.with_suffix(".png") if plot else None
    if plot_path is not None:
        _write_metric_plot(report, plot_path)
    return json_path, markdown_path, plot_path


def _write_metric_plot(report: Mapping[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    baseline = str(report["baseline"])
    candidate = str(report["candidate"])
    width = 0.38
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    groups = (
        (
            axes[0],
            DETERMINISTIC_RETRIEVAL_METRICS,
            "deterministic_retrieval_metrics",
            "Deterministic retrieval (answerable only)",
        ),
        (axes[1], RESPONSE_METRICS, "metrics", "RAGAS response/context"),
    )
    for axis, metric_names, field_name, title in groups:
        before = [
            report["systems"][baseline][field_name][key]["mean"]
            for key in metric_names
        ]
        after = [
            report["systems"][candidate][field_name][key]["mean"]
            for key in metric_names
        ]
        x = list(range(len(metric_names)))
        axis.bar(
            [value - width / 2 for value in x],
            [value if value is not None else 0 for value in before],
            width,
            label=baseline,
        )
        axis.bar(
            [value + width / 2 for value in x],
            [value if value is not None else 0 for value in after],
            width,
            label=candidate,
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel("Score")
        axis.set_title(title)
        axis.set_xticks(x, [label.replace("_", "\n") for label in metric_names])
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
