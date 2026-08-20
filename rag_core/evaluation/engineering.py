"""Offline evaluation for the engineering-knowledge RAG.

This module deliberately evaluates retrieval and routing without invoking an
LLM or the network.  A strategy is represented by a small predictor callable,
which makes the same runner usable for BM25, dense and hybrid ablations while
remaining easy to test with deterministic fakes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from statistics import mean
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit, urlunsplit

from rag_core.retrieval.engineering.models import EngineeringSearchResult


_IDENTIFIER_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\b"
)
_ROUTES = {"design", "implementation", "official", "comparison", "out_of_scope"}
_DATASET_ROLES = {"development_regression", "holdout"}
_REFUSAL_REASONS = {
    "dynamic_external_state_not_indexed",
    "missing_empirical_evidence",
    "future_commitment_not_documented",
    "missing_production_telemetry",
    "missing_independent_verification",
    "underspecified_dynamic_cost",
    "unverifiable_private_information",
}


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    """One frozen engineering question and its retrieval ground truth."""

    id: str
    question: str
    category: str
    expected_intent: str
    answerable: bool
    relevant_sources: tuple[str, ...]
    expected_answer_points: tuple[str, ...]
    source_scope: str
    source_revision: str
    relevant_symbols: tuple[str, ...] = ()
    expected_route_label: str = ""
    primary_sources: tuple[str, ...] = ()
    supporting_sources: tuple[str, ...] = ()
    refusal_reason: str | None = None
    label_version: str = "engineering-eval/v1"
    dataset_role: str = "development_regression"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationSample":
        required = {
            "id",
            "question",
            "category",
            "expected_intent",
            "answerable",
            "relevant_sources",
            "expected_answer_points",
            "source_scope",
            "source_revision",
            "relevant_symbols",
            "expected_route",
            "primary_sources",
            "supporting_sources",
            "refusal_reason",
            "label_version",
            "dataset_role",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise ValueError(f"evaluation sample is missing fields: {missing}")
        if value.get("label_version") != "engineering-eval/v2":
            raise ValueError("evaluation samples must use engineering-eval/v2")
        if value.get("dataset_role") not in _DATASET_ROLES:
            raise ValueError("unsupported evaluation dataset_role")
        if type(value.get("answerable")) is not bool:
            raise ValueError("answerable must be a JSON boolean")
        for key in (
            "relevant_sources",
            "primary_sources",
            "supporting_sources",
            "relevant_symbols",
            "expected_answer_points",
        ):
            if not isinstance(value.get(key), list) or not all(
                isinstance(item, str) and item.strip() for item in value[key]
            ):
                raise ValueError(f"{key} must be a list of non-empty strings")
        for key in (
            "id",
            "question",
            "category",
            "expected_intent",
            "source_scope",
            "source_revision",
        ):
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise ValueError(f"{key} must be a non-empty string")

        route = str(value["expected_route"])
        if route not in _ROUTES:
            raise ValueError(f"unsupported expected_route: {route}")
        primary = tuple(item.strip() for item in value["primary_sources"])
        supporting = tuple(item.strip() for item in value["supporting_sources"])
        relevant = tuple(item.strip() for item in value["relevant_sources"])
        symbols = tuple(item.strip() for item in value["relevant_symbols"])
        if len(set(map(normalize_source, relevant))) != len(relevant):
            raise ValueError("relevant_sources contains duplicates")
        if len(set(symbols)) != len(symbols):
            raise ValueError("relevant_symbols contains duplicates")
        if set(map(normalize_source, primary)).intersection(
            map(normalize_source, supporting)
        ):
            raise ValueError("primary_sources and supporting_sources must be disjoint")
        if relevant != (*primary, *supporting):
            raise ValueError(
                "relevant_sources must equal primary_sources + supporting_sources"
            )
        refusal_reason = value.get("refusal_reason")
        if value["answerable"] and not primary:
            raise ValueError("v2 answerable samples require primary_sources")
        if value["answerable"] and refusal_reason is not None:
            raise ValueError("answerable samples cannot declare refusal_reason")
        if not value["answerable"] and (primary or supporting):
            raise ValueError("unanswerable samples cannot declare evidence sources")
        if not value["answerable"] and refusal_reason not in _REFUSAL_REASONS:
            raise ValueError("unanswerable samples require a supported refusal_reason")
        return cls(
            id=str(value["id"]),
            question=str(value["question"]),
            category=str(value["category"]),
            expected_intent=str(value["expected_intent"]),
            answerable=value["answerable"],
            relevant_sources=relevant,
            expected_answer_points=tuple(
                str(item) for item in value["expected_answer_points"]
            ),
            source_scope=str(value["source_scope"]),
            source_revision=str(value["source_revision"]),
            relevant_symbols=symbols,
            expected_route_label=route,
            primary_sources=primary,
            supporting_sources=supporting,
            refusal_reason=(
                str(refusal_reason)
                if refusal_reason is not None
                else None
            ),
            label_version="engineering-eval/v2",
            dataset_role=str(value["dataset_role"]),
        )


@dataclass(frozen=True, slots=True)
class EvaluationPrediction:
    """Normalized output expected from one retrieval/generation strategy."""

    predicted_intent: str
    results: tuple[EngineeringSearchResult, ...] = ()
    refused: bool = False
    latency_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class EngineeringPredictor(Protocol):
    def __call__(
        self, sample: EvaluationSample, top_k: int
    ) -> EvaluationPrediction | Mapping[str, Any]: ...


def load_jsonl(path: str | Path) -> list[EvaluationSample]:
    """Load and validate a UTF-8 JSONL evaluation dataset."""

    dataset_path = Path(path)
    samples: list[EvaluationSample] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {dataset_path}:{line_number}: {exc}") from exc
        sample = EvaluationSample.from_mapping(payload)
        if sample.id in seen_ids:
            raise ValueError(f"duplicate evaluation id: {sample.id}")
        if sample.answerable and not sample.relevant_sources:
            raise ValueError(f"answerable sample {sample.id} has no relevant_sources")
        if not sample.answerable and sample.relevant_sources:
            raise ValueError(f"unanswerable sample {sample.id} must not list sources")
        seen_ids.add(sample.id)
        samples.append(sample)
    if not samples:
        raise ValueError(f"evaluation dataset is empty: {dataset_path}")
    return samples


def load_jsonl_files(paths: Iterable[str | Path]) -> list[EvaluationSample]:
    """Load multiple datasets and reject IDs duplicated across files."""

    samples: list[EvaluationSample] = []
    seen_ids: set[str] = set()
    for path in paths:
        for sample in load_jsonl(path):
            if sample.id in seen_ids:
                raise ValueError(f"duplicate evaluation id across datasets: {sample.id}")
            seen_ids.add(sample.id)
            samples.append(sample)
    if not samples:
        raise ValueError("at least one evaluation dataset is required")
    return samples


def expected_route(sample: EvaluationSample) -> str:
    """Map fine-grained dataset intents onto the source router taxonomy."""

    if sample.expected_route_label:
        return normalize_route(sample.expected_route_label)
    intent = sample.expected_intent.casefold()
    scope = sample.source_scope.casefold()
    # Evidence absence is an answerability label, not a source-routing label.
    # In-scope questions about unknown production metrics should be retrieved
    # and then refused, rather than being counted as router errors.
    if not sample.answerable or "unanswerable" in intent:
        return ""
    if "comparison" in intent:
        return "comparison"
    if scope.startswith("official") or sample.category.startswith("official_"):
        return "official"
    if any(
        marker in intent
        for marker in (
            "architecture",
            "boundary",
            "maturity",
            "differentiate",
            "inventory_current",
        )
    ):
        return "design"
    return "implementation"


def normalize_route(value: Any) -> str:
    """Normalize strings, enums and route objects to one comparable value."""

    if value is None:
        return ""
    if hasattr(value, "intent"):
        value = value.intent
    if hasattr(value, "value"):
        value = value.value
    text = str(value).strip().casefold().replace("-", "_")
    aliases = {
        "internal": "implementation",
        "code": "implementation",
        "project_design": "design",
        "external": "official",
        "spec": "official",
        "abstain": "out_of_scope",
        "unanswerable": "out_of_scope",
    }
    return aliases.get(text, text)


def normalize_source(source: str) -> str:
    """Normalize URLs and file paths while preserving their identity."""

    text = source.strip().replace("\\", "/")
    parsed = urlsplit(text)
    if parsed.scheme.casefold() in {"http", "https"}:
        path = parsed.path.rstrip("/") or "/"
        return urlunsplit(
            (parsed.scheme.casefold(), parsed.netloc.casefold(), path, parsed.query, "")
        )
    while text.startswith("./"):
        text = text[2:]
    parts = [part for part in PurePosixPath(text).parts if part not in {"/", "."}]
    return "/".join(parts).casefold()


def source_matches(expected: str, actual: str) -> bool:
    """Match a repository-relative truth path to relative or absolute output."""

    expected_normalized = normalize_source(expected)
    actual_normalized = normalize_source(actual)
    if expected_normalized == actual_normalized:
        return True
    if expected_normalized.startswith(("http://", "https://")):
        return False
    return actual_normalized.endswith("/" + expected_normalized)


def _report_source(result: EngineeringSearchResult) -> str:
    """Use repository-relative live paths in persisted evaluation reports."""

    relative = result.metadata.get("relative_path")
    if isinstance(relative, str) and relative.strip():
        return relative.strip().replace("\\", "/")
    source = result.source.strip().replace("\\", "/")
    parsed = urlsplit(source)
    if parsed.scheme.casefold() in {"http", "https"}:
        return source
    if re.match(r"^[A-Za-z]:/", source) or source.startswith("/"):
        raise ValueError(
            "evaluation result contains an absolute local source without relative_path"
        )
    return source


def explicit_symbols(sample: EvaluationSample) -> tuple[str, ...]:
    """Return code-like symbols explicitly named by a symbol-tracing question.

    ``relevant_symbols`` is supported as an optional future annotation.  The
    frozen Mini-Nanobot dataset predates that field, so for its symbol category
    we conservatively extract only dotted, underscored or CamelCase names from
    the question.  Generic prose tokens and acronyms such as API are excluded.
    """

    if sample.relevant_symbols:
        return sample.relevant_symbols
    if sample.category != "code_symbol_call_chain":
        return ()
    symbols: list[str] = []
    for token in _IDENTIFIER_RE.findall(sample.question):
        is_dotted = "." in token
        is_snake = "_" in token
        is_camel = (
            any(char.islower() for char in token)
            and any(char.isupper() for char in token[1:])
        )
        if is_dotted or is_snake or is_camel:
            symbols.append(token)
    return tuple(dict.fromkeys(symbols))


def _result_symbols(result: EngineeringSearchResult) -> tuple[str, ...]:
    values: list[str] = []
    if result.symbol:
        values.append(result.symbol)
    for key in ("symbol", "symbol_name", "qualified_name"):
        value = result.metadata.get(key)
        if isinstance(value, str) and value:
            values.append(value)
    return tuple(dict.fromkeys(values))


def _symbol_matches(expected: str, actual: str) -> bool:
    left = expected.strip()
    right = actual.strip()
    return right == left or right.endswith("." + left)


def _coerce_result(value: Any) -> EngineeringSearchResult:
    if isinstance(value, EngineeringSearchResult):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("prediction results must be EngineeringSearchResult or mappings")
    metadata = value.get("metadata") or {}
    return EngineeringSearchResult(
        content=str(value.get("content", value.get("page_content", ""))),
        source=str(value.get("source", metadata.get("source", ""))),
        score=float(value.get("score", 0.0)),
        corpus=str(value.get("corpus", metadata.get("corpus", "internal"))),
        authority=str(value.get("authority", metadata.get("authority", "unknown"))),
        line_start=value.get("line_start", metadata.get("line_start")),
        line_end=value.get("line_end", metadata.get("line_end")),
        symbol=value.get("symbol", metadata.get("symbol_name")),
        retriever=str(value.get("retriever", "")),
        metadata=metadata,
    )


def coerce_prediction(value: EvaluationPrediction | Mapping[str, Any]) -> EvaluationPrediction:
    if isinstance(value, EvaluationPrediction):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("predictor must return EvaluationPrediction or a mapping")
    route = value.get("predicted_intent", value.get("intent", value.get("route", "")))
    return EvaluationPrediction(
        predicted_intent=normalize_route(route),
        results=tuple(_coerce_result(item) for item in value.get("results", ())),
        refused=bool(value.get("refused", value.get("abstained", False))),
        latency_ms=(
            float(value["latency_ms"]) if value.get("latency_ms") is not None else None
        ),
        metadata=value.get("metadata") or {},
    )


def _score_sample(
    sample: EvaluationSample,
    prediction: EvaluationPrediction,
    *,
    top_k: int,
) -> dict[str, Any]:
    raw_results = prediction.results[: max(top_k, top_k * 3)]
    results: list[EngineeringSearchResult] = []
    seen_sources: set[str] = set()
    for result in raw_results:
        identity = normalize_source(result.source)
        if identity in seen_sources:
            continue
        seen_sources.add(identity)
        results.append(result)
        if len(results) >= top_k:
            break
    primary_sources = sample.primary_sources or sample.relevant_sources
    supporting_sources = sample.supporting_sources
    retrieved_sources = tuple(_report_source(result) for result in results)

    matched_sources = [
        source
        for source in primary_sources
        if any(source_matches(source, retrieved) for retrieved in retrieved_sources)
    ]
    file_recall = (
        len(matched_sources) / len(primary_sources) if primary_sources else None
    )
    matched_supporting = [
        source
        for source in supporting_sources
        if any(source_matches(source, retrieved) for retrieved in retrieved_sources)
    ]
    supporting_recall = (
        len(matched_supporting) / len(supporting_sources)
        if supporting_sources
        else None
    )
    reciprocal_rank = 0.0
    hit = None
    if primary_sources:
        hit = bool(matched_sources)
        for rank, result in enumerate(results, start=1):
            if any(source_matches(source, result.source) for source in primary_sources):
                reciprocal_rank = 1.0 / rank
                break

    grades: list[int] = []
    for result in results:
        if any(source_matches(source, result.source) for source in primary_sources):
            grades.append(2)
        elif any(source_matches(source, result.source) for source in supporting_sources):
            grades.append(1)
        else:
            grades.append(0)
    ideal = sorted(
        [2] * len(primary_sources) + [1] * len(supporting_sources), reverse=True
    )[:top_k]
    ndcg = _ndcg(grades, ideal) if primary_sources else None
    citation_precision = (
        sum(grade > 0 for grade in grades) / len(grades) if grades else None
    )

    expected_symbols = explicit_symbols(sample)
    retrieved_symbols = tuple(
        symbol for result in raw_results[:top_k] for symbol in _result_symbols(result)
    )
    matched_symbols = [
        symbol
        for symbol in expected_symbols
        if any(_symbol_matches(symbol, retrieved) for retrieved in retrieved_symbols)
    ]
    symbol_recall = (
        len(matched_symbols) / len(expected_symbols) if expected_symbols else None
    )

    wanted_route = expected_route(sample)
    actual_route = normalize_route(prediction.predicted_intent)
    oracle_route = bool(prediction.metadata.get("oracle_route"))
    live_primary_hit = None
    if wanted_route in {"implementation", "comparison"} and primary_sources:
        live_primary_hit = any(
            result.metadata.get("live_verification")
            and any(source_matches(source, result.source) for source in primary_sources)
            for result in raw_results[:top_k]
        )
    return {
        "id": sample.id,
        "category": sample.category,
        "answerable": sample.answerable,
        "expected_route": wanted_route,
        "predicted_route": actual_route,
        "routing_correct": (
            None if oracle_route or not wanted_route else actual_route == wanted_route
        ),
        "file_recall_at_k": file_recall,
        "primary_recall_at_k": file_recall,
        "hit_rate_at_k": hit,
        "primary_hit_at_k": hit,
        "supporting_recall_at_k": supporting_recall,
        "ndcg_at_k": ndcg,
        "citation_precision_at_k": citation_precision,
        "reciprocal_rank": reciprocal_rank if primary_sources else None,
        "symbol_recall_at_k": symbol_recall,
        "expected_sources": list(primary_sources),
        "primary_sources": list(primary_sources),
        "supporting_sources": list(supporting_sources),
        "retrieved_sources": list(retrieved_sources),
        "matched_sources": matched_sources,
        "matched_supporting_sources": matched_supporting,
        "expected_symbols": list(expected_symbols),
        "retrieved_symbols": list(retrieved_symbols),
        "matched_symbols": matched_symbols,
        "refused": prediction.refused,
        "expected_refusal_reason": sample.refusal_reason,
        "predicted_refusal_reason": prediction.metadata.get("refusal_reason"),
        "refusal_reason_correct": (
            prediction.refused
            and sample.refusal_reason is not None
            and prediction.metadata.get("refusal_reason") == sample.refusal_reason
        ) if not sample.answerable else None,
        "live_verification_attempted": prediction.metadata.get(
            "live_verification_attempted"
        ),
        "live_revision": prediction.metadata.get("live_revision"),
        "live_primary_hit_at_k": live_primary_hit,
        "latency_ms": prediction.latency_ms,
    }


def _ndcg(grades: Sequence[int], ideal: Sequence[int]) -> float:
    def dcg(values: Sequence[int]) -> float:
        return sum(
            (2**relevance - 1) / math.log2(rank + 1)
            for rank, relevance in enumerate(values, start=1)
        )

    denominator = dcg(ideal)
    return dcg(grades) / denominator if denominator else 0.0


def evaluate_strategy(
    samples: Sequence[EvaluationSample],
    predictor: EngineeringPredictor,
    *,
    strategy_name: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate one strategy and return aggregate plus per-question metrics."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    details: list[dict[str, Any]] = []
    for sample in samples:
        started = time.perf_counter()
        prediction = coerce_prediction(predictor(sample, top_k))
        if prediction.latency_ms is None:
            prediction = EvaluationPrediction(
                predicted_intent=prediction.predicted_intent,
                results=prediction.results,
                refused=prediction.refused,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                metadata=prediction.metadata,
            )
        details.append(_score_sample(sample, prediction, top_k=top_k))

    def average_present(key: str) -> float:
        values = [float(row[key]) for row in details if row[key] is not None]
        return mean(values) if values else 0.0

    unanswerable = [row for row in details if not row["answerable"]]
    answerable = [row for row in details if row["answerable"]]
    refusal_tp = sum(row["refused"] for row in unanswerable)
    refusal_fp = sum(row["refused"] for row in answerable)
    refusal_fn = len(unanswerable) - refusal_tp
    correct_refusal_reasons = sum(
        bool(row["refusal_reason_correct"]) for row in unanswerable
    )
    refusal_precision = (
        refusal_tp / (refusal_tp + refusal_fp) if refusal_tp + refusal_fp else 0.0
    )
    refusal_recall = refusal_tp / len(unanswerable) if unanswerable else 0.0
    refusal_f1 = (
        2 * refusal_precision * refusal_recall / (refusal_precision + refusal_recall)
        if refusal_precision + refusal_recall
        else 0.0
    )
    latencies = sorted(
        float(row["latency_ms"])
        for row in details
        if row["latency_ms"] is not None
    )
    routing_rows = [row for row in details if row["routing_correct"] is not None]
    metrics = {
        "routing_accuracy": (
            mean(float(row["routing_correct"]) for row in routing_rows)
            if routing_rows
            else None
        ),
        "routing_macro_f1": (
            _macro_f1(routing_rows, "expected_route", "predicted_route")
            if routing_rows
            else None
        ),
        "file_recall_at_k": average_present("file_recall_at_k"),
        "primary_recall_at_k": average_present("primary_recall_at_k"),
        "hit_rate_at_k": average_present("hit_rate_at_k"),
        "primary_hit_at_k": average_present("primary_hit_at_k"),
        "supporting_recall_at_k": average_present("supporting_recall_at_k"),
        "ndcg_at_k": average_present("ndcg_at_k"),
        "citation_precision_at_k": average_present("citation_precision_at_k"),
        "symbol_recall_at_k": average_present("symbol_recall_at_k"),
        "live_primary_hit_at_k": average_present("live_primary_hit_at_k"),
        "mrr": average_present("reciprocal_rank"),
        "unanswerable_refusal_rate": (
            mean(float(row["refused"]) for row in unanswerable) if unanswerable else 0.0
        ),
        "answerable_refusal_rate": (
            mean(float(row["refused"]) for row in answerable) if answerable else 0.0
        ),
        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,
        "refusal_f1": refusal_f1,
        "refusal_reason_accuracy": (
            correct_refusal_reasons / refusal_tp if refusal_tp else 0.0
        ),
        "strict_refusal_reason_recall": (
            correct_refusal_reasons / len(unanswerable) if unanswerable else 0.0
        ),
        "mean_latency_ms": average_present("latency_ms"),
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        # P99 is retained for exploratory engineering diagnostics.  Small
        # offline question sets do not provide enough tail samples to treat
        # it as a production SLA estimate; the report states that boundary.
        "p99_latency_ms": _percentile(latencies, 0.99),
        "question_count": len(details),
        "routing_eligible_questions": sum(
            row["routing_correct"] is not None for row in details
        ),
        "file_eligible_questions": sum(row["file_recall_at_k"] is not None for row in details),
        "symbol_eligible_questions": sum(
            row["symbol_recall_at_k"] is not None for row in details
        ),
        "unanswerable_questions": len(unanswerable),
    }
    return {"strategy": strategy_name, "top_k": top_k, "metrics": metrics, "details": details}


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _macro_f1(
    rows: Sequence[Mapping[str, Any]], expected_key: str, predicted_key: str
) -> float:
    labels = sorted(
        {
            str(row[expected_key])
            for row in rows
            if row.get(expected_key) not in {None, ""}
        }
    )
    scores: list[float] = []
    for label in labels:
        tp = sum(
            row.get(expected_key) == label and row.get(predicted_key) == label
            for row in rows
        )
        fp = sum(
            row.get(expected_key) != label and row.get(predicted_key) == label
            for row in rows
        )
        fn = sum(
            row.get(expected_key) == label and row.get(predicted_key) != label
            for row in rows
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return mean(scores) if scores else 0.0


def run_ablation(
    samples: Sequence[EvaluationSample],
    predictors: Mapping[str, EngineeringPredictor],
    *,
    top_k: int = 5,
    dataset_name: str = "engineering-eval",
    baseline: str = "bm25",
    suite: str = "e2e",
    run_metadata: Mapping[str, Any] | None = None,
    validate_contract: bool = False,
    warmup: bool = False,
) -> dict[str, Any]:
    """Run BM25/dense/hybrid (or any named) strategies under one contract."""

    if not predictors:
        raise ValueError("at least one predictor is required")
    if not samples:
        raise ValueError("at least one evaluation sample is required")
    strategy_metadata = {
        name: dict(getattr(predictor, "evaluation_metadata", {}) or {})
        for name, predictor in predictors.items()
    }
    if validate_contract:
        _validate_suite_contract(suite, strategy_metadata)
    if baseline not in predictors:
        raise ValueError(f"baseline strategy is not present: {baseline}")
    if warmup:
        warmup_sample = next(
            (sample for sample in samples if sample.answerable), samples[0]
        )
        for predictor in predictors.values():
            predictor(warmup_sample, top_k)
    strategies = {
        name: evaluate_strategy(
            samples, predictor, strategy_name=name, top_k=top_k
        )
        for name, predictor in predictors.items()
    }
    baseline_metrics = strategies[baseline]["metrics"]
    delta_keys = (
        "routing_accuracy",
        "routing_macro_f1",
        "primary_recall_at_k",
        "primary_hit_at_k",
        "supporting_recall_at_k",
        "ndcg_at_k",
        "citation_precision_at_k",
        "symbol_recall_at_k",
        "mrr",
        "refusal_f1",
        "answerable_refusal_rate",
        "mean_latency_ms",
    )
    deltas = {
        name: {
            key: (
                float(result["metrics"][key]) - float(baseline_metrics[key])
                if result["metrics"][key] is not None
                and baseline_metrics[key] is not None
                else None
            )
            for key in delta_keys
        }
        for name, result in strategies.items()
    }
    return {
        "schema_version": "engineering-eval/v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "question_count": len(samples),
        "top_k": top_k,
        "baseline": baseline,
        "suite": suite,
        "run_metadata": {
            **dict(run_metadata or {}),
            "latency_scope": "warm_requests" if warmup else "uncontrolled",
        },
        "strategy_metadata": strategy_metadata,
        "strategies": strategies,
        "deltas_from_baseline": deltas,
    }


def _validate_suite_contract(
    suite: str, strategy_metadata: Mapping[str, Mapping[str, Any]]
) -> None:
    if suite not in {"index", "e2e"}:
        raise ValueError(f"unsupported evaluation suite: {suite}")
    build_ids: set[str] = set()
    embedding_models: set[str] = set()
    candidate_contracts: set[tuple[int, int, int]] = set()
    for name, metadata in strategy_metadata.items():
        if not metadata:
            raise ValueError(f"strategy {name} has no evaluation contract metadata")
        if suite == "index":
            expected = {
                "oracle_route": True,
                "live_verification": False,
                "answer_generation": False,
                "fail_closed": True,
            }
        else:
            expected = {
                "oracle_route": False,
                "live_verification": True,
                "answer_generation": True,
                "fail_closed": True,
            }
        mismatched = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) is not value
        }
        if mismatched:
            raise ValueError(
                f"strategy {name} is incompatible with {suite} suite: {mismatched}"
            )
        build_id = str(metadata.get("build_id") or "").strip()
        embedding_model = str(metadata.get("embedding_model") or "").strip()
        if not build_id:
            raise ValueError(f"strategy {name} has no build_id")
        if not embedding_model:
            raise ValueError(f"strategy {name} has no embedding_model")
        build_ids.add(build_id)
        embedding_models.add(embedding_model)
        if suite == "index":
            strategy = str(metadata.get("retrieval_strategy") or "").strip()
            if strategy != name:
                raise ValueError(
                    f"strategy {name} has mismatched retrieval_strategy: {strategy!r}"
                )
            try:
                candidate_contract = tuple(
                    int(metadata[key])
                    for key in (
                        "candidate_budget_multiplier",
                        "partition_candidate_multiplier",
                        "federated_candidate_multiplier",
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"strategy {name} has an invalid candidate-budget contract"
                ) from exc
            if any(value <= 0 for value in candidate_contract):
                raise ValueError(
                    f"strategy {name} has a non-positive candidate-budget contract"
                )
            candidate_contracts.add(candidate_contract)
    if len(build_ids) != 1:
        raise ValueError("all formal strategies must share one non-empty build_id")
    if len(embedding_models) != 1:
        raise ValueError(
            "all formal strategies must share one non-empty embedding_model"
        )
    if suite == "index" and len(candidate_contracts) != 1:
        raise ValueError("all index strategies must share one candidate-budget contract")


def _format_metric(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact, interview-friendly ablation table."""

    top_k = int(report["top_k"])
    lines = [
        "# Engineering RAG evaluation",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Questions: {report['question_count']}",
        f"- Top-K: {top_k}",
        f"- Baseline: `{report['baseline']}`",
        f"- Suite: `{report.get('suite', 'e2e')}`",
        "",
        "| Strategy | Route acc. | Route macro-F1 | Primary Hit@K | Primary Recall@K | nDCG@K | Symbol Recall@K | MRR | Refusal F1 | False refusal | P50 ms | P95 ms | P99 ms* |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in report["strategies"].items():
        metrics = result["metrics"]
        lines.append(
            "| {name} | {routing} | {route_f1} | {hit} | {primary} | "
            "{ndcg} | {symbol} | {mrr} | {refusal} | "
            "{false_refusal} | {p50} | {p95} | {p99} |".format(
                name=name,
                routing=_format_metric(metrics["routing_accuracy"]),
                route_f1=_format_metric(metrics["routing_macro_f1"]),
                hit=_format_metric(metrics["primary_hit_at_k"]),
                primary=_format_metric(metrics["primary_recall_at_k"]),
                ndcg=_format_metric(metrics["ndcg_at_k"]),
                symbol=_format_metric(metrics["symbol_recall_at_k"]),
                mrr=_format_metric(metrics["mrr"]),
                refusal=_format_metric(metrics["refusal_f1"]),
                false_refusal=_format_metric(metrics["answerable_refusal_rate"]),
                p50=_format_metric(metrics["p50_latency_ms"], digits=2),
                p95=_format_metric(metrics["p95_latency_ms"], digits=2),
                p99=_format_metric(metrics["p99_latency_ms"], digits=2),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- Source metrics use explicit v2 primary/supporting labels and de-duplicate repeated chunks from the same source.",
            "- Index ablation uses oracle routes, answerable questions only, and disables live rg/AST/Git plus answer generation.",
            "- End-to-end evaluation fixes the hybrid index and evaluates routing, live verification, citations, and refusal separately.",
            "- Symbol Recall@K uses explicit `relevant_symbols`; inspect the eligible sample count.",
            "- Refusal is an explicit predictor decision. An empty retrieval result is not silently counted as a safe refusal.",
            "- File/citation metrics are source-level, not passage entailment or answer correctness metrics.",
            "- Latency is measured after one discarded warm-up call per strategy; model/index loading and report serialization are excluded.",
            "- `P99 ms*` is exploratory for these small offline datasets; it is not a statistically stable production tail-latency SLA.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], output_prefix: str | Path) -> tuple[Path, Path]:
    """Write sibling JSON and Markdown reports atomically enough for local use."""

    prefix = Path(output_prefix)
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    markdown_path = prefix.with_suffix(".md")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _load_predictors(factory_spec: str) -> Mapping[str, EngineeringPredictor]:
    module_name, separator, attribute = factory_spec.partition(":")
    if not separator:
        raise ValueError("factory must use module.path:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    predictors = factory()
    if not isinstance(predictors, Mapping):
        raise TypeError("predictor factory must return a mapping of strategy names to callables")
    return predictors


def _validate_evaluation_snapshot(
    snapshot_path: str | Path,
    dataset_paths: Sequence[Path],
    strategy_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    path = Path(snapshot_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "engineering-eval-snapshot/v1":
        raise ValueError("unsupported engineering evaluation snapshot schema")
    manifest_build_id = str(payload.get("manifest_build_id") or "")
    build_ids = {
        str(metadata.get("build_id") or "")
        for metadata in strategy_metadata.values()
    }
    if build_ids != {manifest_build_id}:
        raise RuntimeError(
            "evaluation snapshot/index build mismatch: "
            f"snapshot={manifest_build_id}, predictors={sorted(build_ids)}"
        )
    entries = {
        str(item.get("name")): item
        for item in payload.get("datasets", [])
        if isinstance(item, Mapping)
    }
    for dataset in dataset_paths:
        entry = entries.get(dataset.name)
        actual_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
        if entry is None or entry.get("sha256") != actual_hash:
            raise RuntimeError(
                f"evaluation dataset is not frozen in {path.name}: {dataset.name}"
            )
    payload["snapshot_path"] = path.name
    payload["snapshot_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        nargs="+",
        help="one or more UTF-8 JSONL datasets",
    )
    parser.add_argument(
        "--factory",
        default=None,
        help="module.path:callable returning {'bm25': predictor, ...}",
    )
    parser.add_argument("--output", required=True, help="report path prefix")
    parser.add_argument(
        "--snapshot", default="data/eval/evaluation_snapshot.json"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--suite", choices=("index", "e2e"), default="index")
    args = parser.parse_args(argv)

    samples = load_jsonl_files(args.dataset)
    if args.suite == "index":
        samples = [sample for sample in samples if sample.answerable]
    factory = args.factory or (
        "rag_core.evaluation.engineering_adapter:create_default_predictors"
        if args.suite == "index"
        else "rag_core.evaluation.engineering_adapter:create_default_e2e_predictors"
    )
    predictors = _load_predictors(factory)
    baseline = args.baseline or (
        "bm25" if args.suite == "index" else next(iter(predictors))
    )
    dataset_files = [Path(path).resolve() for path in args.dataset]
    strategy_metadata = {
        name: dict(getattr(predictor, "evaluation_metadata", {}) or {})
        for name, predictor in predictors.items()
    }
    snapshot = _validate_evaluation_snapshot(
        args.snapshot, dataset_files, strategy_metadata
    )
    run_metadata = {
        "factory": factory,
        "dataset_files": [
            {
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in dataset_files
        ],
        "dataset_roles": sorted({sample.dataset_role for sample in samples}),
        "label_versions": sorted({sample.label_version for sample in samples}),
        "evaluation_snapshot": snapshot,
    }
    report = run_ablation(
        samples,
        predictors,
        top_k=args.top_k,
        dataset_name=" + ".join(Path(path).name for path in args.dataset),
        baseline=baseline,
        suite=args.suite,
        run_metadata=run_metadata,
        validate_contract=True,
        warmup=True,
    )
    json_path, markdown_path = write_report(report, args.output)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via the public API in tests
    raise SystemExit(main())
