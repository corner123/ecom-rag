"""Strict, explicit RAGAS evaluation helpers.

This module intentionally does not rely on RAGAS' provider defaults.  A real
evaluation must receive both an LLM wrapper and an embeddings wrapper (or use
``from_langchain`` to build those wrappers explicitly).  Metric failures are
reported as missing scores with error metadata; they are never converted to
zero, because zero is a valid metric value and would hide an invalid run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import fmean
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


class RAGASEvaluationConfigurationError(RuntimeError):
    """Raised when an evaluation is missing an explicit model dependency."""


class RAGASSampleValidationError(ValueError):
    """Raised when a sample cannot be represented as a RAGAS single turn."""


class RAGASEvaluationError(RuntimeError):
    """Raised by legacy convenience methods when any metric failed.

    The structured report remains available on ``report`` so callers can
    inspect the exact metric and exception instead of losing that information.
    """

    def __init__(self, message: str, report: "RAGASEvaluationReport") -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class RAGASSample:
    """Provider-neutral representation of one RAGAS single-turn sample."""

    user_input: str
    retrieved_contexts: List[str]
    response: str
    reference: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RAGASSample":
        """Load current field names, with safe aliases for the old evaluator."""

        user_input = value.get("user_input", value.get("question"))
        retrieved_contexts = value.get("retrieved_contexts", value.get("contexts"))
        response = value.get("response", value.get("answer"))
        reference = value.get("reference", value.get("ground_truth"))

        string_fields = {
            "user_input": user_input,
            "response": response,
            "reference": reference,
        }
        for name, field_value in string_fields.items():
            if not isinstance(field_value, str) or not field_value.strip():
                raise RAGASSampleValidationError(
                    f"{name} must be a non-empty string"
                )

        if not isinstance(retrieved_contexts, list) or not all(
            isinstance(context, str) for context in retrieved_contexts
        ):
            raise RAGASSampleValidationError(
                "retrieved_contexts must be a list of strings"
            )

        return cls(
            user_input=user_input,
            retrieved_contexts=list(retrieved_contexts),
            response=response,
            reference=reference,
        )


@dataclass(frozen=True)
class RAGASMetricObservation:
    score: Optional[float]
    status: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class RAGASMetricSummary:
    mean: Optional[float]
    successful: int
    failed: int
    coverage: float


@dataclass(frozen=True)
class RAGASSampleResult:
    sample_index: int
    user_input: str
    metrics: Dict[str, RAGASMetricObservation]


@dataclass(frozen=True)
class RAGASEvaluationReport:
    sample_count: int
    metric_count: int
    overall_coverage: float
    summaries: Dict[str, RAGASMetricSummary]
    samples: List[RAGASSampleResult]

    @property
    def failure_count(self) -> int:
        return sum(summary.failed for summary in self.summaries.values())

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the report."""

        return asdict(self)


class RAGASEvaluator:
    """Evaluate RAG outputs with explicitly configured RAGAS dependencies.

    ``metrics`` and ``evaluate_fn`` are injectable so unit tests and offline
    contract checks can exercise the complete adapter without making network
    requests.  Production callers normally use ``from_langchain``.
    """

    def __init__(
        self,
        llm: Any = None,
        embeddings: Any = None,
        *,
        llm_wrapper: Any = None,
        embeddings_wrapper: Any = None,
        metrics: Optional[Mapping[str, Any]] = None,
        include_answer_correctness: bool = False,
        evaluate_fn: Optional[Callable[..., Any]] = None,
        run_config: Any = None,
    ) -> None:
        if llm is not None and llm_wrapper is not None:
            raise RAGASEvaluationConfigurationError(
                "pass llm or llm_wrapper, not both"
            )
        if embeddings is not None and embeddings_wrapper is not None:
            raise RAGASEvaluationConfigurationError(
                "pass embeddings or embeddings_wrapper, not both"
            )

        # ``llm`` remains accepted for the old RAGEngine construction path, but
        # is wrapped explicitly here rather than delegated to RAGAS defaults.
        if llm is not None:
            llm_wrapper = self._wrap_langchain_llm(llm, run_config=run_config)
        if embeddings is not None:
            embeddings_wrapper = self._wrap_langchain_embeddings(
                embeddings, run_config=run_config
            )

        self.llm_wrapper = llm_wrapper
        self.embeddings_wrapper = embeddings_wrapper
        self._metrics = dict(metrics) if metrics is not None else None
        self.include_answer_correctness = include_answer_correctness
        self._evaluate_fn = evaluate_fn
        self.run_config = run_config

    @classmethod
    def from_langchain(
        cls,
        *,
        llm: Any,
        embeddings: Any,
        include_answer_correctness: bool = False,
        evaluate_fn: Optional[Callable[..., Any]] = None,
        run_config: Any = None,
    ) -> "RAGASEvaluator":
        """Build explicit RAGAS wrappers around LangChain dependencies."""

        return cls(
            llm=llm,
            embeddings=embeddings,
            include_answer_correctness=include_answer_correctness,
            evaluate_fn=evaluate_fn,
            run_config=run_config,
        )

    @staticmethod
    def _wrap_langchain_llm(llm: Any, *, run_config: Any) -> Any:
        try:
            from ragas.llms import LangchainLLMWrapper
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RAGASEvaluationConfigurationError(
                "ragas is required to wrap the evaluation LLM"
            ) from exc
        return LangchainLLMWrapper(llm, run_config=run_config)

    @staticmethod
    def _wrap_langchain_embeddings(embeddings: Any, *, run_config: Any) -> Any:
        try:
            from ragas.embeddings import LangchainEmbeddingsWrapper
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RAGASEvaluationConfigurationError(
                "ragas is required to wrap evaluation embeddings"
            ) from exc
        return LangchainEmbeddingsWrapper(embeddings, run_config=run_config)

    def _resolve_metrics(self) -> Dict[str, Any]:
        if self._metrics is not None:
            if not self._metrics:
                raise RAGASEvaluationConfigurationError(
                    "at least one RAGAS metric is required"
                )
            if self._evaluate_fn is None and (
                self.llm_wrapper is None or self.embeddings_wrapper is None
            ):
                raise RAGASEvaluationConfigurationError(
                    "real RAGAS evaluation requires explicit llm_wrapper and "
                    "embeddings_wrapper, including with custom metrics"
                )
            return dict(self._metrics)

        if self.llm_wrapper is None:
            raise RAGASEvaluationConfigurationError(
                "an explicit llm_wrapper is required for RAGAS evaluation"
            )
        if self.embeddings_wrapper is None:
            raise RAGASEvaluationConfigurationError(
                "an explicit embeddings_wrapper is required for RAGAS evaluation"
            )

        try:
            from ragas.metrics import (
                AnswerCorrectness,
                Faithfulness,
                LLMContextPrecisionWithReference,
                LLMContextRecall,
                ResponseRelevancy,
            )
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RAGASEvaluationConfigurationError(
                "ragas 0.2.x metrics are not installed"
            ) from exc

        metrics: Dict[str, Any] = {
            "faithfulness": Faithfulness(llm=self.llm_wrapper),
            "answer_relevancy": ResponseRelevancy(
                llm=self.llm_wrapper,
                embeddings=self.embeddings_wrapper,
            ),
            "context_precision": LLMContextPrecisionWithReference(
                llm=self.llm_wrapper
            ),
            "context_recall": LLMContextRecall(llm=self.llm_wrapper),
        }
        if self.include_answer_correctness:
            metrics["answer_correctness"] = AnswerCorrectness(
                llm=self.llm_wrapper,
                embeddings=self.embeddings_wrapper,
            )
        return metrics

    def _resolve_evaluate_fn(self) -> Callable[..., Any]:
        if self._evaluate_fn is not None:
            return self._evaluate_fn
        try:
            from ragas import evaluate
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RAGASEvaluationConfigurationError(
                "ragas is required for evaluation"
            ) from exc
        return evaluate

    @staticmethod
    def _to_ragas_dataset(sample: RAGASSample) -> Any:
        try:
            from ragas import EvaluationDataset, SingleTurnSample
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RAGASEvaluationConfigurationError(
                "ragas 0.2.x is required to build EvaluationDataset"
            ) from exc

        ragas_sample = SingleTurnSample(
            user_input=sample.user_input,
            retrieved_contexts=sample.retrieved_contexts,
            response=sample.response,
            reference=sample.reference,
        )
        return EvaluationDataset(samples=[ragas_sample])

    @staticmethod
    def _extract_metric_score(result: Any, metric_name: str) -> float:
        try:
            values = result[metric_name]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"RAGAS result does not contain metric {metric_name!r}"
            ) from exc

        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            if len(values) != 1:
                raise ValueError(
                    f"expected one {metric_name!r} score, received {len(values)}"
                )
            raw_score = values[0]
        else:
            raw_score = values

        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"metric {metric_name!r} returned a non-numeric score"
            ) from exc

        if not math.isfinite(score):
            raise ValueError(f"metric {metric_name!r} returned a non-finite score")
        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"metric {metric_name!r} returned {score}; expected [0, 1]"
            )
        return score

    @staticmethod
    def _error_observation(exc: Exception) -> RAGASMetricObservation:
        # Keep reports useful without allowing an unbounded provider response
        # to bloat the persisted evaluation artifact.
        message = str(exc).strip() or repr(exc)
        return RAGASMetricObservation(
            score=None,
            status="error",
            error_type=type(exc).__name__,
            error_message=message[:500],
        )

    def evaluate_samples(
        self,
        samples: Iterable[RAGASSample | Mapping[str, Any]],
        *,
        metric_names: Optional[Iterable[str]] = None,
    ) -> RAGASEvaluationReport:
        """Run strict per-sample, per-metric evaluation.

        Runtime metric failures are captured as explicit ``null`` scores with
        error metadata.  Configuration and sample validation errors are raised
        immediately because no meaningful evaluation can be attempted.
        """

        normalised_samples = [
            sample
            if isinstance(sample, RAGASSample)
            else RAGASSample.from_mapping(sample)
            for sample in samples
        ]
        if not normalised_samples:
            raise RAGASSampleValidationError("at least one sample is required")

        metrics = self._resolve_metrics()
        if metric_names is not None:
            requested = tuple(dict.fromkeys(str(name) for name in metric_names))
            if not requested:
                raise RAGASEvaluationConfigurationError(
                    "metric_names must contain at least one metric"
                )
            unknown = [name for name in requested if name not in metrics]
            if unknown:
                raise RAGASEvaluationConfigurationError(
                    "unknown RAGAS metric names: " + ", ".join(unknown)
                )
            metrics = {name: metrics[name] for name in requested}
        evaluate_fn = self._resolve_evaluate_fn()
        results: List[RAGASSampleResult] = []

        for sample_index, sample in enumerate(normalised_samples):
            metric_results: Dict[str, RAGASMetricObservation] = {}
            dataset = self._to_ragas_dataset(sample)

            for public_name, metric in metrics.items():
                ragas_name = getattr(metric, "name", public_name)
                try:
                    raw_result = evaluate_fn(
                        dataset=dataset,
                        metrics=[metric],
                        llm=self.llm_wrapper,
                        embeddings=self.embeddings_wrapper,
                        run_config=self.run_config,
                        raise_exceptions=True,
                        show_progress=False,
                        batch_size=1,
                    )
                    score = self._extract_metric_score(raw_result, ragas_name)
                    metric_results[public_name] = RAGASMetricObservation(
                        score=score,
                        status="ok",
                    )
                except Exception as exc:  # error is preserved in the report
                    metric_results[public_name] = self._error_observation(exc)

            results.append(
                RAGASSampleResult(
                    sample_index=sample_index,
                    user_input=sample.user_input,
                    metrics=metric_results,
                )
            )

        summaries: Dict[str, RAGASMetricSummary] = {}
        successful_total = 0
        for metric_name in metrics:
            observations = [result.metrics[metric_name] for result in results]
            scores = [
                observation.score
                for observation in observations
                if observation.score is not None
            ]
            successful = len(scores)
            failed = len(observations) - successful
            successful_total += successful
            summaries[metric_name] = RAGASMetricSummary(
                mean=fmean(scores) if scores else None,
                successful=successful,
                failed=failed,
                coverage=successful / len(observations),
            )

        possible_total = len(results) * len(metrics)
        return RAGASEvaluationReport(
            sample_count=len(results),
            metric_count=len(metrics),
            overall_coverage=successful_total / possible_total,
            summaries=summaries,
            samples=results,
        )

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> Dict[str, float]:
        """Legacy success-only view; raise if any score is unavailable."""

        report = self.evaluate_samples(
            [
                RAGASSample(
                    user_input=question,
                    retrieved_contexts=contexts,
                    response=answer,
                    reference=ground_truth,
                )
            ]
        )
        if report.failure_count:
            raise RAGASEvaluationError(
                "one or more RAGAS metrics failed; inspect exception.report",
                report,
            )
        return {
            metric_name: float(summary.mean)
            for metric_name, summary in report.summaries.items()
            if summary.mean is not None
        }

    def evaluate_batch(self, eval_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Legacy mean-only view; raise if any metric observation failed."""

        report = self.evaluate_samples(eval_data)
        if report.failure_count:
            raise RAGASEvaluationError(
                "one or more RAGAS metrics failed; inspect exception.report",
                report,
            )
        return {
            metric_name: float(summary.mean)
            for metric_name, summary in report.summaries.items()
            if summary.mean is not None
        }
