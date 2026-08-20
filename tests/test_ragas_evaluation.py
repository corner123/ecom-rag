import math

import pytest

from rag_core.evaluation.ragas_eval import (
    RAGASEvaluationConfigurationError,
    RAGASEvaluationError,
    RAGASEvaluator,
    RAGASSampleValidationError,
)


class FakeMetric:
    def __init__(self, name):
        self.name = name


def sample(**overrides):
    value = {
        "user_input": "What does the runtime do?",
        "retrieved_contexts": ["The runtime executes tools."],
        "response": "It executes tools.",
        "reference": "The runtime executes registered tools.",
    }
    value.update(overrides)
    return value


def test_strict_evaluator_builds_single_turn_dataset_and_injects_dependencies():
    llm_wrapper = object()
    embeddings_wrapper = object()
    metrics = {
        "faithfulness": FakeMetric("faithfulness"),
        "answer_relevancy": FakeMetric("answer_relevancy"),
        "context_precision": FakeMetric("llm_context_precision_with_reference"),
        "context_recall": FakeMetric("context_recall"),
    }
    calls = []

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        ragas_sample = kwargs["dataset"].samples[0]
        assert ragas_sample.user_input == "What does the runtime do?"
        assert ragas_sample.retrieved_contexts == ["The runtime executes tools."]
        assert ragas_sample.response == "It executes tools."
        assert ragas_sample.reference == "The runtime executes registered tools."
        metric_name = kwargs["metrics"][0].name
        return {metric_name: [0.8]}

    evaluator = RAGASEvaluator(
        llm_wrapper=llm_wrapper,
        embeddings_wrapper=embeddings_wrapper,
        metrics=metrics,
        evaluate_fn=fake_evaluate,
    )
    report = evaluator.evaluate_samples([sample()])

    assert len(calls) == 4
    assert report.sample_count == 1
    assert report.metric_count == 4
    assert report.overall_coverage == 1.0
    assert report.summaries["faithfulness"].mean == pytest.approx(0.8)
    for call in calls:
        assert call["llm"] is llm_wrapper
        assert call["embeddings"] is embeddings_wrapper
        assert call["raise_exceptions"] is True
        assert call["show_progress"] is False
        assert call["batch_size"] == 1


def test_metric_failure_is_null_with_error_and_reduces_coverage():
    metrics = {
        "faithfulness": FakeMetric("faithfulness"),
        "answer_relevancy": FakeMetric("answer_relevancy"),
        "context_precision": FakeMetric("llm_context_precision_with_reference"),
        "context_recall": FakeMetric("context_recall"),
    }

    def fake_evaluate(**kwargs):
        metric_name = kwargs["metrics"][0].name
        if metric_name == "answer_relevancy":
            raise TimeoutError("judge timed out")
        return {metric_name: [0.6]}

    evaluator = RAGASEvaluator(
        llm_wrapper=object(),
        embeddings_wrapper=object(),
        metrics=metrics,
        evaluate_fn=fake_evaluate,
    )
    report = evaluator.evaluate_samples([sample()])

    failed = report.samples[0].metrics["answer_relevancy"]
    assert failed.score is None
    assert failed.status == "error"
    assert failed.error_type == "TimeoutError"
    assert failed.error_message == "judge timed out"
    assert report.summaries["answer_relevancy"].mean is None
    assert report.summaries["answer_relevancy"].coverage == 0.0
    assert report.overall_coverage == pytest.approx(0.75)
    assert report.failure_count == 1
    assert report.to_dict()["samples"][0]["metrics"]["answer_relevancy"]["score"] is None


def test_non_finite_score_is_an_explicit_error_not_zero():
    metric = FakeMetric("faithfulness")

    def fake_evaluate(**kwargs):
        return {"faithfulness": [math.nan]}

    evaluator = RAGASEvaluator(
        llm_wrapper=object(),
        embeddings_wrapper=object(),
        metrics={"faithfulness": metric},
        evaluate_fn=fake_evaluate,
    )
    report = evaluator.evaluate_samples([sample()])

    observation = report.samples[0].metrics["faithfulness"]
    assert observation.score is None
    assert observation.status == "error"
    assert observation.error_type == "ValueError"
    assert "non-finite" in observation.error_message


def test_legacy_single_view_raises_with_structured_report_on_failure():
    metric = FakeMetric("faithfulness")

    def fake_evaluate(**kwargs):
        raise RuntimeError("provider unavailable")

    evaluator = RAGASEvaluator(
        metrics={"faithfulness": metric},
        evaluate_fn=fake_evaluate,
    )

    with pytest.raises(RAGASEvaluationError) as exc_info:
        evaluator.evaluate_single("q", "a", ["c"], "r")

    assert exc_info.value.report.failure_count == 1
    assert exc_info.value.report.samples[0].metrics["faithfulness"].score is None


def test_legacy_batch_aliases_are_supported_when_all_metrics_succeed():
    metric = FakeMetric("faithfulness")

    def fake_evaluate(**kwargs):
        return {"faithfulness": [0.4]}

    evaluator = RAGASEvaluator(
        metrics={"faithfulness": metric},
        evaluate_fn=fake_evaluate,
    )
    scores = evaluator.evaluate_batch(
        [
            {
                "question": "q",
                "contexts": ["c"],
                "answer": "a",
                "ground_truth": "r",
            }
        ]
    )

    assert scores == {"faithfulness": pytest.approx(0.4)}


def test_default_suite_requires_both_explicit_wrappers():
    evaluator = RAGASEvaluator(llm_wrapper=object())

    with pytest.raises(
        RAGASEvaluationConfigurationError,
        match="embeddings_wrapper",
    ):
        evaluator.evaluate_samples([sample()])


def test_custom_metrics_cannot_reenable_real_ragas_provider_defaults():
    evaluator = RAGASEvaluator(
        metrics={"faithfulness": FakeMetric("faithfulness")},
    )

    with pytest.raises(
        RAGASEvaluationConfigurationError,
        match="explicit llm_wrapper and embeddings_wrapper",
    ):
        evaluator.evaluate_samples([sample()])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"user_input": ""}, "user_input"),
        ({"retrieved_contexts": "not-a-list"}, "retrieved_contexts"),
        ({"response": None}, "response"),
        ({"reference": ""}, "reference"),
    ],
)
def test_invalid_samples_fail_before_metric_execution(overrides, message):
    evaluator = RAGASEvaluator(
        metrics={"faithfulness": FakeMetric("faithfulness")},
        evaluate_fn=lambda **_: {"faithfulness": [1.0]},
    )

    with pytest.raises(RAGASSampleValidationError, match=message):
        evaluator.evaluate_samples([sample(**overrides)])


def test_empty_batch_is_rejected_instead_of_returning_zero_means():
    evaluator = RAGASEvaluator(
        metrics={"faithfulness": FakeMetric("faithfulness")},
        evaluate_fn=lambda **_: {"faithfulness": [1.0]},
    )

    with pytest.raises(RAGASSampleValidationError, match="at least one"):
        evaluator.evaluate_samples([])
