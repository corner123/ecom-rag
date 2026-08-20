from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_core.engineering.grounding import GroundedAnswerer
from rag_core.engineering.sufficiency import SufficiencyProfile
from rag_core.engineering.models import (
    AnswerOutcome,
    EvidenceCitation,
    GenerationStatus,
    RetrievalOutcome,
    RetrievedEvidence,
)
from rag_core.evaluation.ragas_eval import (
    RAGASEvaluationReport,
    RAGASMetricObservation,
    RAGASSampleResult,
)
from rag_core.evaluation.response_experiment import (
    RAGASJudgeBundle,
    ResponseExperimentValidationError,
    _score_deterministic_retrieval,
    build_response_snapshot,
    generate_response_records,
    judge_response_records,
    recompute_deterministic_retrieval_metrics,
    run_response_experiment,
    validate_frozen_response_inputs,
    write_response_snapshot,
)
from rag_core.evaluation.response_dataset import (
    dataset_sha256,
    file_sha256,
    load_response_jsonl,
)
from rag_core.evaluation.retrieval_profiles import RetrievalExperimentProfile
from rag_core.ingestion import BuildManifest
from rag_core.retrieval.engineering import EngineeringSearchResult, SourceIntent


def _dataset_payload() -> dict[str, object]:
    return {
        "id": "response-1",
        "question": "QueryEngine 如何执行循环？",
        "category": "runtime",
        "answerable": True,
        "expected_route": "implementation",
        "reference_answer": "QueryEngine 通过显式循环驱动工具调用。SECRET_REFERENCE",
        "reference_claims": [
            {
                "claim_id": "C1",
                "text": "QueryEngine 使用显式循环。",
                "required": True,
                "evidence_sources": ["mini_nanobot/core/query_engine.py"],
                "locator": "mini_nanobot/core/query_engine.py#L1-L10",
            }
        ],
        "difficulty": "easy",
        "query_variant": "canonical",
        "dataset_role": "development_response",
        "source_revision": "revision-test",
        "reference_review_status": "human_reviewed",
        "label_version": "response-eval/v3",
    }


def _frozen_files(tmp_path: Path):
    dataset = tmp_path / "response.jsonl"
    dataset.write_text(
        json.dumps(_dataset_payload(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    BuildManifest(
        build_id="build-test",
        created_at="2026-08-13T00:00:00+00:00",
        sources=[],
        documents=[],
        chunks=[],
    ).write(manifest)
    index_root = tmp_path / "index"
    index_root.mkdir()
    (index_root / "partitions.json").write_text(
        json.dumps({"build_id": "build-test"}) + "\n", encoding="utf-8"
    )
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            build_response_snapshot(
                dataset_path=dataset,
                manifest_path=manifest,
                index_root=index_root,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset, manifest, index_root, snapshot


def _evidence(source: str, content: str, citation_id: str) -> RetrievedEvidence:
    result = EngineeringSearchResult(
        content=content,
        source=source,
        corpus="internal",
        authority="code",
        retriever="fake",
        metadata={"evidence_role": "current_implementation"},
    )
    citation = EvidenceCitation(
        citation_id=citation_id,
        source=source,
        corpus="internal",
        authority="code",
        evidence_role="current_implementation",
    )
    return RetrievedEvidence(citation=citation, result=result)


class FakeAnswerer(GroundedAnswerer):
    def __init__(self) -> None:
        generator = SimpleNamespace(
            last_usage={
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
            }
        )
        super().__init__(
            generator,
            provider="fake-provider",
            model="fake-generator",
        )

    def answer(self, retrieval: RetrievalOutcome) -> AnswerOutcome:
        raw = [
            _evidence(
                "workspace/mini_nanobot/core/query_engine.py",
                "RAW_CONTEXT_A",
                "E1",
            ),
            _evidence("b.py", "RAW_CONTEXT_B", "E2"),
        ]
        return AnswerOutcome(
            query=retrieval.query,
            intent=retrieval.intent,
            answer="答案只依据第一条上下文 [E1]",
            refused=False,
            refusal_reason=None,
            citations=list(retrieval.citations),
            generation_mode="model",
            generation_provider="fake-provider",
            generation_model="fake-generator",
            retrieved_evidence=raw,
            generation_context=[raw[0]],
            answer_citations=[raw[0].citation],
            generation=GenerationStatus(
                status="model",
                attempted=True,
                succeeded=True,
                provider="fake-provider",
                model="fake-generator",
                retrieved_count=2,
                context_count=1,
            ),
        )


class FakeService:
    def __init__(self) -> None:
        self.answerer = FakeAnswerer()
        self.questions: list[str] = []

    def retrieve(self, query: str, *, top_k: int):
        self.questions.append(query)
        evidence = _evidence(
            "workspace/mini_nanobot/core/query_engine.py",
            "RAW_CONTEXT_A",
            "E1",
        )
        return RetrievalOutcome(
            query=query,
            intent=SourceIntent.IMPLEMENTATION,
            results=[evidence.result],
            citations=[evidence.citation],
            sufficient_evidence=True,
        )


class StickyUsageGenerator:
    """Keep usage until called again, matching the original leak precondition."""

    provider = "fake-provider"
    model = "fake-generator"

    def __init__(self) -> None:
        self.last_usage = {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
        }

    def __call__(self, prompt: str) -> str:
        return "The explicit loop drives the tool execution. [E1]"


class ModelThenRefusalService(FakeService):
    """Return sufficient evidence once, then refuse before model generation."""

    def __init__(self) -> None:
        super().__init__()
        generator = StickyUsageGenerator()
        self.answerer = GroundedAnswerer(
            generator,
            provider=generator.provider,
            model=generator.model,
        )

    def retrieve(self, query: str, *, top_k: int):
        outcome = super().retrieve(query, top_k=top_k)
        if "does not exist" not in query:
            return outcome
        return RetrievalOutcome(
            query=query,
            intent=SourceIntent.IMPLEMENTATION,
            results=outcome.results,
            citations=outcome.citations,
            sufficient_evidence=False,
            refusal_reason="missing_current_implementation_evidence",
        )


class CapturingGenerator:
    provider = "fake-provider"
    model = "fake-generator"

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5}

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "QueryEngine 使用显式循环 [E1]"


class FakeEvaluator:
    def __init__(self, scores: dict[str, float | None]) -> None:
        self.scores = scores
        self.samples = []

    def evaluate_samples(self, samples, *, metric_names=None):
        sample = list(samples)[0]
        self.samples.append(sample)
        selected = (
            self.scores
            if metric_names is None
            else {name: self.scores[name] for name in metric_names}
        )
        observations = {
            name: RAGASMetricObservation(
                score=score,
                status="ok" if score is not None else "error",
                error_type=None if score is not None else "JudgeError",
                error_message=None if score is not None else "structured failure",
            )
            for name, score in selected.items()
        }
        return RAGASEvaluationReport(
            sample_count=1,
            metric_count=len(observations),
            overall_coverage=(
                sum(value is not None for value in selected.values())
                / len(selected)
            ),
            summaries={},
            samples=[
                RAGASSampleResult(
                    sample_index=0,
                    user_input=sample.user_input,
                    metrics=observations,
                )
            ],
        )


class InterruptingEvaluator(FakeEvaluator):
    def __init__(self, scores, *, interrupt_metric: str) -> None:
        super().__init__(scores)
        self.interrupt_metric = interrupt_metric

    def evaluate_samples(self, samples, *, metric_names=None):
        selected = list(metric_names or self.scores)
        if selected == [self.interrupt_metric]:
            raise KeyboardInterrupt(f"interrupted at {self.interrupt_metric}")
        return super().evaluate_samples(samples, metric_names=metric_names)


def _metadata() -> dict[str, object]:
    return {
        "experiment_id": "experiment-test",
        "dataset_canonical_sha256": "dataset-test",
    }


def _judges() -> RAGASJudgeBundle:
    return RAGASJudgeBundle(
        retrieval=FakeEvaluator(
            {"context_precision": 0.8, "context_recall": 0.7}
        ),
        response=FakeEvaluator(
            {
                "faithfulness": 0.9,
                "answer_relevancy": 0.7,
                "answer_correctness": 0.6,
            }
        ),
        metadata={
            "provider": "fake",
            "model": "fake-judge",
            "temperature": 0,
            "prompt_sha256": "prompt-test",
        },
    )


def test_snapshot_validation_fails_closed_on_dataset_change(tmp_path: Path):
    dataset, manifest, index_root, snapshot = _frozen_files(tmp_path)

    frozen = validate_frozen_response_inputs(
        dataset_path=dataset,
        snapshot_path=snapshot,
        manifest_path=manifest,
        index_root=index_root,
    )
    assert frozen.manifest.build_id == "build-test"
    assert len(frozen.samples) == 1

    dataset.write_text(dataset.read_text("utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ResponseExperimentValidationError, match="dataset file SHA"):
        validate_frozen_response_inputs(
            dataset_path=dataset,
            snapshot_path=snapshot,
            manifest_path=manifest,
            index_root=index_root,
        )


def test_snapshot_writer_never_overwrites(tmp_path: Path):
    dataset, manifest, index_root, _snapshot = _frozen_files(tmp_path)
    output = tmp_path / "new-snapshot.json"

    snapshot = write_response_snapshot(
        dataset_path=dataset,
        manifest_path=manifest,
        index_root=index_root,
        output_path=output,
    )

    assert snapshot["manifest_build_id"] == "build-test"
    assert output.is_file()
    with pytest.raises(FileExistsError):
        write_response_snapshot(
            dataset_path=dataset,
            manifest_path=manifest,
            index_root=index_root,
            output_path=output,
        )


def test_generation_persists_raw_and_actual_context_without_reference(tmp_path: Path):
    dataset, _manifest, _index_root, _snapshot = _frozen_files(tmp_path)
    sample = load_response_jsonl(dataset)[0]
    output = tmp_path / "generated.json"
    service = FakeService()

    artifact = generate_response_records(
        service=service,
        samples=[sample],
        output_path=output,
        metadata=_metadata(),
        top_k=5,
    )

    record = artifact["records"][0]
    assert service.questions == [sample.question]
    assert artifact["stage"] == "generated"
    assert [item["content"] for item in record["retrieved_evidence"]] == [
        "RAW_CONTEXT_A",
        "RAW_CONTEXT_B",
    ]
    assert [item["content"] for item in record["generation_context"]] == [
        "RAW_CONTEXT_A"
    ]
    assert record["answer_citations"][0]["citation_id"] == "E1"
    assert record["generation_usage"] == {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
    }
    assert "reference_answer" not in record
    assert "SECRET_REFERENCE" not in output.read_text("utf-8")
    assert record["retrieval_latency_ms"] is not None
    assert record["generation_latency_ms"] is not None
    assert record["total_latency_ms"] is not None

    with pytest.raises(FileExistsError):
        generate_response_records(
            service=service,
            samples=[sample],
            output_path=output,
            metadata=_metadata(),
            top_k=5,
        )
    resumed = generate_response_records(
        service=service,
        samples=[sample],
        output_path=output,
        metadata=_metadata(),
        top_k=5,
        resume=True,
    )
    assert len(resumed["records"]) == 1
    assert service.questions == [sample.question]


def test_generation_usage_does_not_leak_from_model_answer_into_next_refusal(
    tmp_path: Path,
):
    dataset, _manifest, _index_root, _snapshot = _frozen_files(tmp_path)
    first = load_response_jsonl(dataset)[0]
    second = replace(
        first,
        id="response-2",
        question="What is an implementation that does not exist?",
        answerable=False,
        reference_answer="The evidence is insufficient, so refuse.",
        reference_claims=(),
    )
    output = tmp_path / "model-then-refusal.json"
    service = ModelThenRefusalService()

    artifact = generate_response_records(
        service=service,
        samples=[first, second],
        output_path=output,
        metadata=_metadata(),
        top_k=5,
    )

    model_record, refusal_record = artifact["records"]
    assert model_record["generation_status"]["attempted"] is True
    assert model_record["generation_usage"] == {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
    }
    assert refusal_record["generation_status"]["attempted"] is False
    assert refusal_record["generation_usage"] is None


def test_actual_generator_prompt_never_receives_reference(tmp_path: Path):
    dataset, _manifest, _index_root, _snapshot = _frozen_files(tmp_path)
    sample = load_response_jsonl(dataset)[0]
    generator = CapturingGenerator()
    service = FakeService()
    service.answerer = GroundedAnswerer(
        generator,
        provider=generator.provider,
        model=generator.model,
    )
    output = tmp_path / "generated-with-prompt.json"

    artifact = generate_response_records(
        service=service,
        samples=[sample],
        output_path=output,
        metadata=_metadata(),
        top_k=5,
    )

    assert len(generator.prompts) == 1
    assert sample.question in generator.prompts[0]
    assert "RAW_CONTEXT_A" in generator.prompts[0]
    assert "SECRET_REFERENCE" not in generator.prompts[0]
    assert "SECRET_REFERENCE" not in output.read_text("utf-8")
    assert artifact["records"][0]["generation_succeeded"] is True


def test_judge_uses_raw_context_for_retrieval_and_actual_context_for_answer(
    tmp_path: Path,
):
    dataset, _manifest, _index_root, _snapshot = _frozen_files(tmp_path)
    sample = load_response_jsonl(dataset)[0]
    generated_path = tmp_path / "generated.json"
    generated = generate_response_records(
        service=FakeService(),
        samples=[sample],
        output_path=generated_path,
        metadata=_metadata(),
        top_k=5,
    )
    retrieval = FakeEvaluator(
        {"context_precision": 0.8, "context_recall": None}
    )
    response = FakeEvaluator(
        {
            "faithfulness": 0.9,
            "answer_relevancy": 0.7,
            "answer_correctness": 0.6,
        }
    )
    output = tmp_path / "judged.json"

    judged = judge_response_records(
        replay_path=generated_path,
        output_path=output,
        samples=[sample],
        metadata=_metadata(),
        judges=RAGASJudgeBundle(
            retrieval=retrieval,
            response=response,
            metadata={
                "provider": "fake",
                "model": "fake-judge",
                "temperature": 0,
                "prompt_sha256": "prompt-test",
            },
        ),
    )

    record = judged["records"][0]
    assert retrieval.samples[0].retrieved_contexts == [
        "RAW_CONTEXT_A",
        "RAW_CONTEXT_B",
    ]
    assert response.samples[0].retrieved_contexts == ["RAW_CONTEXT_A"]
    assert retrieval.samples[0].reference == sample.reference_answer
    assert response.samples[0].reference == sample.reference_answer
    assert record["metrics"] == {
        "context_precision": 0.8,
        "context_recall": None,
        "faithfulness": 0.9,
        "answer_relevancy": 0.7,
        "answer_correctness": 0.6,
    }
    assert record["deterministic_retrieval_metrics"] == {
        "hit_at_k": 1.0,
        "required_claim_recall_at_k": 1.0,
        "mrr": 1.0,
        "source_precision_at_k": 0.5,
        "source_option_recall": 1.0,
    }
    assert record["deterministic_retrieval_metric_status"] == "complete"
    assert record["deterministic_retrieval_metric_error"] is None
    assert record["judge_status"] == "partial_or_failed"
    assert record["metric_errors"] == [
        {
            "metric": "context_recall",
            "code": "ragas_metric_error",
            "error_type": "JudgeError",
            "message": "structured failure",
        }
    ]
    assert judged["metadata"]["judge"]["temperature"] == 0


def test_unanswerable_is_not_scored_as_retrieval_failure(tmp_path: Path):
    payload = _dataset_payload()
    payload.update(
        {
            "id": "negative-boundary",
            "answerable": False,
            "expected_route": "out_of_scope",
            "reference_answer": "该问题不属于当前知识库范围。",
            "reference_claims": [
                {
                    "claim_id": "C1",
                    "text": "应拒绝回答。",
                    "required": True,
                    "evidence_sources": [],
                    "locator": "boundary:out-of-scope",
                }
            ],
            "query_variant": "boundary",
        }
    )
    dataset = tmp_path / "negative.jsonl"
    dataset.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    sample = load_response_jsonl(dataset)[0]
    generated_path = tmp_path / "negative-generated.json"
    generate_response_records(
        service=FakeService(),
        samples=[sample],
        output_path=generated_path,
        metadata=_metadata(),
        top_k=5,
    )
    judged = judge_response_records(
        replay_path=generated_path,
        output_path=tmp_path / "negative-judged.json",
        samples=[sample],
        metadata=_metadata(),
        judges=_judges(),
    )

    record = judged["records"][0]
    assert record["deterministic_retrieval_metric_status"] == "not_applicable"
    assert all(
        value is None
        for value in record["deterministic_retrieval_metrics"].values()
    )
    assert record["deterministic_retrieval_metric_error"] is None


def test_deterministic_retrieval_scores_rank_recall_and_unique_sources(
    tmp_path: Path,
):
    payload = _dataset_payload()
    payload["reference_claims"][0]["evidence_sources"] = [
        "mini_nanobot/core/query_engine.py",
        "docs/runtime.md",
    ]
    dataset = tmp_path / "ranked.jsonl"
    dataset.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    sample = load_response_jsonl(dataset)[0]
    record = {
        "retrieved_evidence": [
            {"source": "distractor.md", "content": "irrelevant"},
            {
                "source": "repo/mini_nanobot/core/query_engine.py",
                "content": "implementation",
            },
            {
                "source": "repo/mini_nanobot/core/query_engine.py",
                "content": "duplicate chunk",
            },
            {"source": "docs/runtime.md", "content": "design"},
        ],
        "failure": None,
    }

    _score_deterministic_retrieval(record, sample)

    assert record["deterministic_retrieval_metrics"] == {
        "hit_at_k": 1.0,
        "required_claim_recall_at_k": 1.0,
        "mrr": 0.5,
        "source_precision_at_k": pytest.approx(2 / 3),
        "source_option_recall": 1.0,
    }


def test_deterministic_retrieval_scores_claim_options_and_file_locators(tmp_path: Path):
    payload = _dataset_payload()
    payload["reference_claims"] = [
        {
            "claim_id": "C1",
            "text": "one required fact",
            "required": True,
            "evidence_sources": [
                "mini_nanobot/core/query_engine.py#symbol:QueryEngine.submit_message",
                "docs/runtime.md#section-loop",
            ],
            "locator": "mini_nanobot/core/query_engine.py#symbol:QueryEngine.submit_message",
        },
        {
            "claim_id": "C2",
            "text": "another required fact",
            "required": True,
            "evidence_sources": ["docs/missing.md#section-checkpoint"],
            "locator": "docs/missing.md#section-checkpoint",
        },
    ]
    dataset = tmp_path / "claims.jsonl"
    dataset.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    sample = load_response_jsonl(dataset)[0]
    record = {
        "retrieved_evidence": [
            {"source": "repo/mini_nanobot/core/query_engine.py", "content": "x"},
            {"source": "repo/mini_nanobot/core/query_engine.py#symbol:Other", "content": "y"},
            {"source": "distractor.md", "content": "z"},
        ],
        "failure": None,
    }

    _score_deterministic_retrieval(record, sample)

    assert record["deterministic_retrieval_metrics"] == {
        "hit_at_k": 1.0,
        "required_claim_recall_at_k": 0.5,
        "mrr": 1.0,
        "source_precision_at_k": 0.5,
        "source_option_recall": pytest.approx(1 / 3),
    }


def test_deterministic_retrieval_keeps_url_fragment_semantics(tmp_path: Path):
    payload = _dataset_payload()
    payload["reference_claims"][0]["evidence_sources"] = [
        "https://example.com/spec#required-section"
    ]
    payload["reference_claims"][0]["locator"] = (
        "https://example.com/spec#required-section"
    )
    dataset = tmp_path / "url.jsonl"
    dataset.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    sample = load_response_jsonl(dataset)[0]
    record = {
        "retrieved_evidence": [
            {"source": "https://example.com/spec#other-section", "content": "x"}
        ],
        "failure": None,
    }

    _score_deterministic_retrieval(record, sample)

    assert record["deterministic_retrieval_metrics"]["hit_at_k"] == 0.0


def test_offline_rescore_validates_dataset_and_refuses_overwrite(tmp_path: Path):
    dataset = tmp_path / "response.jsonl"
    dataset.write_text(json.dumps(_dataset_payload()) + "\n", encoding="utf-8")
    sample = load_response_jsonl(dataset)[0]
    artifact_path = tmp_path / "judged.json"
    artifact = {
        "schema_version": "engineering-response-experiment/v1",
        "stage": "judged",
        "metadata": {
            "dataset_file_sha256": file_sha256(dataset),
            "dataset_canonical_sha256": dataset_sha256([sample]),
        },
        "records": [
            {
                "id": sample.id,
                "question": sample.question,
                "answerable": True,
                "retrieved_evidence": [
                    {"source": "mini_nanobot/core/query_engine.py#symbol:X"}
                ],
                "failure": None,
                "metrics": {},
            }
        ],
    }
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    output = tmp_path / "rescored.json"

    result = recompute_deterministic_retrieval_metrics(
        artifact_path=artifact_path,
        dataset_path=dataset,
        output_path=output,
    )

    assert result["stage"] == "judged"
    assert result["records"][0]["deterministic_retrieval_metrics"][
        "required_claim_recall_at_k"
    ] == 1.0
    assert result["metadata"]["deterministic_retrieval_rescore"][
        "network_or_model_calls"
    ] is False
    with pytest.raises(ResponseExperimentValidationError, match="overwrite"):
        recompute_deterministic_retrieval_metrics(
            artifact_path=artifact_path,
            dataset_path=dataset,
            output_path=output,
        )


def test_resume_rejects_different_experiment_identity(tmp_path: Path):
    dataset, _manifest, _index_root, _snapshot = _frozen_files(tmp_path)
    sample = load_response_jsonl(dataset)[0]
    output = tmp_path / "generated.json"
    generate_response_records(
        service=FakeService(),
        samples=[sample],
        output_path=output,
        metadata=_metadata(),
        top_k=5,
    )

    with pytest.raises(ResponseExperimentValidationError, match="experiment_id"):
        generate_response_records(
            service=FakeService(),
            samples=[sample],
            output_path=output,
            metadata={"experiment_id": "different"},
            top_k=5,
            resume=True,
        )


def test_full_experiment_generates_then_judges_in_place(tmp_path: Path):
    dataset, manifest, index_root, snapshot = _frozen_files(tmp_path)
    output = tmp_path / "full.json"

    artifact = run_response_experiment(
        dataset_path=dataset,
        snapshot_path=snapshot,
        manifest_path=manifest,
        index_root=index_root,
        mini_nanobot_repo=tmp_path,
        output_path=output,
        profile=RetrievalExperimentProfile(name="baseline"),
        service=FakeService(),
        judges=_judges(),
    )

    assert artifact["stage"] == "judged"
    assert artifact["records"][0]["judge_status"] == "complete"
    assert artifact["records"][0]["metrics"]["faithfulness"] == 0.9
    assert json.loads(output.read_text("utf-8"))["stage"] == "judged"


def test_sufficiency_profile_is_part_of_replay_identity_and_code_hashes(
    tmp_path: Path,
):
    dataset, manifest, index_root, snapshot = _frozen_files(tmp_path)
    profile = RetrievalExperimentProfile(name="same-retrieval")

    legacy = run_response_experiment(
        dataset_path=dataset,
        snapshot_path=snapshot,
        manifest_path=manifest,
        index_root=index_root,
        mini_nanobot_repo=tmp_path,
        output_path=tmp_path / "legacy.json",
        profile=profile,
        sufficiency_profile=SufficiencyProfile.LEGACY_EXACT_SLASH,
        generate_only=True,
        service=FakeService(),
    )
    optimized = run_response_experiment(
        dataset_path=dataset,
        snapshot_path=snapshot,
        manifest_path=manifest,
        index_root=index_root,
        mini_nanobot_repo=tmp_path,
        output_path=tmp_path / "optimized.json",
        profile=profile,
        sufficiency_profile=(
            SufficiencyProfile.SPLIT_NATURAL_SLASH_CONCEPTS
        ),
        generate_only=True,
        service=FakeService(),
    )

    assert legacy["metadata"]["sufficiency_profile"] == "legacy_exact_slash"
    assert optimized["metadata"]["sufficiency_profile"] == (
        "split_natural_slash_concepts"
    )
    assert legacy["metadata"]["experiment_id"] != optimized["metadata"][
        "experiment_id"
    ]
    assert set(legacy["metadata"]["code_sha256"]) >= {
        "sufficiency",
        "support_selection",
        "routing",
        "index",
        "retrieval_profiles",
    }


def test_support_selection_profile_is_part_of_experiment_identity(
    tmp_path: Path,
):
    dataset, manifest, index_root, snapshot = _frozen_files(tmp_path)
    profile = RetrievalExperimentProfile(name="same-retrieval")

    legacy = run_response_experiment(
        dataset_path=dataset,
        snapshot_path=snapshot,
        manifest_path=manifest,
        index_root=index_root,
        mini_nanobot_repo=tmp_path,
        output_path=tmp_path / "legacy-support.json",
        profile=profile,
        support_selection_profile="legacy_first",
        generate_only=True,
        service=FakeService(),
    )
    optimized = run_response_experiment(
        dataset_path=dataset,
        snapshot_path=snapshot,
        manifest_path=manifest,
        index_root=index_root,
        mini_nanobot_repo=tmp_path,
        output_path=tmp_path / "query-aware-support.json",
        profile=profile,
        support_selection_profile="query_aware_diverse",
        generate_only=True,
        service=FakeService(),
    )

    assert legacy["metadata"]["support_selection_profile"] == "legacy_first"
    assert optimized["metadata"]["support_selection_profile"] == (
        "query_aware_diverse"
    )
    assert legacy["metadata"]["experiment_id"] != optimized["metadata"][
        "experiment_id"
    ]


def test_judge_only_replays_generation_and_rejects_profile_mismatch(tmp_path: Path):
    dataset, manifest, index_root, snapshot = _frozen_files(tmp_path)
    generated_path = tmp_path / "generated.json"
    profile = RetrievalExperimentProfile(name="baseline")
    run_response_experiment(
        dataset_path=dataset,
        snapshot_path=snapshot,
        manifest_path=manifest,
        index_root=index_root,
        mini_nanobot_repo=tmp_path,
        output_path=generated_path,
        profile=profile,
        generate_only=True,
        service=FakeService(),
    )
    judged_path = tmp_path / "judged.json"

    judged = run_response_experiment(
        dataset_path=dataset,
        snapshot_path=snapshot,
        manifest_path=manifest,
        index_root=index_root,
        mini_nanobot_repo=tmp_path,
        output_path=judged_path,
        profile=profile,
        judge_only=True,
        replay_path=generated_path,
        judges=_judges(),
    )

    assert judged["stage"] == "judged"
    generated_response = json.loads(generated_path.read_text("utf-8"))["records"][0][
        "response"
    ]
    assert judged["records"][0]["response"] == generated_response
    with pytest.raises(ResponseExperimentValidationError, match="profile"):
        run_response_experiment(
            dataset_path=dataset,
            snapshot_path=snapshot,
            manifest_path=manifest,
            index_root=index_root,
            mini_nanobot_repo=tmp_path,
            output_path=tmp_path / "wrong-profile.json",
            profile=RetrievalExperimentProfile(
                name="candidate", dense_weight=0.5, bm25_weight=1.0
            ),
            judge_only=True,
            replay_path=generated_path,
            judges=_judges(),
        )


def test_retry_from_only_calls_failed_metrics_and_preserves_successes_and_history(
    tmp_path: Path,
):
    dataset, _manifest, _index_root, _snapshot = _frozen_files(tmp_path)
    first = load_response_jsonl(dataset)[0]
    second = replace(first, id="response-2", question="Second complete question")
    generated_path = tmp_path / "generated-two.json"
    generate_response_records(
        service=FakeService(),
        samples=[first, second],
        output_path=generated_path,
        metadata=_metadata(),
        top_k=5,
    )
    partial_path = tmp_path / "partial.json"
    partial = judge_response_records(
        replay_path=generated_path,
        output_path=partial_path,
        samples=[first, second],
        metadata=_metadata(),
        judges=RAGASJudgeBundle(
            retrieval=FakeEvaluator(
                {"context_precision": 0.81, "context_recall": None}
            ),
            response=FakeEvaluator(
                {
                    "faithfulness": 0.91,
                    "answer_relevancy": None,
                    "answer_correctness": 0.61,
                }
            ),
            metadata=_judges().metadata,
        ),
    )
    # Simulate another question having completed before the interrupted/partial
    # run.  Only Judge fields change; frozen generation identity is untouched.
    partial["records"][1]["metrics"].update(
        {"context_recall": 0.72, "answer_relevancy": 0.73}
    )
    partial["records"][1]["metric_errors"] = []
    partial["records"][1]["judge_status"] = "complete"
    partial_path.write_text(json.dumps(partial), encoding="utf-8")
    old_successes = {
        name: partial["records"][0]["metrics"][name]
        for name in ("context_precision", "faithfulness", "answer_correctness")
    }
    old_history = list(partial["records"][0]["error_history"])

    retry_judges = _judges()
    retried = judge_response_records(
        replay_path=generated_path,
        retry_from_path=partial_path,
        output_path=tmp_path / "retried.json",
        samples=[first, second],
        metadata=_metadata(),
        judges=retry_judges,
    )

    assert len(retry_judges.retrieval.samples) == 1
    assert len(retry_judges.response.samples) == 1
    assert retried["records"][0]["metrics"] | old_successes == retried[
        "records"
    ][0]["metrics"]
    assert {
        name: retried["records"][0]["metrics"][name]
        for name in old_successes
    } == old_successes
    assert retried["records"][0]["judge_status"] == "complete"
    assert retried["records"][0]["metric_errors"] == []
    assert retried["records"][0]["error_history"][: len(old_history)] == old_history
    assert len(retried["records"][0]["error_history"]) == len(old_history)
    # The already-complete second question caused no evaluator call.
    assert all(
        sample.user_input == first.question
        for sample in (*retry_judges.retrieval.samples, *retry_judges.response.samples)
    )


def test_retry_from_rejects_judge_configuration_change_before_calls(tmp_path: Path):
    dataset, _manifest, _index_root, _snapshot = _frozen_files(tmp_path)
    sample = load_response_jsonl(dataset)[0]
    generated = tmp_path / "generated.json"
    generate_response_records(
        service=FakeService(),
        samples=[sample],
        output_path=generated,
        metadata=_metadata(),
        top_k=5,
    )
    partial_path = tmp_path / "partial.json"
    judge_response_records(
        replay_path=generated,
        output_path=partial_path,
        samples=[sample],
        metadata=_metadata(),
        judges=_judges(),
    )
    changed = _judges()
    changed = RAGASJudgeBundle(
        retrieval=changed.retrieval,
        response=changed.response,
        metadata={**changed.metadata, "model": "different-judge"},
    )

    with pytest.raises(ResponseExperimentValidationError, match="configuration differs"):
        judge_response_records(
            replay_path=generated,
            retry_from_path=partial_path,
            output_path=tmp_path / "must-not-exist.json",
            samples=[sample],
            metadata=_metadata(),
            judges=changed,
        )
    assert changed.retrieval.samples == []
    assert changed.response.samples == []


def test_interrupt_checkpoints_each_metric_and_retry_continues_in_new_output(
    tmp_path: Path,
):
    dataset, _manifest, _index_root, _snapshot = _frozen_files(tmp_path)
    sample = load_response_jsonl(dataset)[0]
    generated = tmp_path / "generated.json"
    generate_response_records(
        service=FakeService(),
        samples=[sample],
        output_path=generated,
        metadata=_metadata(),
        top_k=5,
    )
    partial_path = tmp_path / "interrupted.json"
    interrupting = RAGASJudgeBundle(
        retrieval=InterruptingEvaluator(
            {"context_precision": 0.83, "context_recall": 0.74},
            interrupt_metric="context_recall",
        ),
        response=FakeEvaluator(
            {
                "faithfulness": 0.9,
                "answer_relevancy": 0.8,
                "answer_correctness": 0.7,
            }
        ),
        metadata=_judges().metadata,
    )
    with pytest.raises(KeyboardInterrupt):
        judge_response_records(
            replay_path=generated,
            output_path=partial_path,
            samples=[sample],
            metadata=_metadata(),
            judges=interrupting,
        )

    persisted = json.loads(partial_path.read_text("utf-8"))
    assert persisted["stage"] == "judging"
    assert persisted["records"][0]["metrics"]["context_precision"] == 0.83
    assert len(persisted["records"][0]["judge_attempts"]) == 1

    retry_judges = _judges()
    resumed = judge_response_records(
        replay_path=generated,
        retry_from_path=partial_path,
        output_path=tmp_path / "resumed.json",
        samples=[sample],
        metadata=_metadata(),
        judges=retry_judges,
    )
    assert resumed["records"][0]["metrics"]["context_precision"] == 0.83
    assert len(retry_judges.retrieval.samples) == 1
    assert len(retry_judges.response.samples) == 3
    assert resumed["records"][0]["judge_status"] == "complete"
