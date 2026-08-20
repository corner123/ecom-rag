from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_core.evaluation.engineering import (
    EvaluationPrediction,
    EvaluationSample,
    _report_source,
    evaluate_strategy,
    expected_route,
    explicit_symbols,
    load_jsonl,
    render_markdown,
    run_ablation,
    source_matches,
    write_report,
)
from rag_core.retrieval.engineering.models import EngineeringSearchResult
from rag_core.retrieval.engineering.federated import FederatedRetriever
from rag_core.retrieval.engineering.routing import SourceIntentRouter
from rag_core.evaluation.engineering_adapter import _retriever_view
from scripts.publish_evaluation_reports import publish


def _samples() -> list[EvaluationSample]:
    return [
        EvaluationSample(
            id="code-1",
            question="QueryEngine.submit_message 的调用链是什么？",
            category="code_symbol_call_chain",
            expected_intent="trace_current_code",
            answerable=True,
            relevant_sources=("mini_nanobot/core/query_engine.py",),
            expected_answer_points=("进入 query",),
            source_scope="frozen_repository_docs_and_code",
            source_revision="abc1234",
        ),
        EvaluationSample(
            id="design-1",
            question="为什么使用手写循环？",
            category="design_architecture",
            expected_intent="explain_architecture",
            answerable=True,
            relevant_sources=("docs/adr/0001-loop.md",),
            expected_answer_points=("边界透明",),
            source_scope="frozen_repository_docs_and_code",
            source_revision="abc1234",
        ),
        EvaluationSample(
            id="boundary-1",
            question="线上 SLA 是多少？",
            category="unanswerable_boundary",
            expected_intent="recognize_unanswerable",
            answerable=False,
            relevant_sources=(),
            expected_answer_points=("没有生产证据",),
            source_scope="outside_frozen_repository",
            source_revision="abc1234",
        ),
    ]


def _predictor(sample: EvaluationSample, top_k: int) -> EvaluationPrediction:
    del top_k
    if sample.id == "code-1":
        return EvaluationPrediction(
            predicted_intent="implementation",
            results=(
                EngineeringSearchResult(content="noise", source="README.md"),
                EngineeringSearchResult(
                    content="def submit_message(): ...",
                    source="D:/work/Mini-Nanobot/mini_nanobot/core/query_engine.py",
                    symbol="QueryEngine.submit_message",
                    metadata={"relative_path": "mini_nanobot/core/query_engine.py"},
                ),
            ),
            latency_ms=2.0,
        )
    if sample.id == "design-1":
        return EvaluationPrediction(
            predicted_intent="design",
            results=(
                EngineeringSearchResult(
                    content="decision", source="docs/adr/0001-loop.md"
                ),
            ),
            latency_ms=3.0,
        )
    return EvaluationPrediction(
        predicted_intent="out_of_scope", refused=True, latency_ms=1.0
    )


def test_engineering_metrics_cover_route_file_symbol_mrr_and_refusal():
    result = evaluate_strategy(_samples(), _predictor, strategy_name="hybrid", top_k=5)
    metrics = result["metrics"]

    assert metrics["routing_accuracy"] == 1.0
    assert metrics["file_recall_at_k"] == 1.0
    assert metrics["symbol_recall_at_k"] == 1.0
    assert metrics["mrr"] == pytest.approx(0.75)
    assert metrics["unanswerable_refusal_rate"] == 1.0
    assert metrics["answerable_refusal_rate"] == 0.0
    assert metrics["p50_latency_ms"] == pytest.approx(2.0)
    assert metrics["p95_latency_ms"] == pytest.approx(2.9)
    assert metrics["p99_latency_ms"] == pytest.approx(2.98)
    assert metrics["file_eligible_questions"] == 2
    assert metrics["symbol_eligible_questions"] == 1


def test_ablation_writes_machine_and_human_readable_reports(tmp_path: Path):
    report = run_ablation(
        _samples(),
        {"bm25": _predictor, "dense": _predictor, "hybrid": _predictor},
        top_k=3,
        dataset_name="fixture.jsonl",
    )
    json_path, markdown_path = write_report(report, tmp_path / "ablation")

    assert json.loads(json_path.read_text(encoding="utf-8"))["baseline"] == "bm25"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "| bm25 |" in markdown
    assert "| dense |" in markdown
    assert "| hybrid |" in markdown
    assert "P99 ms*" in markdown
    assert "not a statistically stable production" in markdown
    assert render_markdown(report) == markdown


def test_source_and_symbol_normalization_are_conservative():
    sample = _samples()[0]
    assert source_matches(
        "mini_nanobot/core/query_engine.py",
        r"D:\work\Mini-Nanobot\mini_nanobot\core\query_engine.py",
    )
    assert not source_matches("query.py", "query_engine.py")
    assert explicit_symbols(sample) == ("QueryEngine.submit_message",)


def test_persisted_report_source_requires_repository_relative_live_path():
    result = EngineeringSearchResult(
        content="def submit_message(): ...",
        source=r"D:\work\Mini-Nanobot\mini_nanobot\core\query_engine.py",
        metadata={"relative_path": r"mini_nanobot\core\query_engine.py"},
    )
    assert _report_source(result) == "mini_nanobot/core/query_engine.py"

    unsafe = EngineeringSearchResult(
        content="def submit_message(): ...",
        source=r"D:\work\Mini-Nanobot\mini_nanobot\core\query_engine.py",
    )
    with pytest.raises(ValueError, match="absolute local source"):
        _report_source(unsafe)


def test_public_report_redacts_paths_without_changing_metrics(tmp_path: Path):
    raw_path = tmp_path / "e2e.json"
    payload = {
        "metrics": {"primary_hit_at_k": 0.875, "latency_p95_ms": 123.5},
        "details": [
            {
                "id": "sample-1",
                "retrieved_sources": [
                    r"D:\work\Mini-Nanobot\mini_nanobot\core\query_engine.py"
                ],
            }
        ],
    }
    raw_path.write_text(json.dumps(payload), encoding="utf-8")
    raw_path.with_suffix(".md").write_text("# Frozen report\n", encoding="utf-8")

    public_json, public_markdown = publish(raw_path, tmp_path / "public")
    published = json.loads(public_json.read_text(encoding="utf-8"))

    assert published["metrics"] == payload["metrics"]
    assert published["details"][0]["retrieved_sources"] == [
        "mini_nanobot/core/query_engine.py"
    ]
    assert published["publication_metadata"]["metrics_unchanged"] is True
    assert published["publication_metadata"]["frozen_holdout_first_run"] is False
    assert "D:\\work\\Mini-Nanobot" not in public_json.read_text(encoding="utf-8")
    assert public_markdown.read_text(encoding="utf-8").startswith("> Public copy:")


def test_public_report_refuses_unrecognized_or_markdown_host_paths(tmp_path: Path):
    raw_path = tmp_path / "unsafe.json"
    raw_path.write_text(
        json.dumps({"retrieved_sources": [r"D:\private\unmapped.txt"]}),
        encoding="utf-8",
    )
    raw_path.with_suffix(".md").write_text("# report\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unrecognized absolute source"):
        publish(raw_path, tmp_path / "public")

    raw_path.write_text(json.dumps({"metrics": {"mrr": 1.0}}), encoding="utf-8")
    raw_path.with_suffix(".md").write_text(
        r"host path: D:\work\private-repository" + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="raw Markdown"):
        publish(raw_path, tmp_path / "public")


def test_public_holdout_report_is_self_describing(tmp_path: Path):
    raw_path = tmp_path / "e2e_holdout_first_run_build-1.json"
    raw_path.write_text(
        json.dumps({"run_metadata": {"dataset_roles": ["holdout"]}}),
        encoding="utf-8",
    )
    raw_path.with_suffix(".md").write_text("# Holdout\n", encoding="utf-8")

    public_json, public_markdown = publish(raw_path, tmp_path / "public")
    payload = json.loads(public_json.read_text(encoding="utf-8"))
    assert payload["publication_metadata"]["frozen_holdout_first_run"] is True
    assert "first formal run" in public_markdown.read_text(encoding="utf-8")


def test_loader_rejects_invalid_answerability_contract(tmp_path: Path):
    invalid = {
        "id": "bad",
        "question": "unknown",
        "category": "unanswerable_boundary",
        "expected_intent": "recognize_unanswerable",
        "answerable": False,
        "relevant_sources": ["README.md"],
        "expected_answer_points": ["none"],
        "source_scope": "outside_frozen_repository",
        "source_revision": "abc",
        "relevant_symbols": [],
        "expected_route": "out_of_scope",
        "primary_sources": ["README.md"],
        "supporting_sources": [],
        "refusal_reason": "unverifiable_private_information",
        "label_version": "engineering-eval/v2",
        "dataset_role": "development_regression",
    }
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot declare evidence sources"):
        load_jsonl(path)


def test_official_dataset_has_20_versioned_official_questions():
    root = Path(__file__).resolve().parents[1]
    samples = load_jsonl(root / "data/eval/official_engineering_specs.jsonl")

    assert len(samples) == 20
    assert len({sample.id for sample in samples}) == 20
    assert all(sample.answerable for sample in samples)
    assert all(sample.source_scope == "official_specifications" for sample in samples)
    assert all(sample.source_revision for sample in samples)
    assert all(
        source.startswith("https://")
        for sample in samples
        for source in sample.relevant_sources
    )


def test_real_ablation_adapter_changes_only_partition_retriever():
    class StaticRetriever:
        def __init__(self, name: str):
            self.name = name

        def search(self, query: str, top_k: int = 5):
            return [
                EngineeringSearchResult(
                    content=query,
                    source=f"{self.name}.md",
                    retriever=self.name,
                )
            ][:top_k]

    class Hybrid(StaticRetriever):
        def __init__(self):
            super().__init__("hybrid")
            self.bm25 = StaticRetriever("bm25")
            self.dense = StaticRetriever("dense")

    original = FederatedRetriever()
    original.add_partition("internal", "design", Hybrid(), weight=1.25)

    class FakeIndex:
        federated = original

    for mode in ("bm25", "dense", "hybrid"):
        view = _retriever_view(FakeIndex(), mode)
        assert view.partitions[0].weight == 1.25
        assert view.search("question", top_k=1)[0].source == f"{mode}.md"


def test_oracle_index_predictions_do_not_publish_fake_routing_scores():
    sample = _samples()[0]

    def oracle(_sample: EvaluationSample, _top_k: int) -> EvaluationPrediction:
        return EvaluationPrediction(
            predicted_intent="implementation",
            results=(),
            metadata={"oracle_route": True},
        )

    result = evaluate_strategy([sample], oracle, strategy_name="bm25", top_k=5)
    assert result["metrics"]["routing_accuracy"] is None
    assert result["metrics"]["routing_macro_f1"] is None
    assert result["metrics"]["routing_eligible_questions"] == 0


def test_formal_suite_rejects_predictor_contract_mismatch():
    class Predictor:
        evaluation_metadata = {
            "build_id": "build-1",
            "embedding_model": "model-1",
            "oracle_route": True,
            "live_verification": False,
            "answer_generation": False,
            "fail_closed": True,
        }

        def __call__(self, sample: EvaluationSample, top_k: int) -> EvaluationPrediction:
            return _predictor(sample, top_k)

    with pytest.raises(ValueError, match="incompatible with e2e"):
        run_ablation(
            _samples(),
            {"bm25": Predictor()},
            baseline="bm25",
            suite="e2e",
            validate_contract=True,
        )


def test_formal_suite_requires_every_strategy_to_declare_model_and_budget():
    class Predictor:
        def __init__(self, *, model: str, candidate_budget: int) -> None:
            self.evaluation_metadata = {
                "build_id": "build-1",
                "embedding_model": model,
                "oracle_route": True,
                "live_verification": False,
                "answer_generation": False,
                "fail_closed": True,
                "retrieval_strategy": "bm25",
                "candidate_budget_multiplier": candidate_budget,
                "partition_candidate_multiplier": 1,
                "federated_candidate_multiplier": 1,
            }

        def __call__(self, sample: EvaluationSample, top_k: int) -> EvaluationPrediction:
            return _predictor(sample, top_k)

    with pytest.raises(ValueError, match="no embedding_model"):
        run_ablation(
            _samples(),
            {"bm25": Predictor(model="", candidate_budget=3)},
            baseline="bm25",
            suite="index",
            validate_contract=True,
        )

    bm25 = Predictor(model="model-1", candidate_budget=3)
    dense = Predictor(model="model-1", candidate_budget=4)
    dense.evaluation_metadata["retrieval_strategy"] = "dense"
    with pytest.raises(ValueError, match="candidate-budget contract"):
        run_ablation(
            _samples(),
            {"bm25": bm25, "dense": dense},
            baseline="bm25",
            suite="index",
            validate_contract=True,
        )


def test_development_router_contract_is_an_explicit_regression_set():
    root = Path(__file__).resolve().parents[1]
    datasets = (
        root / "data/eval/mini_nanobot_internal.jsonl",
        root / "data/eval/official_engineering_specs.jsonl",
    )
    router = SourceIntentRouter()
    samples = [sample for path in datasets for sample in load_jsonl(path)]
    assert all(
        router.route(sample.question).intent.value == expected_route(sample)
        for sample in samples
    )
