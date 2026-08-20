from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_core.engineering.index import HybridPartitionRetriever
from rag_core.evaluation.engineering import EvaluationSample, run_ablation
from rag_core.evaluation.engineering_adapter import (
    _retriever_view,
    create_retrieval_profile_predictors,
)
from rag_core.evaluation.retrieval_profiles import (
    PRODUCTION_DEFAULT_PROFILE,
    RetrievalExperimentProfile,
    build_retrieval_profile_grid,
)
from rag_core.retrieval.engineering import (
    EngineeringSearchResult,
    FederatedRetriever,
)


class RecordingRetriever:
    def __init__(self, name: str) -> None:
        self.name = name
        self.requested_top_k: list[int] = []

    def search(self, query: str, top_k: int = 5):
        self.requested_top_k.append(top_k)
        return [
            EngineeringSearchResult(
                content=query,
                source=f"{self.name}.md",
                corpus="internal",
                authority="code",
                retriever=self.name,
            )
        ]


def _fake_index():
    dense = RecordingRetriever("dense")
    bm25 = RecordingRetriever("bm25")
    hybrid = HybridPartitionRetriever(dense, bm25)
    federated = FederatedRetriever(candidate_multiplier=3, fail_open=True)
    federated.add_partition("internal", "code", hybrid, weight=1.25)
    return SimpleNamespace(federated=federated), dense, bm25, hybrid


def _sample() -> EvaluationSample:
    return EvaluationSample(
        id="implementation-1",
        question="Where is QueryEngine implemented?",
        category="implementation",
        expected_intent="implementation",
        answerable=True,
        relevant_sources=("dense.md",),
        expected_answer_points=("QueryEngine implementation",),
        source_scope="internal",
        source_revision="test",
        expected_route_label="implementation",
        primary_sources=("dense.md",),
    )


def test_production_default_view_reuses_original_hybrid_configuration():
    index, _dense, _bm25, original = _fake_index()

    view = _retriever_view(index, "hybrid")

    assert view.candidate_multiplier == 3
    assert view.fail_open is True
    assert view.partitions[0].retriever is original
    assert original.dense_weight == PRODUCTION_DEFAULT_PROFILE.dense_weight == 1.0
    assert original.bm25_weight == PRODUCTION_DEFAULT_PROFILE.bm25_weight == 1.0
    assert (
        original.candidate_multiplier
        == PRODUCTION_DEFAULT_PROFILE.partition_candidate_multiplier
        == 3
    )


def test_profile_weights_and_budgets_flow_to_both_retrieval_layers():
    index, dense, bm25, original = _fake_index()
    profile = RetrievalExperimentProfile(
        name="bm25-heavy-test",
        dense_weight=0.25,
        bm25_weight=1.5,
        partition_candidate_multiplier=2,
        federated_candidate_multiplier=4,
    )

    view = _retriever_view(index, "hybrid", profile=profile)
    configured = view.partitions[0].retriever
    view.search("checkpoint", top_k=1)

    assert configured is not original
    assert isinstance(configured, HybridPartitionRetriever)
    assert configured.dense_weight == 0.25
    assert configured.bm25_weight == 1.5
    assert configured.candidate_multiplier == 2
    assert configured.fail_open is False
    assert view.candidate_multiplier == 4
    assert view.fail_open is False
    # Federated top_k=1 -> partition top_k=4 -> dense/BM25 candidate top_k=8.
    assert dense.requested_top_k == [8]
    assert bm25.requested_top_k == [8]


def test_profile_factory_records_prediction_and_report_metadata(monkeypatch):
    index, _dense, _bm25, _original = _fake_index()
    index.stats = lambda: {
        "build_id": "build-test",
        "embedding_model": "embedding-test",
        "document_count": 2,
        "partitions": [],
    }
    observed_runtime_backends: list[str | None] = []

    def fake_load(*_args, **kwargs):
        observed_runtime_backends.append(kwargs.get("runtime_backend"))
        return index

    monkeypatch.setattr(
        "rag_core.evaluation.engineering_adapter.EngineeringIndex.load",
        fake_load,
    )
    monkeypatch.setattr(
        "rag_core.evaluation.engineering_adapter._assert_current_build",
        lambda _index: None,
    )
    profile = RetrievalExperimentProfile(
        name="bm25-heavy-candidate",
        dense_weight=0.5,
        bm25_weight=1.0,
        partition_candidate_multiplier=2,
        federated_candidate_multiplier=2,
    )

    predictors = create_retrieval_profile_predictors(
        index_root="unused",
        profiles=[profile],
        candidate_budget_multiplier=1,
    )
    predictor = predictors[profile.name]
    prediction = predictor(_sample(), top_k=1)
    report = run_ablation(
        [_sample()],
        predictors,
        top_k=1,
        baseline=profile.name,
        suite="index",
    )

    assert observed_runtime_backends == ["faiss"]
    assert predictor.evaluation_metadata["development_grid"] is True
    assert predictor.evaluation_metadata["retrieval_profile"] == profile.name
    assert predictor.evaluation_metadata["hybrid_dense_weight"] == 0.5
    assert predictor.evaluation_metadata["hybrid_bm25_weight"] == 1.0
    assert predictor.evaluation_metadata["partition_candidate_multiplier"] == 2
    assert predictor.evaluation_metadata["federated_candidate_multiplier"] == 2
    assert predictor.evaluation_metadata["candidate_budget_multiplier"] == 1
    assert prediction.metadata["retrieval_profile"] == profile.name
    assert prediction.metadata["retrieval_profile_schema"].endswith("/v1")
    assert (
        report["strategy_metadata"][profile.name]["retrieval_profile"]
        == profile.name
    )
    assert (
        report["strategy_metadata"][profile.name]["hybrid_dense_weight"]
        == 0.5
    )


def test_grid_is_deterministic_and_does_not_claim_a_winner():
    profiles = build_retrieval_profile_grid(
        dense_weights=(1.0, 0.5),
        bm25_weights=(1.0,),
        candidate_multipliers=(1, 3),
    )

    assert [profile.name for profile in profiles] == [
        "hybrid-dw1-bw1-cm1",
        "hybrid-dw1-bw1-cm3",
        "hybrid-dw0p5-bw1-cm1",
        "hybrid-dw0p5-bw1-cm3",
    ]
    assert all("winner" not in profile.name for profile in profiles)


@pytest.mark.parametrize(
    "profile",
    [
        lambda: RetrievalExperimentProfile(name="Invalid Name"),
        lambda: RetrievalExperimentProfile(
            name="zero", dense_weight=0, bm25_weight=0
        ),
        lambda: RetrievalExperimentProfile(
            name="bad-cm", partition_candidate_multiplier=0
        ),
    ],
)
def test_invalid_profiles_are_rejected(profile):
    with pytest.raises(ValueError):
        profile()


def test_view_rejects_ambiguous_or_non_hybrid_profile_use():
    index, _dense, _bm25, _original = _fake_index()
    profile = RetrievalExperimentProfile(name="candidate")

    with pytest.raises(ValueError, match="only valid for hybrid"):
        _retriever_view(index, "dense", profile=profile)
    with pytest.raises(ValueError, match="cannot be supplied together"):
        _retriever_view(
            index,
            "hybrid",
            profile=profile,
            candidate_multiplier=2,
        )
