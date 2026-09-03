from __future__ import annotations

from dataclasses import dataclass

import pytest

from trade_agent.retrieval.profiles import RetrievalProfile, load_retrieval_profile


@dataclass(frozen=True)
class FakeHit:
    chunk_id: str
    source_type: str = "customs_profile"
    fact_type: str = "trade_activity"
    fact_value: str = "operating"

    @property
    def metadata(self):
        return self

    @property
    def record(self):
        return self


def profile_fixture(rrf_k: int = 60) -> RetrievalProfile:
    return RetrievalProfile(
        profile_id="balanced-v1",
        version="trade-source-profiles-v1",
        rrf_k=rrf_k,
        retriever_weights={"dense": 0.7, "bm25": 0.3},
        source_fact_priors={"customs_profile:trade_activity": 0.9},
        default_prior=0.5,
        candidate_limits={"dense": 10, "bm25": 10, "output": 10},
    )


def rankings_fixture():
    return {
        "dense": [FakeHit("same"), FakeHit("bm25_only"), FakeHit("dense_second")],
        "bm25": [FakeHit("bm25_only"), FakeHit("same"), FakeHit("bm25_second")],
    }


def test_weighted_rrf_uses_rank_not_raw_score() -> None:
    from trade_agent.retrieval.fusion import weighted_rrf

    result = weighted_rrf(rankings_fixture(), profile_fixture(rrf_k=60))
    assert result[0].chunk_id == "same"
    assert result[0].components["dense"].rank == 1
    assert result[0].components["bm25"].rank == 2
    expected = 0.9 * (
        0.7 / (60 + 1)
        + 0.3 / (60 + 2)
    )
    assert result[0].score == pytest.approx(expected)
    assert result[0].relevance_subtotal == pytest.approx(
        0.7 / (60 + 1) + 0.3 / (60 + 2)
    )
    assert result[0].source_prior == 0.9
    assert result[0].prior_contribution == pytest.approx(expected)


def test_missing_required_retriever_is_rejected() -> None:
    from trade_agent.retrieval.fusion import weighted_rrf

    with pytest.raises(ValueError, match="missing required retriever"):
        weighted_rrf({"dense": rankings_fixture()["dense"]}, profile_fixture())


def test_unknown_retriever_and_duplicate_candidates_are_rejected() -> None:
    from trade_agent.retrieval.fusion import weighted_rrf

    with pytest.raises(ValueError, match="unknown retriever"):
        weighted_rrf(
            {"dense": [FakeHit("same")], "bm25": [], "vector": [FakeHit("same")]},
            profile_fixture(),
        )
    with pytest.raises(ValueError, match="duplicate"):
        weighted_rrf(
            {"dense": [FakeHit("same"), FakeHit("same")], "bm25": []},
            profile_fixture(),
        )


def test_source_prior_does_not_resolve_conflict() -> None:
    from trade_agent.retrieval.fusion import weighted_rrf

    rankings = {
        "dense": [
            FakeHit("operating", source_type="official_website", fact_type="company_status", fact_value="operating"),
            FakeHit("closed", source_type="customs_profile", fact_type="company_status", fact_value="closed"),
        ],
        "bm25": [
            FakeHit("closed", source_type="customs_profile", fact_type="company_status", fact_value="closed"),
            FakeHit("operating", source_type="official_website", fact_type="company_status", fact_value="operating"),
        ],
    }
    profile = RetrievalProfile(
        profile_id="balanced-v1",
        version="trade-source-profiles-v1",
        rrf_k=60,
        retriever_weights={"dense": 0.5, "bm25": 0.5},
        source_fact_priors={},
        default_prior=1.0,
        candidate_limits={"output": 10},
    )
    hits = weighted_rrf(rankings, profile)
    assert {hit.record.fact_value for hit in hits[:2]} == {"operating", "closed"}
    assert hits[0].chunk_id == "closed"
    assert hits[1].chunk_id == "operating"


def test_default_profile_loads_from_versioned_yaml() -> None:
    profile = load_retrieval_profile()
    assert profile.profile_id == "balanced-v1"
    assert profile.version == "trade-source-profiles-v1"
    assert profile.rrf_k == 60
    assert profile.retriever_weights["dense"] == 0.7
    assert profile.prior("customs_profile", "trade_activity") == 0.95
    assert profile.prior("social", "market_signal") == 0.35


def test_yaml_profile_rejects_out_of_bounds_prior(tmp_path) -> None:
    import yaml
    from trade_agent.retrieval.profiles import load_retrieval_profile

    path = tmp_path / "profiles.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": "bad-v1",
                "default_profile": "bad",
                "profiles": {
                    "bad": {
                        "rrf_k": 60,
                        "retriever_weights": {"dense": 0.7, "bm25": 0.3},
                        "source_fact_priors": {"customs_profile:trade_activity": 1.2},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="profile"):
        load_retrieval_profile(path)
