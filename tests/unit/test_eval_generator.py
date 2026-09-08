"""Tests for deterministic, label-separated trade evaluation generation."""

from __future__ import annotations

import json
from pathlib import Path

from trade_agent.evaluation.models import TaskType


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "demo/trade_intel_seed/manifests/corpus_manifest.json"


def test_development_covers_all_trade_task_families_and_keeps_labels_in_references() -> None:
    from trade_agent.evaluation.generator import REQUIRED_TASK_TYPES, generate_development

    bundle = generate_development(json.loads(MANIFEST.read_text(encoding="utf-8")))

    assert len(bundle.cases) >= 36
    assert REQUIRED_TASK_TYPES == set(TaskType)
    assert REQUIRED_TASK_TYPES <= {case.task_type for case in bundle.cases}
    assert any(not case.answerable for case in bundle.cases)
    assert all(case.dataset_role == "development" and case.visibility == "public" for case in bundle.cases)
    assert all("claim_text" not in case.model_dump() for case in bundle.cases)
    assert {claim.claim_id for claim in bundle.claims} >= {
        claim_id for case in bundle.cases for claim_id in case.key_claim_ids
    }
    assert not bundle.evidence
    assert bundle.matches
    assert {decision.business_decision_id for decision in bundle.decisions} == {
        case.business_decision_id for case in bundle.cases if case.business_decision_id
    }
    assert all(match.runtime_evidence_id is None for match in bundle.matches)

    operating_claim_ids = {claim_id for case in bundle.cases if case.task_type is TaskType.OPERATING_STATUS for claim_id in case.key_claim_ids}
    operating_claims = [claim.claim_text for claim in bundle.claims if claim.claim_id in operating_claim_ids]
    assert any("actual reported signal" in claim for claim in operating_claims)


def test_private_holdout_is_disjoint_and_written_only_to_requested_path(tmp_path: Path) -> None:
    from trade_agent.evaluation.generator import generate_development, generate_private_holdout

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    development = generate_development(manifest)
    output = tmp_path / "holdout"

    holdout = generate_private_holdout(manifest, secret_seed=91, output=output)

    assert output.joinpath("holdout_private.jsonl").is_file()
    assert all(case.dataset_role == "holdout" and case.visibility == "private" for case in holdout.cases)
    for field in ("entity", "event", "chunk_hash", "canonical_url", "source_revision", "reference_match_id"):
        assert {getattr(item, field) for item in development.matches}.isdisjoint(
            getattr(item, field) for item in holdout.matches
        )


def test_private_holdout_secret_changes_selected_private_material(tmp_path: Path) -> None:
    from trade_agent.evaluation.generator import generate_private_holdout

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    first = generate_private_holdout(manifest, secret_seed=91, output=tmp_path / "first")
    second = generate_private_holdout(manifest, secret_seed=92, output=tmp_path / "second")

    assert {match.chunk_hash for match in first.matches} != {match.chunk_hash for match in second.matches}
