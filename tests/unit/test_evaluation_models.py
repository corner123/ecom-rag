from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from trade_agent.evaluation.hashing import canonical_hash, hash_file
from trade_agent.evaluation.models import (
    BusinessDecision,
    EvaluationCase,
    EvaluationSnapshot,
    PerQueryResult,
    ReferenceClaim,
    ReferenceEvidence,
    RunManifest,
    evaluation_json_schema,
)


def _hash() -> str:
    return "a" * 64


def _snapshot() -> EvaluationSnapshot:
    return EvaluationSnapshot(
        snapshot_id="snapshot-dev-20260830",
        dataset_hash=_hash(),
        reference_hash=_hash(),
        corpus_hash=_hash(),
        index_hash=_hash(),
        profile_hash=_hash(),
        model_hash=_hash(),
        prompt_hash=_hash(),
        evaluator_hash=_hash(),
        code_hash=_hash(),
    )


def test_case_references_labels_by_id_not_embedded_answer() -> None:
    """Removing ID-only label pointers or accepting gold content must break this test."""
    case = EvaluationCase.validated_fixture()

    assert case.reference_evidence_set_id
    assert case.key_claim_ids
    assert not hasattr(case, "ground_truth_context")
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate({**case.model_dump(mode="python"), "ground_truth_context": "gold"})


def test_holdout_requires_private_role() -> None:
    """Removing the holdout visibility gate must break this test."""
    with pytest.raises(ValidationError):
        EvaluationCase.validated_fixture(dataset_role="holdout", visibility="public")


def test_contracts_reject_malformed_identifiers_and_are_immutable() -> None:
    """Removing bounds, duplicate guards, or frozen config must break this test."""
    case = EvaluationCase.validated_fixture()

    with pytest.raises(ValidationError):
        EvaluationCase.model_validate({**case.model_dump(mode="python"), "case_id": "x" * 129})
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate({**case.model_dump(mode="python"), "key_claim_ids": ["claim-a", "claim-a"]})
    with pytest.raises(ValidationError):
        case.case_id = "other"  # type: ignore[misc]


def test_reference_and_decision_contracts_bind_only_ids_and_bounded_label_text() -> None:
    """Removing reference IDs or label-text bounds must break this test."""
    evidence = ReferenceEvidence(
        reference_evidence_id="reference-evidence-1",
        reference_evidence_set_id="reference-set-1",
        evidence_id="rag_" + "1" * 64,
        required=True,
    )
    claim = ReferenceClaim(
        claim_id="claim_" + "2" * 64,
        reference_evidence_set_id="reference-set-1",
        evidence_ids=(evidence.evidence_id,),
        claim_text="The supplier has a verified operating-status record.",
    )
    decision = BusinessDecision(
        business_decision_id="decision-1",
        key_claim_ids=(claim.claim_id,),
        decision_text="Prioritize verification before outreach.",
    )

    assert claim.evidence_ids == (evidence.evidence_id,)
    assert decision.key_claim_ids == (claim.claim_id,)
    with pytest.raises(ValidationError):
        ReferenceClaim.model_validate({**claim.model_dump(mode="json"), "claim_text": "x" * 4_001})


def test_snapshot_and_manifest_capture_every_frozen_input_and_backend_status() -> None:
    """Dropping a reproducibility hash or degradation status must break this test."""
    snapshot = _snapshot()
    manifest = RunManifest(
        run_id="run-20260830-001",
        snapshot=snapshot,
        dataset_hash=_hash(),
        reference_hash=_hash(),
        corpus_hash=_hash(),
        index_hash=_hash(),
        profile_hash=_hash(),
        model_hash=_hash(),
        prompt_hash=_hash(),
        evaluator_hash=_hash(),
        code_hash=_hash(),
        backend_statuses={"reranker": "degraded", "judge": "not_run"},
        degraded_components=("reranker",),
    )

    assert manifest.snapshot == snapshot
    assert manifest.backend_statuses["reranker"] == "degraded"
    assert manifest.degraded_components == ("reranker",)
    with pytest.raises(ValidationError):
        RunManifest.model_validate({**manifest.model_dump(mode="json"), "backend_statuses": {"reranker": "available"}, "degraded_components": ("reranker",)})


def test_per_query_result_keeps_recomputable_ids_and_never_embeds_gold_labels() -> None:
    """Adding gold labels to runtime traces or allowing duplicate selections must break this test."""
    result = PerQueryResult(
        run_id="run-20260830-001",
        case_id="case-dev-001",
        status="completed",
        retrieved_evidence_ids=("rag_" + "1" * 64,),
        produced_claim_ids=("claim_" + "2" * 64,),
        backend_statuses={"retrieval": "available"},
        degraded_components=(),
        latency_ms=12.5,
    )

    assert result.retrieved_evidence_ids
    with pytest.raises(ValidationError):
        PerQueryResult.model_validate({**result.model_dump(mode="json"), "gold_answer": "leak"})
    with pytest.raises(ValidationError):
        PerQueryResult.model_validate({**result.model_dump(mode="json"), "retrieved_evidence_ids": ["rag_" + "1" * 64] * 2})


def test_hashing_is_deterministic_for_models_and_file_bytes(tmp_path: Path) -> None:
    """Changing canonical serialization or byte hashing must break this test."""
    first = EvaluationCase.validated_fixture()
    second = EvaluationCase.model_validate_json(first.model_dump_json())
    path = tmp_path / "fixture.json"
    path.write_bytes(b"trade-evaluation\n")

    assert canonical_hash(first) == canonical_hash(second)
    assert hash_file(path) == hashlib.sha256(b"trade-evaluation\n").hexdigest()


def test_exported_schema_matches_the_model_contract_and_rejects_same_invalid_fixture() -> None:
    """Hand-editing the schema or loosening contract validation must break this test."""
    schema_path = Path("data/eval/trade_intel/schemas/evaluation-v1.schema.json")
    exported = json.loads(schema_path.read_text(encoding="utf-8"))

    assert exported == evaluation_json_schema()
    assert exported["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert "EvaluationCase" in exported["$defs"]
    valid = EvaluationCase.validated_fixture().model_dump(mode="json")
    assert EvaluationCase.model_validate_json(json.dumps(valid)).case_id == valid["case_id"]
    assert exported["$defs"]["EvaluationCase"]["allOf"] == [
        {
            "if": {"properties": {"dataset_role": {"const": "holdout"}}},
            "then": {"properties": {"visibility": {"const": "private"}}},
        }
    ]
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate({**valid, "visibility": "internal"})
