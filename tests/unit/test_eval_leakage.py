"""Tests for release gates between development and private holdout data."""

from __future__ import annotations

from datetime import date
import sys

from trade_agent.evaluation.models import EvaluationCase, TaskType


def _case(case_id: str, question: str, role: str) -> EvaluationCase:
    return EvaluationCase.validated_fixture(
        case_id=case_id,
        question=question,
        task_type=TaskType.EXACT_COMPANY_LOOKUP,
        dataset_role=role,
        visibility="private" if role == "holdout" else "public",
        as_of_date=date(2026, 8, 30),
        reference_evidence_set_id=f"reference-set-{case_id}",
    )


def test_near_duplicate_holdout_question_is_detected() -> None:
    from trade_agent.evaluation.generator import EvaluationBundle
    from trade_agent.evaluation.leakage import LeakageAuditor

    development = EvaluationBundle(cases=(_case("development-1", "What is the verified operating status of Harbor CN Imports 01?", "development"),))
    holdout = EvaluationBundle(cases=(_case("holdout-1", "Please tell me Harbor CN Imports 01's verified operating status.", "holdout"),))

    report = LeakageAuditor().audit(development, holdout, corpus=())

    assert report.passed is False
    assert report.near_duplicate_questions


def test_reference_label_contamination_in_indexed_content_is_detected() -> None:
    from trade_agent.evaluation.generator import EvaluationBundle
    from trade_agent.evaluation.leakage import LeakageAuditor

    development = EvaluationBundle(cases=(_case("development-1", "Which supplier serves HS 010121?", "development"),))
    holdout = EvaluationBundle(cases=(_case("holdout-1", "Which supplier serves HS 020130?", "holdout"),))
    indexed_content = ({"content": "Gold answer: preferred supplier", "reference_label": "preferred supplier"},)

    report = LeakageAuditor().audit(development, holdout, indexed_content)

    assert report.passed is False
    assert report.reference_label_contamination


def test_similar_content_with_distinct_hashes_and_shared_template_are_detected() -> None:
    from trade_agent.evaluation.generator import EvaluationBundle
    from trade_agent.evaluation.leakage import LeakageAuditor

    development = EvaluationBundle(
        cases=(_case("development-1", "What is the status of Harbor CN Imports 01?", "development"),),
        provenance=({"near_contents": ("Harbor exporter reported a 25 percent capacity increase this quarter.",), "template_families": ("operating_status:publisher-signal",)},),
    )
    holdout = EvaluationBundle(
        cases=(_case("holdout-1", "What is the status of River DE Exports 03?", "holdout"),),
        provenance=({"near_contents": ("River exporter reported a 27 percent capacity increase this quarter.",), "template_families": ("operating_status:publisher-signal",)},),
    )

    report = LeakageAuditor().audit(development, holdout, corpus=())

    assert report.passed is False
    assert report.near_chunk_hashes
    assert report.entity_event_templates


def test_approved_synthetic_boilerplate_cannot_approve_copied_substantive_content() -> None:
    from trade_agent.evaluation.generator import EvaluationBundle, ReferenceMatch
    from trade_agent.evaluation.leakage import LeakageAuditor

    copied = "SYNTHETIC DEMONSTRATION ONLY — FICTIONAL DATA; NOT FOR PRODUCTION USE. Capacity expanded by 40 percent."
    development = EvaluationBundle(
        cases=(_case("development-1", "What is Harbor CN Imports 01 status?", "development"),),
        matches=(ReferenceMatch.validated_fixture(
            reference_match_id="reference-match-development-1", reference_evidence_set_id="reference-set-development-1",
            entity="Harbor CN Imports 01", event="news-1", near_content=copied, approved_synthetic_template=True,
        ),),
    )
    holdout = EvaluationBundle(
        cases=(_case("holdout-1", "What is River DE Exports 03 status?", "holdout"),),
        matches=(ReferenceMatch.validated_fixture(
            reference_match_id="reference-match-holdout-1", reference_evidence_set_id="reference-set-holdout-1",
            entity="River DE Exports 03", event="news-2", near_content=copied, approved_synthetic_template=True,
        ),),
    )

    report = LeakageAuditor().audit(development, holdout, corpus=())

    assert report.passed is False
    assert report.near_chunk_hashes


def test_generated_partitions_use_distinct_actual_template_forms(tmp_path) -> None:
    import json
    from pathlib import Path
    from trade_agent.evaluation.generator import generate_development, generate_private_holdout

    manifest = json.loads((Path(__file__).resolve().parents[2] / "demo/trade_intel_seed/manifests/corpus_manifest.json").read_text())
    development = generate_development(manifest)
    holdout = generate_private_holdout(manifest, 91, tmp_path / "holdout")
    dev_forms = {form for item in development.provenance for form in item["template_families"]}
    holdout_forms = {form for item in holdout.provenance for form in item["template_families"]}

    assert dev_forms.isdisjoint(holdout_forms)


def test_validator_passes_indexable_source_content_to_contamination_gate(monkeypatch, tmp_path) -> None:
    from trade_agent.evaluation.generator import EvaluationBundle
    from trade_agent.evaluation.models import ReferenceClaim
    from scripts import validate_trade_eval

    development_case = _case("development-1", "Which supplier serves HS 010121?", "development")
    development = EvaluationBundle(
        cases=(development_case,),
        claims=(ReferenceClaim(
            claim_id="claim_" + "1" * 64,
            reference_evidence_set_id=development_case.reference_evidence_set_id,
            evidence_ids=(), claim_text="preferred supplier",
        ),),
    )
    holdout = EvaluationBundle(cases=(_case("holdout-1", "Which supplier serves HS 020130?", "holdout"),))
    bundles = iter((development, holdout))
    monkeypatch.setattr(validate_trade_eval, "read_bundle", lambda *_: next(bundles))
    monkeypatch.setattr(validate_trade_eval, "indexed_corpus_content", lambda _: ({"content": "preferred supplier"},))
    monkeypatch.setattr(sys, "argv", ["validate_trade_eval", "--dev", str(tmp_path), "--holdout", str(tmp_path)])

    assert validate_trade_eval.main() == 1
