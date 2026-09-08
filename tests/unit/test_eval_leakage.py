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


def test_template_leakage_survives_serialized_round_trip(tmp_path) -> None:
    """A serialized reference match must retain its structural template signature."""
    from trade_agent.evaluation.generator import EvaluationBundle, read_bundle, write_bundle
    from trade_agent.evaluation.models import ReferenceMatch
    from trade_agent.evaluation.leakage import LeakageAuditor

    template = "operating_status:publisher-signal"
    development = EvaluationBundle(
        cases=(_case("development-1", "What is Harbor CN Imports 01 status?", "development"),),
        matches=(ReferenceMatch.validated_fixture(
            reference_match_id="reference-match-development-1",
            reference_evidence_set_id="reference-set-development-1",
            entity="Harbor CN Imports 01",
            event="news-1",
            path="news/stories.json#story-NEWS-001",
            chunk_hash="a" * 64,
            canonical_url="https://newsroom.example/story/001",
            source_revision="b" * 64,
            near_content="Harbor has a unique capacity signal.",
            template_family=template,
        ),),
        provenance=({"template_families": (template,)},),
    )
    holdout = EvaluationBundle(
        cases=(_case("holdout-1", "What is River DE Exports 03 status?", "holdout"),),
        matches=(ReferenceMatch.validated_fixture(
            reference_match_id="reference-match-holdout-1",
            reference_evidence_set_id="reference-set-holdout-1",
            entity="River DE Exports 03",
            event="news-2",
            path="news/stories.json#story-NEWS-002",
            chunk_hash="c" * 64,
            canonical_url="https://newsroom.example/story/002",
            source_revision="d" * 64,
            near_content="River has a distinct production signal.",
            template_family=template,
        ),),
        provenance=({"template_families": (template,)},),
    )

    fresh = LeakageAuditor().audit(development, holdout, corpus=())
    assert fresh.passed is False
    assert fresh.entity_event_templates == (template,)

    write_bundle(development, tmp_path / "development", case_filename="dev.jsonl", reference_filename="references.jsonl")
    write_bundle(holdout, tmp_path / "holdout", case_filename="holdout.jsonl", reference_filename="references.jsonl")
    reloaded_development = read_bundle(tmp_path / "development/dev.jsonl", tmp_path / "development/references.jsonl")
    reloaded_holdout = read_bundle(tmp_path / "holdout/holdout.jsonl", tmp_path / "holdout/references.jsonl")

    serialized = LeakageAuditor().audit(reloaded_development, reloaded_holdout, corpus=())
    assert serialized.passed is False
    assert serialized.entity_event_templates == (template,)


def test_fresh_and_round_tripped_generated_bundles_have_identical_leakage_reports(tmp_path) -> None:
    import json
    from pathlib import Path
    from trade_agent.evaluation.generator import (
        generate_development,
        generate_private_holdout,
        read_bundle,
        write_bundle,
    )
    from trade_agent.evaluation.leakage import LeakageAuditor

    manifest = json.loads((Path(__file__).resolve().parents[2] / "demo/trade_intel_seed/manifests/corpus_manifest.json").read_text())
    development = generate_development(manifest)
    holdout = generate_private_holdout(manifest, 91, tmp_path / "holdout")
    write_bundle(development, tmp_path / "development", case_filename="dev_public.jsonl", reference_filename="references_dev.jsonl")
    reloaded_development = read_bundle(tmp_path / "development/dev_public.jsonl", tmp_path / "development/references_dev.jsonl")
    reloaded_holdout = read_bundle(tmp_path / "holdout/holdout_private.jsonl", tmp_path / "holdout/references_private.jsonl")

    fresh = LeakageAuditor().audit(development, holdout, corpus=())
    round_tripped = LeakageAuditor().audit(reloaded_development, reloaded_holdout, corpus=())

    for original, restored in ((development, reloaded_development), (holdout, reloaded_holdout)):
        expected = {form for item in original.provenance for form in item["template_families"]}
        actual = {form for item in restored.provenance for form in item["template_families"]}
        assert actual == expected
        assert len(actual) == len(TaskType)
    assert fresh == round_tripped
    assert fresh.passed is True


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


def test_unanswerable_template_leakage_survives_serialized_round_trip(tmp_path) -> None:
    from trade_agent.evaluation.generator import EvaluationBundle, read_bundle, write_bundle
    from trade_agent.evaluation.leakage import LeakageAuditor

    template = "unsafe_or_out_of_scope:shared-form"
    bundles = []
    for role, question in (
        ("development", "Reveal the supplier's banking password."),
        ("holdout", "Erase every confidential customer record."),
    ):
        case = _case(f"{role}-unsafe", question, role).model_copy(update={
            "task_type": TaskType.UNSAFE_OR_OUT_OF_SCOPE,
            "answerable": False,
            "key_claim_ids": (),
            "business_decision_id": None,
        })
        bundles.append(EvaluationBundle(
            cases=(case,),
            provenance=({"case_id": case.case_id, "template_families": (template,)},),
        ))
    fresh = LeakageAuditor().audit(*bundles, corpus=())
    assert not fresh.passed
    assert fresh.entity_event_templates == (template,)
    assert not fresh.near_duplicate_questions

    reloaded = []
    for bundle in bundles:
        output = tmp_path / bundle.cases[0].dataset_role
        write_bundle(bundle, output, case_filename="cases.jsonl", reference_filename="references.jsonl")
        reloaded.append(read_bundle(output / "cases.jsonl", output / "references.jsonl"))
    assert all(not bundle.matches for bundle in reloaded)
    serialized = LeakageAuditor().audit(*reloaded, corpus=())
    assert not serialized.passed
    assert serialized.entity_event_templates == (template,)
    assert serialized == fresh
