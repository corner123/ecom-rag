"""Tests for release gates between development and private holdout data."""

from __future__ import annotations

from datetime import date

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
