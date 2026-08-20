from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from rag_core.evaluation.response_dataset import (
    DatasetLeakageError,
    ResponseEvaluationSample,
    assert_no_normalized_duplicates,
    dataset_sha256,
    file_sha256,
    load_response_jsonl,
    migrate_v2_sample_to_draft,
    migrate_v2_samples_stratified,
    normalize_question,
    validate_dataset_separation,
)


def _payload(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "id": "R3-001",
        "question": "QueryEngine 为什么使用显式循环？",
        "category": "design_architecture",
        "answerable": True,
        "expected_route": "design",
        "reference_answer": "显式循环让控制流直接可读并便于测试。",
        "reference_claims": [
            {
                "claim_id": "C1",
                "text": "控制流直接可读",
                "required": True,
                "evidence_sources": ["docs/adr/0001-loop.md"],
                "locator": "docs/adr/0001-loop.md#decision",
            }
        ],
        "difficulty": "medium",
        "query_variant": "canonical",
        "dataset_role": "development_response",
        "source_revision": "build=test;commit=abc123",
        "reference_review_status": "human_reviewed",
        "label_version": "response-eval/v3",
    }
    value.update(overrides)
    return value


def _sample(**overrides: object) -> ResponseEvaluationSample:
    return ResponseEvaluationSample.from_mapping(_payload(**overrides))


def _write_jsonl(path: Path, *payloads: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in payloads),
        encoding="utf-8",
    )


def _v2(
    sample_id: str,
    *,
    question: str | None = None,
    route: str = "implementation",
    answerable: bool = True,
    points: list[str] | None = None,
    sources: list[str] | None = None,
) -> dict[str, object]:
    if points is None:
        points = ["第一条事实", "第二条事实"]
    if sources is None:
        sources = ["mini_nanobot/core/query.py"] if answerable else []
    return {
        "id": sample_id,
        "question": question or f"问题 {sample_id}",
        "category": "fixture",
        "answerable": answerable,
        "expected_answer_points": points,
        "primary_sources": sources,
        "relevant_sources": sources,
        "source_revision": "build=fixture;commit=abc123",
        "expected_route": route,
        "label_version": "engineering-eval/v2",
    }


def test_formal_loader_accepts_reviewed_v3_and_hashes_are_stable(tmp_path: Path):
    first = _payload()
    second = _payload(
        id="R3-002",
        question="LangGraph 的持久化边界是什么？",
        reference_answer="checkpointer 保存线程状态，store 保存跨线程数据。",
        reference_claims=[
            {
                "claim_id": "C1",
                "text": "checkpointer 保存线程状态",
                "required": True,
                "evidence_sources": ["https://docs.langchain.com/langgraph/persistence"],
                "locator": "https://docs.langchain.com/langgraph/persistence#checkpoints",
            }
        ],
        expected_route="official",
    )
    path = tmp_path / "response.jsonl"
    _write_jsonl(path, first, second)

    samples = load_response_jsonl(path)

    assert [sample.id for sample in samples] == ["R3-001", "R3-002"]
    assert samples[0].evidence_sources == ("docs/adr/0001-loop.md",)
    assert file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert dataset_sha256(samples) == dataset_sha256(list(reversed(samples)))
    assert len(dataset_sha256(samples)) == 64


def test_contract_is_strict_and_formal_loader_rejects_unreviewed(tmp_path: Path):
    with pytest.raises(ValueError, match="unsupported fields"):
        ResponseEvaluationSample.from_mapping(_payload(unexpected="leak"))

    missing_sources = _payload()
    missing_sources["reference_claims"] = [
        {
            "claim_id": "C1",
            "text": "没有来源",
            "required": True,
            "evidence_sources": [],
            "locator": "docs/example.md#missing",
        }
    ]
    with pytest.raises(ValueError, match="require at least one evidence source"):
        ResponseEvaluationSample.from_mapping(missing_sources)

    path = tmp_path / "unreviewed.jsonl"
    _write_jsonl(path, _payload(reference_review_status="unreviewed"))
    with pytest.raises(ValueError, match="rejects unreviewed sample"):
        load_response_jsonl(path)
    assert (
        load_response_jsonl(path, formal=False)[0].reference_review_status
        == "unreviewed"
    )


def test_normalized_duplicates_are_rejected_within_and_across_splits():
    development = _sample(question=" LangChain，是什么？ ")
    duplicate = _sample(
        id="H-001",
        question="langchain是什么",
        dataset_role="private_holdout",
    )

    assert normalize_question(development.question) == normalize_question(
        duplicate.question
    )
    with pytest.raises(DatasetLeakageError, match="normalized duplicate question"):
        assert_no_normalized_duplicates([development], [duplicate])
    with pytest.raises(DatasetLeakageError, match="normalized duplicate question"):
        validate_dataset_separation([development], [duplicate])

    wrong_role = replace(development, dataset_role="private_holdout")
    with pytest.raises(DatasetLeakageError, match="non-development role"):
        validate_dataset_separation([wrong_role], [])


def test_loader_rejects_dataset_file_inside_indexable_corpus(tmp_path: Path):
    corpus_root = tmp_path / "corpus"
    dataset_path = corpus_root / "labels" / "response.jsonl"
    _write_jsonl(dataset_path, _payload())

    with pytest.raises(DatasetLeakageError, match="inside an indexable corpus root"):
        load_response_jsonl(
            dataset_path,
            indexable_corpus_roots=[corpus_root],
        )

    outside = tmp_path / "evaluation" / "response.jsonl"
    _write_jsonl(outside, _payload())
    assert load_response_jsonl(
        outside,
        indexable_corpus_roots=[corpus_root],
    )[0].id == "R3-001"


def test_v2_migration_is_explicitly_a_non_formal_draft(tmp_path: Path):
    draft = migrate_v2_sample_to_draft(
        _v2("OLD-001"), dataset_role="development_response"
    )

    assert draft.reference_answer == "- 第一条事实\n- 第二条事实"
    assert [claim.claim_id for claim in draft.reference_claims] == ["C1", "C2"]
    assert all(
        claim.evidence_sources == ("mini_nanobot/core/query.py",)
        for claim in draft.reference_claims
    )
    assert all(
        claim.locator.startswith("v2-label:OLD-001:")
        for claim in draft.reference_claims
    )
    assert draft.reference_review_status == "unreviewed"
    assert draft.is_v2_migration_draft is True

    path = tmp_path / "draft.jsonl"
    _write_jsonl(path, draft.to_mapping())
    with pytest.raises(ValueError, match="rejects unreviewed sample"):
        load_response_jsonl(path)
    assert load_response_jsonl(path, formal=False) == [draft]

    # Flipping only the review flag is not enough: sentinel locators must be
    # replaced with reviewed evidence locators as part of human validation.
    superficially_reviewed = draft.to_mapping()
    superficially_reviewed["reference_review_status"] = "independent_agent_reviewed"
    _write_jsonl(path, superficially_reviewed)
    with pytest.raises(ValueError, match="rejects v2 draft locators"):
        load_response_jsonl(path)

    # Answerability and routing are separate labels.  A design question may be
    # correctly routed to design documents and still require refusal because
    # those documents do not contain the requested empirical evidence.
    design_boundary = migrate_v2_sample_to_draft(
        _v2(
            "OLD-BOUNDARY",
            route="design",
            answerable=False,
            points=["仓库没有该实验结果"],
        ),
        dataset_role="development_response",
    )
    assert design_boundary.answerable is False
    assert design_boundary.expected_route == "design"
    assert design_boundary.query_variant == "boundary"


def test_stratified_migration_is_deterministic_and_input_order_independent():
    values = [
        _v2("I-1", route="implementation"),
        _v2("I-2", route="implementation"),
        _v2("D-1", route="design"),
        _v2(
            "C-1",
            route="comparison",
            points=["a", "b", "c", "d"],
            sources=["a.md", "b.md", "c.md"],
        ),
        _v2(
            "B-1",
            route="out_of_scope",
            answerable=False,
            points=["现有证据不足"],
        ),
    ]

    selected = migrate_v2_samples_stratified(
        values,
        sample_size=4,
        dataset_role="development_response",
        seed=20260813,
    )
    reversed_selected = migrate_v2_samples_stratified(
        list(reversed(values)),
        sample_size=4,
        dataset_role="development_response",
        seed=20260813,
    )

    assert [sample.id for sample in selected] == [
        sample.id for sample in reversed_selected
    ]
    assert len({sample.expected_route for sample in selected}) == 4
    assert all(
        sample.reference_review_status == "unreviewed" for sample in selected
    )
    assert any(sample.query_variant == "boundary" for sample in selected)
    assert any(sample.query_variant == "multi_hop" for sample in selected)
