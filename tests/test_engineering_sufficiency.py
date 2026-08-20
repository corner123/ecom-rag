from __future__ import annotations

from rag_core.engineering.sufficiency import (
    EvidenceSufficiencyGuard,
    SufficiencyProfile,
    _hard_anchors,
)
from rag_core.retrieval.engineering import EngineeringSearchResult, SourceIntent


def _design_evidence(content: str) -> list[EngineeringSearchResult]:
    return [
        EngineeringSearchResult(
            content=content,
            source="docs/architecture/context-memory-checkpoint.md",
            corpus="internal",
            authority="design",
            metadata={
                "document_id": "design-1",
                "evidence_role": "internal_design",
            },
        )
    ]


def test_legacy_slash_profile_keeps_exact_anchor_and_rejects_split_evidence() -> None:
    query = (
        "Mini-Nanobot 的 context/checkpoint architecture 为什么同时维护 "
        "canonical history 与 model-facing projection？"
    )
    evidence = _design_evidence(
        "The context architecture keeps canonical history while checkpoint "
        "persistence also stores the model-facing projection."
    )

    guarded, warnings, reason = EvidenceSufficiencyGuard().check(
        query, SourceIntent.DESIGN, evidence
    )

    assert _hard_anchors(query) == ["context/checkpoint"]
    assert guarded is False
    assert reason == "unsupported_anchor"
    assert any("context/checkpoint" in warning for warning in warnings)


def test_split_slash_profile_grounds_natural_concepts_individually() -> None:
    query = (
        "Mini-Nanobot 的 context/checkpoint architecture 为什么同时维护 "
        "canonical history 与 model-facing projection？"
    )
    evidence = _design_evidence(
        "The context architecture keeps canonical history while checkpoint "
        "persistence also stores the model-facing projection."
    )
    profile = SufficiencyProfile.SPLIT_NATURAL_SLASH_CONCEPTS

    guarded, warnings, reason = EvidenceSufficiencyGuard(profile).check(
        query, SourceIntent.DESIGN, evidence
    )

    assert _hard_anchors(query, profile=profile) == ["context", "checkpoint"]
    assert guarded is True
    assert warnings == []
    assert reason is None


def test_unquoted_slash_concepts_still_require_every_token() -> None:
    guarded, warnings, reason = EvidenceSufficiencyGuard(
        SufficiencyProfile.SPLIT_NATURAL_SLASH_CONCEPTS
    ).check(
        "context/unknownframework architecture 为什么这样设计？",
        SourceIntent.DESIGN,
        _design_evidence("The context architecture keeps canonical history."),
    )

    assert guarded is False
    assert reason == "unsupported_anchor"
    assert any("unknownframework" in warning for warning in warnings)


def test_explicit_identifier_and_path_anchors_remain_exact() -> None:
    assert _hard_anchors("为什么使用 `context/checkpoint`？") == [
        "context/checkpoint"
    ]
    assert _hard_anchors("读取 docs/architecture/context.md") == [
        "docs/architecture/context.md"
    ]
    assert _hard_anchors("解释 QueryEngine.submit_message") == [
        "QueryEngine.submit_message"
    ]

    for query, content in (
        ("为什么使用 `context/checkpoint`？", "context and checkpoint are separate"),
        ("读取 docs/architecture/context.md", "docs architecture context md"),
        ("解释 QueryEngine.submit_message", "QueryEngine submit_message"),
    ):
        guarded, _, reason = EvidenceSufficiencyGuard().check(
            query, SourceIntent.DESIGN, _design_evidence(content)
        )
        assert guarded is False
        assert reason == "unsupported_anchor"


def test_unknown_framework_anchor_remains_strict() -> None:
    guarded, warnings, reason = EvidenceSufficiencyGuard().check(
        "什么是 someunknownframework？",
        SourceIntent.DESIGN,
        _design_evidence("This document describes a small Agent framework."),
    )

    assert guarded is False
    assert reason == "unsupported_anchor"
    assert any("someunknownframework" in warning for warning in warnings)


def test_unknown_profile_is_rejected_instead_of_silently_falling_back() -> None:
    try:
        EvidenceSufficiencyGuard("unknown-profile")
    except ValueError as exc:
        assert "unknown-profile" in str(exc)
    else:  # pragma: no cover - defensive failure message
        raise AssertionError("unknown sufficiency profile must fail closed")
