from __future__ import annotations

from dataclasses import dataclass

from trade_agent.evaluation.generation_metrics import evidence_coverage, faithfulness


@dataclass(frozen=True)
class StubClaim:
    claim_id: str
    status: str
    evidence_ids: tuple[str, ...] = ()
    factual: bool | None = None


@dataclass(frozen=True)
class StubAnswer:
    claims: tuple[StubClaim, ...]
    refusal_reason: str | None = None


@dataclass(frozen=True)
class StubEvidence:
    evidence_id: str


@dataclass(frozen=True)
class StubGuard:
    accepted: bool
    claims: tuple[StubClaim, ...]
    refusal_reason: str | None = None


def test_evidence_coverage_counts_factual_claims_only() -> None:
    answer = StubAnswer(
        claims=(
            StubClaim("C1", "supported", ("E1",), factual=True),
            StubClaim("C2", "analysis", (), factual=False),
            StubClaim("C3", "insufficient", (), factual=True),
        )
    )

    assert evidence_coverage(answer) == 0.5


def test_evidence_coverage_excludes_analysis_without_explicit_factual_flag() -> None:
    answer = StubAnswer(claims=(StubClaim("C1", "analysis"),))

    assert evidence_coverage(answer) == 0.0


def test_evidence_coverage_zero_factual_claims_is_not_perfect() -> None:
    assert evidence_coverage(StubAnswer(claims=())) == 0.0


def test_faithfulness_counts_invalid_citation_as_unsupported() -> None:
    supported = StubClaim("C1", "supported", ("E1",), factual=True)
    mutated = StubClaim("C2", "supported", ("E-MUTATED",), factual=True)
    analysis = StubClaim("C3", "analysis", (), factual=False)
    answer = StubAnswer(claims=(supported, mutated, analysis))
    guard = StubGuard(accepted=True, claims=(supported, mutated, analysis))

    result = faithfulness(answer, (StubEvidence("E1"),), guard)

    assert result.faithfulness == 0.5
    assert result.policy == "scored"
    assert result.checked_factual_claim_ids == ("C1", "C2")
    assert result.supported_factual_claim_ids == ("C1",)
    assert result.unsupported_factual_claim_ids == ("C2",)
    assert result.invalid_citation_ids == ("E-MUTATED",)
    assert result.checked_count == 2
    assert result.supported_count == 1


def test_faithfulness_requires_guard_to_retain_supported_claim() -> None:
    first = StubClaim("C1", "supported", ("E1",), factual=True)
    removed = StubClaim("C2", "supported", ("E2",), factual=True)
    answer = StubAnswer(claims=(first, removed))
    guard = StubGuard(accepted=True, claims=(first,))

    result = faithfulness(answer, (StubEvidence("E1"), StubEvidence("E2")), guard)

    assert result.faithfulness == 0.5
    assert result.unsupported_factual_claim_ids == ("C2",)


def test_refusal_has_explicit_unscored_zero_count_policy() -> None:
    answer = StubAnswer(claims=(), refusal_reason="evidence_insufficient")
    guard = StubGuard(accepted=False, claims=(), refusal_reason="evidence_insufficient")

    result = faithfulness(answer, (), guard)

    assert result.faithfulness is None
    assert result.policy == "not_scored_no_checked_answer"
    assert result.checked_count == 0
    assert result.supported_count == 0
    assert result.checked_factual_claim_ids == ()


def test_accepted_analysis_only_answer_has_explicit_zero_denominator_policy() -> None:
    analysis = StubClaim("C1", "analysis", (), factual=False)
    answer = StubAnswer(claims=(analysis,))
    guard = StubGuard(accepted=True, claims=(analysis,))

    result = faithfulness(answer, (), guard)

    assert result.faithfulness is None
    assert result.policy == "not_scored_no_factual_claims"
    assert result.checked_count == 0
