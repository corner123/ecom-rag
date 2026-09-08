"""Auditable development/holdout leakage gates for trade evaluation data."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit

from trade_agent.evaluation.generator import EvaluationBundle
from trade_agent.schemas.source import content_sha256


_WORD = re.compile(r"[a-z0-9]+")
_ENTITY = re.compile(r"\b[A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*){1,3} \d{2}\b", re.IGNORECASE)
_EVENT = re.compile(r"\b(?:\d{6}|\d{4}-\d{2}|\d{4})\b")
_SYNTHETIC_BOILERPLATE = (
    "synthetic demonstration only fictional data not for production use",
    "this fictional report describes a demo only market signal",
)
_NEAR_TEMPLATE_WORDS = frozenset({
    "reported", "report", "total", "trade", "amount", "usd", "for", "hs", "in", "the", "a", "an",
    "lists", "list", "under", "product", "synthetic", "from", "of", "and", "marketplace",
})


@dataclass(frozen=True)
class LeakageReport:
    normalized_questions: tuple[str, ...] = ()
    exact_chunk_hashes: tuple[str, ...] = ()
    near_chunk_hashes: tuple[str, ...] = ()
    approved_near_chunk_hashes: tuple[str, ...] = ()
    canonical_urls: tuple[str, ...] = ()
    source_revisions: tuple[str, ...] = ()
    reference_ids: tuple[str, ...] = ()
    entity_event_templates: tuple[str, ...] = ()
    reference_label_contamination: tuple[str, ...] = ()

    @property
    def near_duplicate_questions(self) -> tuple[str, ...]:
        return self.normalized_questions

    @property
    def passed(self) -> bool:
        return not any((self.normalized_questions, self.exact_chunk_hashes, self.near_chunk_hashes,
                        self.canonical_urls, self.source_revisions, self.reference_ids,
                        self.entity_event_templates, self.reference_label_contamination))


def _normalise(value: str) -> str:
    return " ".join(_WORD.findall(value.casefold()))


def _tokens(value: str) -> set[str]:
    return set(_WORD.findall(value.casefold()))


def _near(left: str, right: str) -> bool:
    a, b = _tokens(left), _tokens(right)
    return bool(a and b) and len(a & b) / len(a | b) >= 0.45


def _near_content(left: str, right: str) -> bool:
    """Detect substantive source-text rewrites even when content hashes differ."""
    left_signal = _signal_tokens(_substantive_content(left))
    right_signal = _signal_tokens(_substantive_content(right))
    return bool(left_signal and right_signal) and len(left_signal & right_signal) / len(left_signal | right_signal) >= 0.55


def _substantive_content(value: str) -> str:
    normalized = _normalise(value)
    for boilerplate in _SYNTHETIC_BOILERPLATE:
        normalized = normalized.replace(boilerplate, " ")
    return " ".join(normalized.split())


def _signal_tokens(value: str) -> set[str]:
    return {
        token for token in _tokens(value)
        if token not in _NEAR_TEMPLATE_WORDS and not token.isdigit()
    }


def _entities(question: str) -> set[str]:
    values = {value.casefold() for value in _ENTITY.findall(question)}
    values.update(value.casefold() for value in re.findall(r"\b[a-z]+ \d{2}\b", question, flags=re.IGNORECASE))
    return values


def _url(value: str) -> str:
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme.casefold(), parsed.netloc.casefold(), parsed.path.rstrip("/"), parsed.query, ""))


def _iter_records(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for nested in value.values():
            yield from _iter_records(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_records(nested)


def _bundle_values(bundle: EvaluationBundle, key: str) -> set[str]:
    values: set[str] = set()
    for item in bundle.provenance:
        value = item.get(key, ())
        if isinstance(value, str):
            values.add(value)
        else:
            values.update(str(part) for part in value)
    return values


def _match_values(bundle: EvaluationBundle, key: str) -> set[str]:
    return {str(getattr(match, key)) for match in bundle.matches}


def _contents(bundle: EvaluationBundle) -> set[str]:
    return _match_values(bundle, "near_content") or _bundle_values(bundle, "near_contents")


def _approved_contents(bundle: EvaluationBundle) -> set[str]:
    return {match.near_content for match in bundle.matches if match.approved_synthetic_template}


class LeakageAuditor:
    """Reject all measurable public/private answer, source, and query overlap."""

    def audit(self, dev: EvaluationBundle, holdout: EvaluationBundle, corpus: Any) -> LeakageReport:
        dev_questions = {_normalise(case.question): case.case_id for case in dev.cases}
        holdout_questions = {_normalise(case.question): case.case_id for case in holdout.cases}
        question_hits = sorted(
            f"{dev_id}:{holdout_id}" for question, dev_id in dev_questions.items()
            for holdout_question, holdout_id in holdout_questions.items()
            if question == holdout_question or (
                _near(question, holdout_question) and _entities(question) & _entities(holdout_question)
            )
        )
        def values(bundle: EvaluationBundle, provenance_key: str, match_key: str) -> set[str]:
            return _bundle_values(bundle, provenance_key) or _match_values(bundle, match_key)
        exact_hashes = sorted(values(dev, "chunk_hashes", "chunk_hash") & values(holdout, "chunk_hashes", "chunk_hash"))
        dev_content, holdout_content = _contents(dev), _contents(holdout)
        approved_dev, approved_holdout = _approved_contents(dev), _approved_contents(holdout)
        near_pairs = [
            (left, right) for left in dev_content for right in holdout_content
            if _near_content(left, right) or (
                not _substantive_content(left) and not _substantive_content(right) and _near(left, right)
            )
        ]
        approved_pairs = [
            (left, right) for left, right in near_pairs
            if left in approved_dev and right in approved_holdout
            and not _substantive_content(left) and not _substantive_content(right)
        ]
        near_hashes = sorted(
            f"{content_sha256(left)[:16]}:{content_sha256(right)[:16]}"
            for left, right in near_pairs if (left, right) not in approved_pairs
        )
        approved_near = sorted(
            f"{content_sha256(left)[:16]}:{content_sha256(right)[:16]}"
            for left, right in approved_pairs
        )
        urls = sorted({_url(value) for value in values(dev, "canonical_urls", "canonical_url")} &
                      {_url(value) for value in values(holdout, "canonical_urls", "canonical_url")})
        revisions = sorted(values(dev, "source_revisions", "source_revision") & values(holdout, "source_revisions", "source_revision"))
        dev_refs = {match.reference_match_id for match in dev.matches} | {match.reference_evidence_set_id for match in dev.matches} | {item.claim_id for item in dev.claims}
        holdout_refs = {match.reference_match_id for match in holdout.matches} | {match.reference_evidence_set_id for match in holdout.matches} | {item.claim_id for item in holdout.claims}
        refs = sorted(dev_refs & holdout_refs)
        def template_values(bundle: EvaluationBundle) -> set[str]:
            return _bundle_values(bundle, "template_families") | _match_values(bundle, "template_family")
        templates = sorted(template_values(dev) & template_values(holdout))
        labels = {claim.claim_text for claim in (*dev.claims, *holdout.claims)}
        contamination: set[str] = set()
        for record in _iter_records(corpus):
            content = str(record.get("content", record.get("text", "")))
            explicit = record.get("reference_label") or record.get("gold_answer") or record.get("answer_label")
            if explicit:
                contamination.add(str(explicit))
            for label in labels:
                if label and label.casefold() in content.casefold():
                    contamination.add(label)
        return LeakageReport(tuple(question_hits), tuple(exact_hashes), tuple(near_hashes), tuple(approved_near),
                             tuple(urls), tuple(revisions), tuple(refs), tuple(templates), tuple(sorted(contamination)))
