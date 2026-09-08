"""Auditable development/holdout leakage gates for trade evaluation data."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit

from trade_agent.evaluation.generator import EvaluationBundle, _facts


_WORD = re.compile(r"[a-z0-9]+")
_ENTITY = re.compile(r"\b[A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*){1,3} \d{2}\b", re.IGNORECASE)
_EVENT = re.compile(r"\b(?:\d{6}|\d{4}-\d{2}|\d{4})\b")


@dataclass(frozen=True)
class LeakageReport:
    normalized_questions: tuple[str, ...] = ()
    exact_chunk_hashes: tuple[str, ...] = ()
    near_chunk_hashes: tuple[str, ...] = ()
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


def _provenance_from_corpus(bundle: EvaluationBundle, corpus: Any) -> dict[str, set[str]]:
    """Recover source identity when bundles have been read from JSONL files."""
    if not isinstance(corpus, Mapping) or "records" not in corpus:
        return {}
    try:
        by_evidence = {fact.evidence_id: fact for fact in _facts(corpus)}
    except (KeyError, TypeError, ValueError):
        return {}
    selected = [by_evidence[item.evidence_id] for item in bundle.evidence if item.evidence_id in by_evidence]
    return {
        "chunk_hashes": {fact.chunk_hash for fact in selected},
        "near_chunk_hashes": {fact.chunk_hash for fact in selected},
        "canonical_urls": {fact.canonical_url for fact in selected},
        "source_revisions": {fact.source_revision for fact in selected},
        "entity_event_template": {str((fact.entity, fact.event)) for fact in selected},
    }


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
        dev_recovered, holdout_recovered = _provenance_from_corpus(dev, corpus), _provenance_from_corpus(holdout, corpus)
        def values(bundle: EvaluationBundle, recovered: Mapping[str, set[str]], key: str) -> set[str]:
            return _bundle_values(bundle, key) or set(recovered.get(key, set()))
        exact_hashes = sorted(values(dev, dev_recovered, "chunk_hashes") & values(holdout, holdout_recovered, "chunk_hashes"))
        near_hashes = sorted(values(dev, dev_recovered, "near_chunk_hashes") &
                             values(holdout, holdout_recovered, "near_chunk_hashes"))
        urls = sorted({_url(value) for value in values(dev, dev_recovered, "canonical_urls")} &
                      {_url(value) for value in values(holdout, holdout_recovered, "canonical_urls")})
        revisions = sorted(values(dev, dev_recovered, "source_revisions") & values(holdout, holdout_recovered, "source_revisions"))
        dev_refs = {item.reference_evidence_set_id for item in dev.evidence} | {item.claim_id for item in dev.claims}
        holdout_refs = {item.reference_evidence_set_id for item in holdout.evidence} | {item.claim_id for item in holdout.claims}
        refs = sorted(dev_refs & holdout_refs)
        templates = sorted(values(dev, dev_recovered, "entity_event_template") &
                           values(holdout, holdout_recovered, "entity_event_template"))
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
        return LeakageReport(tuple(question_hits), tuple(exact_hashes), tuple(near_hashes), tuple(urls),
                             tuple(revisions), tuple(refs), tuple(templates), tuple(sorted(contamination)))
