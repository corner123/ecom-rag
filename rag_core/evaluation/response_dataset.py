"""Strict response-evaluation dataset contract and leakage guards.

The existing ``engineering-eval/v2`` datasets are retrieval ground truth.  In
particular, their ``expected_answer_points`` are useful drafting material but
are not automatically trustworthy response references.  This module defines
the separate ``response-eval/v3`` contract used by LLM/RAGAS evaluation and
keeps migration deliberately review-gated.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


LABEL_VERSION = "response-eval/v3"
DATASET_ROLES = frozenset({"development_response", "private_holdout"})
DIFFICULTIES = frozenset({"easy", "medium", "hard"})
QUERY_VARIANTS = frozenset(
    {"canonical", "paraphrase", "typo", "multi_hop", "boundary"}
)
EXPECTED_ROUTES = frozenset(
    {"design", "implementation", "official", "comparison", "out_of_scope"}
)
REFERENCE_REVIEW_STATUSES = frozenset(
    {"unreviewed", "independent_agent_reviewed", "human_reviewed"}
)

_CLAIM_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_V2_DRAFT_LOCATOR_PREFIX = "v2-label:"


class DatasetLeakageError(ValueError):
    """Raised when evaluation labels could leak across a split or into an index."""


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_exact_fields(
    value: Mapping[str, Any], *, required: set[str], object_name: str
) -> None:
    missing = sorted(required.difference(value))
    extra = sorted(set(value).difference(required))
    if missing:
        raise ValueError(f"{object_name} is missing fields: {missing}")
    if extra:
        raise ValueError(f"{object_name} has unsupported fields: {extra}")


def _string_list(value: Any, field_name: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array")
    items = tuple(_require_non_empty_string(item, field_name) for item in value)
    if not allow_empty and not items:
        raise ValueError(f"{field_name} must contain at least one item")
    return items


def _normalize_source(value: str) -> str:
    return value.replace("\\", "/").strip().casefold().rstrip("/")


@dataclass(frozen=True, slots=True)
class ReferenceClaim:
    """One independently judgeable statement in a reference answer."""

    claim_id: str
    text: str
    required: bool
    evidence_sources: tuple[str, ...]
    locator: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReferenceClaim":
        fields = {"claim_id", "text", "required", "evidence_sources", "locator"}
        _require_exact_fields(value, required=fields, object_name="reference claim")
        claim_id = _require_non_empty_string(value["claim_id"], "claim_id")
        if not _CLAIM_ID.fullmatch(claim_id):
            raise ValueError("claim_id must contain only letters, digits, '.', '_' or '-'")
        if type(value["required"]) is not bool:
            raise ValueError("required must be a JSON boolean")
        sources = _string_list(
            value["evidence_sources"], "evidence_sources", allow_empty=True
        )
        normalized_sources = tuple(_normalize_source(source) for source in sources)
        if len(set(normalized_sources)) != len(normalized_sources):
            raise ValueError("evidence_sources contains duplicates")
        return cls(
            claim_id=claim_id,
            text=_require_non_empty_string(value["text"], "claim text"),
            required=value["required"],
            evidence_sources=sources,
            locator=_require_non_empty_string(value["locator"], "claim locator"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "required": self.required,
            "evidence_sources": list(self.evidence_sources),
            "locator": self.locator,
        }


@dataclass(frozen=True, slots=True)
class ResponseEvaluationSample:
    """One reviewed response-evaluation question and its reference contract."""

    id: str
    question: str
    category: str
    answerable: bool
    expected_route: str
    reference_answer: str
    reference_claims: tuple[ReferenceClaim, ...]
    difficulty: str
    query_variant: str
    dataset_role: str
    source_revision: str
    reference_review_status: str
    label_version: str = LABEL_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResponseEvaluationSample":
        fields = {
            "id",
            "question",
            "category",
            "answerable",
            "expected_route",
            "reference_answer",
            "reference_claims",
            "difficulty",
            "query_variant",
            "dataset_role",
            "source_revision",
            "reference_review_status",
            "label_version",
        }
        _require_exact_fields(value, required=fields, object_name="response sample")
        if value["label_version"] != LABEL_VERSION:
            raise ValueError(f"response samples must use {LABEL_VERSION}")
        if type(value["answerable"]) is not bool:
            raise ValueError("answerable must be a JSON boolean")
        if value["reference_review_status"] not in REFERENCE_REVIEW_STATUSES:
            raise ValueError(
                "unsupported reference_review_status: "
                f"{value['reference_review_status']}"
            )
        if value["expected_route"] not in EXPECTED_ROUTES:
            raise ValueError(f"unsupported expected_route: {value['expected_route']}")
        if value["difficulty"] not in DIFFICULTIES:
            raise ValueError(f"unsupported difficulty: {value['difficulty']}")
        if value["query_variant"] not in QUERY_VARIANTS:
            raise ValueError(f"unsupported query_variant: {value['query_variant']}")
        if value["dataset_role"] not in DATASET_ROLES:
            raise ValueError(f"unsupported dataset_role: {value['dataset_role']}")
        raw_claims = value["reference_claims"]
        if not isinstance(raw_claims, list) or not raw_claims:
            raise ValueError("reference_claims must be a non-empty JSON array")
        if not all(isinstance(item, Mapping) for item in raw_claims):
            raise ValueError("each reference_claim must be a JSON object")
        claims = tuple(ReferenceClaim.from_mapping(item) for item in raw_claims)
        claim_ids = [claim.claim_id for claim in claims]
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("reference_claims contains duplicate claim_id values")
        answerable = value["answerable"]
        if answerable and any(not claim.evidence_sources for claim in claims):
            raise ValueError("answerable claims require at least one evidence source")
        if not answerable and any(claim.evidence_sources for claim in claims):
            raise ValueError("unanswerable claims cannot declare evidence sources")
        if answerable and value["expected_route"] == "out_of_scope":
            raise ValueError("answerable response samples cannot route to out_of_scope")
        if value["query_variant"] == "boundary" and answerable:
            raise ValueError("boundary query variants must be unanswerable")

        return cls(
            id=_require_non_empty_string(value["id"], "sample id"),
            question=_require_non_empty_string(value["question"], "question"),
            category=_require_non_empty_string(value["category"], "category"),
            answerable=answerable,
            expected_route=str(value["expected_route"]),
            reference_answer=_require_non_empty_string(
                value["reference_answer"], "reference_answer"
            ),
            reference_claims=claims,
            difficulty=str(value["difficulty"]),
            query_variant=str(value["query_variant"]),
            dataset_role=str(value["dataset_role"]),
            source_revision=_require_non_empty_string(
                value["source_revision"], "source_revision"
            ),
            reference_review_status=str(value["reference_review_status"]),
            label_version=LABEL_VERSION,
        )

    @property
    def evidence_sources(self) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []
        for claim in self.reference_claims:
            for source in claim.evidence_sources:
                normalized = _normalize_source(source)
                if normalized not in seen:
                    seen.add(normalized)
                    ordered.append(source)
        return tuple(ordered)

    @property
    def is_v2_migration_draft(self) -> bool:
        return any(
            claim.locator.startswith(_V2_DRAFT_LOCATOR_PREFIX)
            for claim in self.reference_claims
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "category": self.category,
            "answerable": self.answerable,
            "expected_route": self.expected_route,
            "reference_answer": self.reference_answer,
            "reference_claims": [claim.to_mapping() for claim in self.reference_claims],
            "difficulty": self.difficulty,
            "query_variant": self.query_variant,
            "dataset_role": self.dataset_role,
            "source_revision": self.source_revision,
            "reference_review_status": self.reference_review_status,
            "label_version": self.label_version,
        }


def normalize_question(question: str) -> str:
    """Normalize only exact textual variants for deterministic leakage checks.

    NFKC joins compatibility forms, ``casefold`` handles case, and Unicode
    whitespace/punctuation are ignored.  This intentionally does not attempt
    semantic or typo similarity; those require a separately reviewed audit.
    """

    text = unicodedata.normalize("NFKC", question).casefold()
    normalized = "".join(
        char
        for char in text
        if not unicodedata.category(char).startswith(("P", "Z"))
    )
    if not normalized:
        raise ValueError("question becomes empty after normalization")
    return normalized


def assert_no_normalized_duplicates(
    *datasets: Sequence[ResponseEvaluationSample],
) -> None:
    """Reject duplicate IDs or normalized questions within/across datasets."""

    seen_ids: dict[str, ResponseEvaluationSample] = {}
    seen_questions: dict[str, ResponseEvaluationSample] = {}
    for dataset in datasets:
        for sample in dataset:
            previous_id = seen_ids.get(sample.id)
            if previous_id is not None:
                raise DatasetLeakageError(
                    f"duplicate response sample id: {sample.id} "
                    f"({previous_id.dataset_role} vs {sample.dataset_role})"
                )
            seen_ids[sample.id] = sample
            normalized = normalize_question(sample.question)
            previous_question = seen_questions.get(normalized)
            if previous_question is not None:
                raise DatasetLeakageError(
                    "normalized duplicate question: "
                    f"{previous_question.id} ({previous_question.dataset_role}) vs "
                    f"{sample.id} ({sample.dataset_role})"
                )
            seen_questions[normalized] = sample


def validate_dataset_separation(
    development: Sequence[ResponseEvaluationSample],
    private_holdout: Sequence[ResponseEvaluationSample],
) -> None:
    """Validate role purity and exact normalized split separation."""

    if any(sample.dataset_role != "development_response" for sample in development):
        raise DatasetLeakageError("development split contains a non-development role")
    if any(sample.dataset_role != "private_holdout" for sample in private_holdout):
        raise DatasetLeakageError("holdout split contains a non-private_holdout role")
    assert_no_normalized_duplicates(development, private_holdout)


def validate_evaluation_paths_outside_corpus(
    dataset_paths: Iterable[str | Path],
    indexable_corpus_roots: Iterable[str | Path],
) -> None:
    """Ensure question/reference files themselves cannot be indexed as corpus.

    Evidence source documents are expected to be in the corpus.  The leakage
    hazard is the JSONL file that contains questions and reference answers.
    Callers therefore pass the actual indexable roots, not the repository root
    unless the entire repository is truly indexed.
    """

    roots = tuple(Path(root).expanduser().resolve() for root in indexable_corpus_roots)
    for dataset_path in dataset_paths:
        candidate = Path(dataset_path).expanduser().resolve()
        for root in roots:
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            raise DatasetLeakageError(
                f"evaluation dataset is inside an indexable corpus root: {candidate}"
            )


def _validate_formal_sample(sample: ResponseEvaluationSample) -> None:
    if sample.reference_review_status == "unreviewed":
        raise ValueError(
            f"formal response evaluation rejects unreviewed sample: {sample.id}"
        )
    if sample.is_v2_migration_draft:
        raise ValueError(
            f"formal response evaluation rejects v2 draft locators: {sample.id}"
        )


def load_response_jsonl(
    path: str | Path,
    *,
    formal: bool = True,
    indexable_corpus_roots: Iterable[str | Path] = (),
) -> list[ResponseEvaluationSample]:
    """Load a UTF-8 v3 JSONL file and enforce review/leakage constraints."""

    dataset_path = Path(path)
    validate_evaluation_paths_outside_corpus(
        [dataset_path], indexable_corpus_roots
    )
    samples: list[ResponseEvaluationSample] = []
    for line_number, line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON at {dataset_path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"response sample must be a JSON object at {dataset_path}:{line_number}"
            )
        try:
            sample = ResponseEvaluationSample.from_mapping(payload)
            if formal:
                _validate_formal_sample(sample)
        except ValueError as exc:
            raise ValueError(f"{dataset_path}:{line_number}: {exc}") from exc
        samples.append(sample)
    if not samples:
        raise ValueError(f"response evaluation dataset is empty: {dataset_path}")
    assert_no_normalized_duplicates(samples)
    return samples


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of the exact on-disk dataset bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_sha256(samples: Sequence[ResponseEvaluationSample]) -> str:
    """Return an order-independent SHA-256 of canonical validated samples."""

    canonical = "\n".join(
        json.dumps(
            sample.to_mapping(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for sample in sorted(samples, key=lambda item: item.id)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _v2_get(value: Mapping[str, Any] | Any, field_name: str) -> Any:
    if isinstance(value, Mapping):
        if field_name not in value:
            raise ValueError(f"v2 sample is missing field: {field_name}")
        return value[field_name]
    try:
        return getattr(value, field_name)
    except AttributeError as exc:
        raise ValueError(f"v2 sample is missing field: {field_name}") from exc


def _v2_route(value: Mapping[str, Any] | Any) -> str:
    if isinstance(value, Mapping):
        route = value.get("expected_route")
    else:
        route = getattr(value, "expected_route_label", None)
    if not isinstance(route, str) or route not in EXPECTED_ROUTES:
        raise ValueError("v2 sample requires a supported expected_route")
    return route


def _derive_difficulty(
    *, route: str, point_count: int, source_count: int
) -> str:
    if route == "comparison" or point_count >= 4 or source_count >= 3:
        return "hard"
    if point_count >= 2 or source_count >= 2:
        return "medium"
    return "easy"


def _derive_query_variant(*, answerable: bool, route: str) -> str:
    if not answerable:
        return "boundary"
    if route == "comparison":
        return "multi_hop"
    return "canonical"


def migrate_v2_sample_to_draft(
    value: Mapping[str, Any] | Any,
    *,
    dataset_role: str,
) -> ResponseEvaluationSample:
    """Convert one v2 sample into a review-gated v3 drafting record.

    The helper never claims that joined v2 answer points are a reviewed
    reference.  It always sets ``reference_review_status='unreviewed'`` and
    emits sentinel locators that the formal loader refuses even if the status
    is later changed without reviewing the evidence.
    """

    if dataset_role not in DATASET_ROLES:
        raise ValueError(f"unsupported dataset_role: {dataset_role}")
    label_version = _v2_get(value, "label_version")
    if label_version != "engineering-eval/v2":
        raise ValueError("migration input must use engineering-eval/v2")
    sample_id = _require_non_empty_string(_v2_get(value, "id"), "v2 id")
    question = _require_non_empty_string(_v2_get(value, "question"), "v2 question")
    category = _require_non_empty_string(_v2_get(value, "category"), "v2 category")
    answerable = _v2_get(value, "answerable")
    if type(answerable) is not bool:
        raise ValueError("v2 answerable must be a boolean")
    raw_points = _v2_get(value, "expected_answer_points")
    if not isinstance(raw_points, (list, tuple)):
        raise ValueError("v2 expected_answer_points must be a sequence")
    points = tuple(
        _require_non_empty_string(point, "v2 expected_answer_point")
        for point in raw_points
    )
    if not points:
        raise ValueError("v2 expected_answer_points cannot be empty")
    raw_sources = _v2_get(value, "primary_sources")
    if not isinstance(raw_sources, (list, tuple)):
        raise ValueError("v2 primary_sources must be a sequence")
    sources = tuple(
        _require_non_empty_string(source, "v2 primary source")
        for source in raw_sources
    )
    if answerable and not sources:
        raw_relevant = _v2_get(value, "relevant_sources")
        if not isinstance(raw_relevant, (list, tuple)):
            raise ValueError("v2 relevant_sources must be a sequence")
        sources = tuple(
            _require_non_empty_string(source, "v2 relevant source")
            for source in raw_relevant
        )
    if answerable and not sources:
        raise ValueError("answerable v2 samples require evidence sources")
    if not answerable:
        sources = ()
    route = _v2_route(value)
    source_revision = _require_non_empty_string(
        _v2_get(value, "source_revision"), "v2 source_revision"
    )
    claims = tuple(
        ReferenceClaim(
            claim_id=f"C{index}",
            text=point,
            required=True,
            evidence_sources=sources,
            locator=f"{_V2_DRAFT_LOCATOR_PREFIX}{sample_id}:expected_answer_points[{index - 1}]",
        )
        for index, point in enumerate(points, start=1)
    )
    draft = ResponseEvaluationSample(
        id=sample_id,
        question=question,
        category=category,
        answerable=answerable,
        expected_route=route,
        reference_answer="\n".join(f"- {point}" for point in points),
        reference_claims=claims,
        difficulty=_derive_difficulty(
            route=route, point_count=len(points), source_count=len(sources)
        ),
        query_variant=_derive_query_variant(answerable=answerable, route=route),
        dataset_role=dataset_role,
        source_revision=source_revision,
        reference_review_status="unreviewed",
    )
    # Exercise the same strict invariants as JSON loading.
    return ResponseEvaluationSample.from_mapping(draft.to_mapping())


def _stable_rank(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).hexdigest()


def stratified_select(
    samples: Sequence[ResponseEvaluationSample],
    sample_size: int,
    *,
    seed: int = 0,
) -> list[ResponseEvaluationSample]:
    """Select deterministic round-robin strata without input-order dependence."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(samples):
        raise ValueError("sample_size cannot exceed the candidate count")
    assert_no_normalized_duplicates(samples)
    grouped: dict[tuple[bool, str, str, str], deque[ResponseEvaluationSample]] = {}
    pending: dict[tuple[bool, str, str, str], list[ResponseEvaluationSample]] = (
        defaultdict(list)
    )
    for sample in samples:
        stratum = (
            sample.answerable,
            sample.expected_route,
            sample.difficulty,
            sample.query_variant,
        )
        pending[stratum].append(sample)
    for stratum, members in pending.items():
        grouped[stratum] = deque(
            sorted(members, key=lambda item: (_stable_rank(item.id, seed), item.id))
        )

    selected: list[ResponseEvaluationSample] = []
    strata = sorted(grouped, key=lambda item: repr(item))
    while len(selected) < sample_size:
        made_progress = False
        for stratum in strata:
            members = grouped[stratum]
            if members:
                selected.append(members.popleft())
                made_progress = True
                if len(selected) == sample_size:
                    break
        if not made_progress:  # Defensive; size validation makes this unreachable.
            break
    return selected


def migrate_v2_samples_stratified(
    values: Sequence[Mapping[str, Any] | Any],
    *,
    sample_size: int,
    dataset_role: str,
    seed: int = 0,
) -> list[ResponseEvaluationSample]:
    """Migrate v2 candidates to drafts, then take a deterministic stratified subset."""

    drafts = [
        migrate_v2_sample_to_draft(value, dataset_role=dataset_role)
        for value in values
    ]
    return stratified_select(drafts, sample_size, seed=seed)
