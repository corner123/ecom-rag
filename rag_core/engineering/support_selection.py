"""Versioned policies for the single cross-role supporting-evidence slot.

The query-aware policy deliberately uses only request-time information.  It
must never depend on evaluation labels, reference answers or expected sources.
"""

from __future__ import annotations

from enum import Enum
import re
from typing import Iterable, Sequence

from rag_core.retrieval.engineering import EngineeringSearchResult


class SupportSelectionProfile(str, Enum):
    """Stable names for supporting-evidence selection experiments."""

    LEGACY_FIRST = "legacy_first"
    QUERY_AWARE_DIVERSE = "query_aware_diverse"


_LATIN_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
_IDENTIFIER_SPLIT_RE = re.compile(r"[._-]+")
_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def extract_latin_identifier_tokens(text: object) -> frozenset[str]:
    """Return case-folded Latin/identifier tokens and their atomic parts.

    ``QueryEngine.submit_message`` therefore yields the compound identifier as
    well as ``queryengine`` and ``submit``/``message``.  Single-character and
    grammatical English tokens are discarded.  No language model, corpus
    label or evaluation annotation participates in extraction.
    """

    tokens: set[str] = set()
    for match in _LATIN_IDENTIFIER_RE.finditer(str(text or "")):
        raw = match.group(0).strip("._-").casefold()
        if not raw:
            continue
        pieces = [raw, *_IDENTIFIER_SPLIT_RE.split(raw)]
        for piece in pieces:
            if len(piece) < 2 or piece in _QUERY_STOPWORDS:
                continue
            tokens.add(piece)
    return frozenset(tokens)


def logical_file_key(result: EngineeringSearchResult) -> str:
    """Normalize one result to a file-level key for diversity de-duplication.

    A repository-relative metadata path is authoritative when present;
    otherwise the public source is used.  Symbol/section fragments and numeric
    line suffixes do not create a new logical file.
    """

    metadata = result.metadata
    value = str(
        metadata.get("relative_path")
        or metadata.get("document_path")
        or metadata.get("file_path")
        or result.source
    ).strip()
    value = value.split("#", 1)[0].replace("\\", "/")
    value = re.sub(r":\d+(?:-\d+)?$", "", value)
    value = re.sub(r"/+", "/", value)
    return value.casefold()


def _symbol_text(result: EngineeringSearchResult) -> str:
    metadata = result.metadata
    values = (
        result.symbol,
        metadata.get("symbol"),
        metadata.get("symbol_name"),
    )
    return " ".join(str(value) for value in values if value)


def _path_text(result: EngineeringSearchResult) -> str:
    metadata = result.metadata
    values = (
        result.source,
        metadata.get("relative_path"),
        metadata.get("document_path"),
        metadata.get("file_path"),
    )
    return " ".join(str(value) for value in values if value)


def _eligible_candidates(
    selected: Sequence[EngineeringSearchResult],
    supporting: Iterable[EngineeringSearchResult],
) -> list[tuple[int, EngineeringSearchResult]]:
    selected_ids = {item.result_id for item in selected}
    selected_files = {logical_file_key(item) for item in selected}
    selected_roles = {str(item.metadata.get("evidence_role")) for item in selected}
    eligible: list[tuple[int, EngineeringSearchResult]] = []
    seen_files = set(selected_files)
    for rank, item in enumerate(supporting, start=1):
        file_key = logical_file_key(item)
        role = str(item.metadata.get("evidence_role"))
        if (
            item.result_id in selected_ids
            or file_key in seen_files
            or role in selected_roles
        ):
            continue
        eligible.append((rank, item))
        seen_files.add(file_key)
    return eligible


def _legacy_candidate(
    selected: Sequence[EngineeringSearchResult],
    supporting: Sequence[EngineeringSearchResult],
) -> EngineeringSearchResult | None:
    """Reproduce the historical first-result policy byte-for-byte in intent."""

    candidate = next(
        (
            item
            for item in supporting
            if all(item.result_id != existing.result_id for existing in selected)
        ),
        None,
    )
    if candidate is None:
        return None
    roles = {str(item.metadata.get("evidence_role")) for item in selected}
    if str(candidate.metadata.get("evidence_role")) in roles:
        return None
    return candidate


def select_supporting_candidate(
    *,
    query: str,
    primary_results: Sequence[EngineeringSearchResult],
    supporting_candidates: Sequence[EngineeringSearchResult],
    profile: SupportSelectionProfile | str = SupportSelectionProfile.LEGACY_FIRST,
) -> EngineeringSearchResult | None:
    """Choose one supporting result without consuming evaluation supervision.

    ``query_aware_diverse`` scores each eligible original-rank candidate as::

       10 * symbol_matches
      + 2 * path_matches
      + 0.5 * content_matches
      + 3 * novel_identifier_matches
      + 1 / original_rank

    Novel identifiers are query tokens present anywhere in the candidate but
    absent from every primary result's path/symbol.  Primary content is not
    used for novelty because the selector is explicitly choosing a second
    evidence role whose job is to add implementation/design identifiers. Ties
    retain the original retrieval rank. A query with no usable
    Latin/identifier tokens safely falls back to the historical selector.
    """

    selected_profile = SupportSelectionProfile(profile)
    if selected_profile is SupportSelectionProfile.LEGACY_FIRST:
        return _legacy_candidate(primary_results, supporting_candidates)

    query_tokens = extract_latin_identifier_tokens(query)
    if not query_tokens:
        return _legacy_candidate(primary_results, supporting_candidates)

    eligible = _eligible_candidates(primary_results, supporting_candidates)
    if not eligible:
        return None

    primary_identifier_coverage: set[str] = set()
    for item in primary_results:
        primary_identifier_coverage.update(
            query_tokens
            & (
                extract_latin_identifier_tokens(_symbol_text(item))
                | extract_latin_identifier_tokens(_path_text(item))
            )
        )

    scored: list[tuple[float, int, EngineeringSearchResult]] = []
    for rank, item in eligible:
        symbol_matches = query_tokens & extract_latin_identifier_tokens(
            _symbol_text(item)
        )
        path_matches = query_tokens & extract_latin_identifier_tokens(
            _path_text(item)
        )
        content_matches = query_tokens & extract_latin_identifier_tokens(item.content)
        novel_matches = (
            symbol_matches | path_matches | content_matches
        ) - primary_identifier_coverage
        score = (
            10.0 * len(symbol_matches)
            + 2.0 * len(path_matches)
            + 0.5 * len(content_matches)
            + 3.0 * len(novel_matches)
            + 1.0 / rank
        )
        scored.append((score, rank, item))

    # ``rank`` is explicit even though 1/rank normally makes scores distinct;
    # this documents and guarantees stable original-order tie breaking.
    return max(scored, key=lambda row: (row[0], -row[1]))[2]


def ensure_internal_support(
    *,
    query: str,
    results: Sequence[EngineeringSearchResult],
    supporting_candidates: Sequence[EngineeringSearchResult],
    top_k: int,
    profile: SupportSelectionProfile | str = SupportSelectionProfile.LEGACY_FIRST,
) -> list[EngineeringSearchResult]:
    """Preserve the fixed one-slot evidence contract and select its occupant."""

    selected = list(results[:top_k])
    if top_k < 2 or not supporting_candidates:
        return selected
    candidate = select_supporting_candidate(
        query=query,
        primary_results=selected,
        supporting_candidates=supporting_candidates,
        profile=profile,
    )
    if candidate is None:
        return selected
    return [*selected[: top_k - 1], candidate]
