"""Deterministic evidence-scope checks layered above nearest-neighbour search."""

from __future__ import annotations

import re
from typing import Iterable

from rag_core.retrieval.engineering import EngineeringSearchResult, SourceIntent


_ANCHOR_RE = re.compile(
    r"`([^`\r\n]{1,96})`|"
    r"\b(?:[A-Za-z][A-Za-z0-9_-]*[._/][A-Za-z0-9_./-]+|"
    r"[A-Z][A-Za-z0-9]*(?:[A-Z][A-Za-z0-9]*)+|[A-Z]{2,}|"
    r"Redis|Kafka|PostgreSQL|Kubernetes|OAuth|CQRS)\b"
)
_TOPIC_ANCHOR_RE = re.compile(
    r"(?:什么是|介绍(?:一下)?|解释(?:一下)?|如何(?:使用|配置)|怎么(?:使用|配置)|"
    r"what\s+is|define|explain|how\s+to\s+(?:use|configure))"
    r"\s*[`\"'“”]?\s*([A-Za-z][A-Za-z0-9_.\-/]{2,})",
    re.IGNORECASE,
)
_TOPIC_SUFFIX_ANCHOR_RE = re.compile(
    r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9_.\-/]{2,})(?![A-Za-z0-9_])"
    r"\s*[`\"'“”]?\s*(?:是什么|是什[么麼]|怎么(?:使用|配置)|如何(?:使用|配置)|"
    r"what\s+is|definition)",
    re.IGNORECASE,
)
_LATIN_WORD_RE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_-]{2,}(?![A-Za-z0-9_])"
)
_GENERIC_ANCHORS = {
    "api", "cli", "rag", "agent", "mini-nanobot", "python", "json", "http",
    "url", "cpu", "mcp", "langgraph", "docker", "queryengine", "agentstate",
    "what", "is", "are", "how", "why", "does", "do", "use", "uses", "using",
    "define", "explain", "tell", "about", "the", "and", "for", "with", "from",
    "project", "system", "framework", "library", "tool", "official", "docs",
    "documentation", "current", "implementation", "design", "architecture",
}
_MISSING_EMPIRICAL_CLAIM = re.compile(
    r"(?:(?:成功率|准确率|Recall@\d+|吞吐量|失败恢复时间|月均故障率|"
    r"P(?:50|90|95|99)|峰值内存|并发)"
    r".{0,30}(?:多少|是多少|分别|精确)|"
    r"(?:是否|能否).{0,30}(?:独立.*审计|不存在.*逃逸)|"
    r"(?:真实模型|大型仓库).{0,40}(?:成功率|吞吐量|恢复时间))",
    re.IGNORECASE,
)
_MISSING_INDEPENDENT_VERIFICATION = re.compile(
    r"(?:独立.*审计|不存在.*逃逸|SOC\s*2|认证|审计报告)",
    re.IGNORECASE,
)


class EvidenceSufficiencyGuard:
    """Reject out-of-catalog topics and ungrounded hard identifiers.

    RRF scores are rank-fusion values and dense search always has a nearest
    neighbour, so neither is a safe answerability threshold. This guard uses
    the explicitly curated source scope plus lexical hard-anchor presence.
    """

    def check(
        self,
        query: str,
        intent: SourceIntent,
        results: Iterable[EngineeringSearchResult],
    ) -> tuple[bool, list[str], str | None]:
        evidence = list(results)
        warnings: list[str] = []

        if _MISSING_INDEPENDENT_VERIFICATION.search(query):
            return (
                False,
                ["the curated snapshot contains no independent verification"],
                "missing_independent_verification",
            )

        if _MISSING_EMPIRICAL_CLAIM.search(query):
            return (
                False,
                ["the curated snapshot contains no required empirical evidence"],
                "missing_empirical_evidence",
            )

        if intent in {SourceIntent.OFFICIAL, SourceIntent.COMPARISON}:
            official = [
                item
                for item in evidence
                if item.metadata.get("evidence_role") == "external_normative"
            ]
            scope_ok, reason = self._official_scope(
                query,
                official,
                check_anchors=intent is SourceIntent.OFFICIAL,
            )
            if not scope_ok:
                warnings.append(reason)
                code = (
                    "topic_not_indexed"
                    if "subtopic" in reason
                    else "source_not_catalogued"
                )
                return False, warnings, code

        if intent is SourceIntent.DESIGN:
            design = [
                item
                for item in evidence
                if item.metadata.get("evidence_role")
                in {"internal_design", "internal_history"}
            ]
            missing = _missing_hard_anchors(query, design)
            if missing:
                warnings.append(
                    "internal design evidence does not contain required anchors: "
                    + ", ".join(missing)
                )
                return False, warnings, "unsupported_anchor"

        if intent is SourceIntent.IMPLEMENTATION:
            live = [
                item
                for item in evidence
                if item.metadata.get("evidence_role") == "current_implementation"
            ]
            missing = _missing_hard_anchors(query, live)
            if missing:
                warnings.append(
                    "live implementation evidence does not contain required anchors: "
                    + ", ".join(missing)
                )
                return False, warnings, "unsupported_anchor"

        if intent is SourceIntent.COMPARISON:
            live = [
                item
                for item in evidence
                if item.metadata.get("evidence_role") == "current_implementation"
            ]
            anchors = _hard_anchors(query)
            if not anchors or all(
                anchor.casefold()
                not in "\n".join(
                    f"{item.source}\n{item.symbol or ''}\n{item.content}"
                    for item in live
                ).casefold()
                for anchor in anchors
            ):
                return (
                    False,
                    ["comparison has no query anchor verified in current source"],
                    "missing_current_implementation_evidence",
                )

        return True, warnings, None

    def _official_scope(
        self,
        query: str,
        results: list[EngineeringSearchResult],
        *,
        check_anchors: bool,
    ) -> tuple[bool, str]:
        query_folded = query.casefold()
        source_matched = False
        topic_matched = False
        for result in results:
            aliases = _string_list(result.metadata.get("scope_aliases"))
            topics = _string_list(result.metadata.get("covered_topics"))
            if not aliases or not any(alias.casefold() in query_folded for alias in aliases):
                continue
            source_matched = True
            if any(topic.casefold() in query_folded for topic in topics):
                topic_matched = True
        if not source_matched:
            return False, "official source is outside the curated catalog scope"
        if not topic_matched:
            return False, "requested official subtopic is not present in the curated snapshot"
        missing = _missing_hard_anchors(query, results) if check_anchors else []
        if missing:
            return (
                False,
                "official evidence does not contain required anchors: "
                + ", ".join(missing),
            )
        return True, ""


def _missing_hard_anchors(
    query: str, results: Iterable[EngineeringSearchResult]
) -> list[str]:
    anchors = _hard_anchors(query)
    if not anchors:
        return []
    haystack = "\n".join(
        f"{item.source}\n{item.symbol or ''}\n{item.content}" for item in results
    ).casefold()
    return [anchor for anchor in anchors if anchor.casefold() not in haystack]


def _hard_anchors(query: str) -> list[str]:
    anchors: list[str] = []
    topic_matches = (
        *_TOPIC_ANCHOR_RE.finditer(query),
        *_TOPIC_SUFFIX_ANCHOR_RE.finditer(query),
    )
    for match in topic_matches:
        value = match.group(1).strip()
        if value.casefold() not in _GENERIC_ANCHORS:
            anchors.append(value)
    for match in _ANCHOR_RE.finditer(query):
        value = (match.group(1) or match.group(0)).strip()
        folded = value.casefold()
        if folded in _GENERIC_ANCHORS or folded in {item.casefold() for item in anchors}:
            continue
        anchors.append(value)
    # Lower-case framework and product names do not match the CamelCase branch
    # above.  When a query contains exactly one distinctive Latin token, treat
    # it as a required topic anchor instead of accepting an arbitrary nearest
    # neighbour from an unrelated partition.
    distinctive = []
    for match in _LATIN_WORD_RE.finditer(query):
        value = match.group(0)
        folded = value.casefold()
        if folded in _GENERIC_ANCHORS or folded in {
            item.casefold() for item in anchors
        }:
            continue
        distinctive.append(value)
    if not anchors and len({item.casefold() for item in distinctive}) == 1:
        anchors.append(distinctive[0])
    return anchors


def _string_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
