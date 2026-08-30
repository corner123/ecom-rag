"""Bounded physical parsers that report stable, typed failures."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag

from .pdf import pymupdf_extract


@dataclass(frozen=True)
class ParseFailure(Exception):
    code: str
    parser: str
    diagnostic: str


def _plain_text(value: object) -> str:
    return " ".join(str(value).split())


def _append_group(groups: OrderedDict[str, list[str]], path: str, value: str, limit: int) -> None:
    if path not in groups and len(groups) >= limit:
        raise ParseFailure("DOCUMENT_COUNT_LIMIT", "section", "document limit reached")
    groups.setdefault(path, []).append(value)


def parse_html_sections(text: str, title: str, limit: int = 10_000) -> list[tuple[str, str]]:
    """Extract evidence blocks once while maintaining the live HTML heading stack."""
    soup = BeautifulSoup(text, "html.parser")
    for bad in soup.find_all(["script", "style", "noscript", "template"]):
        bad.decompose()

    stack: list[str] = []
    groups: OrderedDict[str, list[str]] = OrderedDict()
    preamble: list[str] = []
    evidence_tags = {"p", "li", "td", "th", "aside", "blockquote", "pre"}
    direct_text_parents = {"body", "main", "section", "article", "div", "header", "footer"}

    for node in soup.descendants:
        if isinstance(node, Tag):
            name = node.name.lower() if node.name else ""
            if re.fullmatch(r"h[1-6]", name):
                heading = _plain_text("".join(node.strings))
                if heading:
                    level = int(name[1])
                    stack = stack[: level - 1] + [heading]
                continue
            if name not in evidence_tags or node.find_parent(evidence_tags) is not None:
                continue
            value = _plain_text("".join(node.strings))
            if not value:
                continue
        elif isinstance(node, NavigableString):
            parent = node.parent
            if parent is None or (parent.name or "").lower() not in direct_text_parents:
                continue
            if parent.find_parent(evidence_tags | {f"h{i}" for i in range(1, 7)}) is not None:
                continue
            value = _plain_text(node)
            if not value:
                continue
        else:
            continue

        if stack:
            _append_group(groups, " > ".join(stack), value, limit)
        else:
            preamble.append(value)

    if groups and preamble:
        first = next(iter(groups))
        groups[first] = preamble + groups[first]
    elif preamble:
        _append_group(groups, title, "\n".join(preamble), limit)

    result = [(path, "\n".join(parts)) for path, parts in groups.items() if parts]
    if not result:
        raise ParseFailure("INVALID_DOCUMENT_SHAPE", "html", "no usable evidence text")
    return result


def parse_markdown_sections(text: str, title: str, limit: int = 10_000) -> list[tuple[str, str]]:
    """Group Markdown blocks under a clean h1>h2>h3 hierarchy."""
    stack: list[str] = []
    buffer: list[str] = []
    preamble: list[str] = []
    groups: OrderedDict[str, list[str]] = OrderedDict()

    def flush() -> None:
        nonlocal buffer
        value = "\n".join(buffer).strip()
        buffer = []
        if not value:
            return
        if stack:
            _append_group(groups, " > ".join(stack), value, limit)
        else:
            preamble.append(value)

    for line in text.splitlines():
        match = re.match(r"^(#{1,6})[ \t]+(.+?)\s*#*\s*$", line)
        if match:
            flush()
            level = len(match.group(1))
            heading = match.group(2).strip()
            stack = stack[: level - 1] + [heading]
        else:
            buffer.append(line)
    flush()

    if groups and preamble:
        first = next(iter(groups))
        groups[first] = preamble + groups[first]
    elif preamble:
        _append_group(groups, title, "\n\n".join(preamble), limit)

    result = [(path, "\n\n".join(parts)) for path, parts in groups.items() if parts]
    if not result:
        raise ParseFailure("INVALID_DOCUMENT_SHAPE", "markdown", "no usable evidence text")
    return result


def json_value(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseFailure("MALFORMED_JSON", "json", "invalid JSON syntax") from exc


def jsonl_values(text: str, limit: int) -> list[tuple[int, dict]]:
    values: list[tuple[int, dict]] = []
    for row, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if len(values) >= limit:
            raise ParseFailure("DOCUMENT_COUNT_LIMIT", "jsonl", "document limit reached")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ParseFailure("MALFORMED_JSON", "jsonl", f"invalid JSON at line {row}") from exc
        if not isinstance(value, dict):
            raise ParseFailure("INVALID_DOCUMENT_SHAPE", "jsonl", f"line {row} must be object")
        values.append((row, value))
    return values


__all__ = ["ParseFailure", "json_value", "jsonl_values", "parse_html_sections", "parse_markdown_sections", "pymupdf_extract"]
