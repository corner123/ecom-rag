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
    container_tags = {
        "html", "body", "main", "section", "article", "div", "header", "footer",
        "nav", "ul", "ol", "dl", "table", "thead", "tbody", "tfoot", "tr",
    }
    heading_tags = {f"h{level}" for level in range(1, 7)}

    def emit(value: str) -> None:
        value = _plain_text(value)
        if not value:
            return
        if stack:
            _append_group(groups, " > ".join(stack), value, limit)
        else:
            preamble.append(value)

    def visit(parent: Tag | BeautifulSoup) -> None:
        """Walk each block once and coalesce adjacent text/inline wrappers in DOM order."""
        inline_run: list[str] = []

        def flush_inline() -> None:
            if inline_run:
                emit("".join(inline_run))
                inline_run.clear()

        for child in parent.children:
            if type(child) is NavigableString:
                inline_run.append(str(child))
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name.lower() if child.name else ""
            if name in {"script", "style", "noscript", "template", "head"}:
                continue
            if name in heading_tags:
                flush_inline()
                heading = _plain_text("".join(child.strings))
                if heading:
                    level = int(name[1])
                    stack[:] = stack[: level - 1] + [heading]
                continue
            if name in evidence_tags:
                flush_inline()
                emit("".join(child.strings))
                continue
            if name in container_tags or child.find(heading_tags | evidence_tags | container_tags):
                flush_inline()
                visit(child)
                continue
            # Inline wrappers belong to the surrounding flow. Consuming all of
            # their leaf strings here prevents both loss and descendant replay.
            inline_run.append("".join(child.strings))
        flush_inline()

    visit(soup)

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

    fence: tuple[str, int] | None = None
    for line in text.splitlines():
        fence_match = re.match(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$", line)
        if fence is not None:
            buffer.append(line)
            marker, minimum = fence
            if re.fullmatch(rf"[ \t]{{0,3}}{re.escape(marker)}{{{minimum},}}[ \t]*", line):
                fence = None
            continue
        if fence_match:
            marker = fence_match.group(1)
            fence = (marker[0], len(marker))
            buffer.append(line)
            continue
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
