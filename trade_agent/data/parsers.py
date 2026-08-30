"""Bounded, typed physical parsers. Router turns ParseFailure into quarantine."""
from __future__ import annotations
import json
from dataclasses import dataclass
from bs4 import BeautifulSoup, NavigableString
from .pdf import pymupdf_extract

@dataclass(frozen=True)
class ParseFailure(Exception):
    code: str
    parser: str
    diagnostic: str

def _hierarchy_from_nodes(nodes, title):
    stack, groups = [title], []
    for node in nodes:
        if not isinstance(node, NavigableString): continue
        if node.parent and node.parent.name in {"script", "style", "noscript"}: continue
        value = " ".join(str(node).split())
        if not value: continue
        parent = node.parent
        heading = parent if parent and parent.name and parent.name.startswith("h") and parent.name[1:].isdigit() else None
        if heading:
            level = int(heading.name[1]); stack = stack[:level]
            stack.append(value); continue
        if parent and parent.name in {"p", "li", "td", "th", "div", "section", "article"}:
            # descendants belong to the nearest evidence container exactly once
            ancestor = parent.parent
            if ancestor and ancestor.name in {"p", "li", "td", "th"}: continue
            groups.append((" > ".join(stack), value))
    return groups

def parse_html_sections(text: str, title: str):
    soup = BeautifulSoup(text, "html.parser")
    result = _hierarchy_from_nodes(soup.descendants, title)
    if not result: raise ParseFailure("INVALID_DOCUMENT_SHAPE", "html", "no usable evidence text")
    return result

def parse_markdown_sections(text: str, title: str):
    stack, buffer, result = [title], [], []
    for line in text.splitlines():
        if line.startswith("#") and line.lstrip("#").strip():
            if "\n".join(buffer).strip(): result.append((" > ".join(stack), "\n".join(buffer).strip()))
            level = len(line) - len(line.lstrip("#")); stack = stack[:level]; stack.append(line[level:].strip()); buffer=[]
        else: buffer.append(line)
    if "\n".join(buffer).strip(): result.append((" > ".join(stack), "\n".join(buffer).strip()))
    if not result: raise ParseFailure("INVALID_DOCUMENT_SHAPE", "markdown", "no usable evidence text")
    return result

def json_value(text: str):
    try: return json.loads(text)
    except json.JSONDecodeError as exc: raise ParseFailure("MALFORMED_JSON", "json", "invalid JSON syntax") from exc

def jsonl_values(text: str, limit: int):
    values=[]
    for row,line in enumerate(text.splitlines(),1):
        if not line.strip(): continue
        if len(values) >= limit: raise ParseFailure("DOCUMENT_COUNT_LIMIT", "jsonl", "document limit reached")
        try: value=json.loads(line)
        except json.JSONDecodeError as exc: raise ParseFailure("MALFORMED_JSON", "jsonl", f"invalid JSON at line {row}") from exc
        if not isinstance(value,dict): raise ParseFailure("INVALID_DOCUMENT_SHAPE", "jsonl", f"line {row} must be object")
        values.append((row,value))
    return values

__all__=["ParseFailure","parse_html_sections","parse_markdown_sections","json_value","jsonl_values","pymupdf_extract"]
