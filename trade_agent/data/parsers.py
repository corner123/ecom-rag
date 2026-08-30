"""Physical parser helpers; router owns fail-closed dispatch and quarantine."""
from bs4 import BeautifulSoup

from .pdf import pymupdf_extract


def parse_html_sections(text: str, title: str) -> list[tuple[str, str]]:
    """One pass through evidence nodes; section text is never re-emitted as children."""
    soup = BeautifulSoup(text, "html.parser")
    containers = soup.find_all("section")
    if containers:
        result = []
        for section in containers:
            heading = section.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])
            body = section.get_text(" ", strip=True)
            if body: result.append((heading.get_text(" ", strip=True) if heading else section.get("id") or title, body))
        return result
    heading, result = title, []
    for node in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        value = node.get_text(" ", strip=True)
        if node.name.startswith("h"): heading = value
        elif value: result.append((heading, value))
    return result


def parse_markdown_sections(text: str, title: str) -> list[tuple[str, str]]:
    path, buffer, result = [title], [], []
    for line in text.splitlines():
        if line.startswith("#"):
            if buffer and "\n".join(buffer).strip(): result.append((" > ".join(path), "\n".join(buffer).strip()))
            level = len(line) - len(line.lstrip("#")); heading = line[level:].strip()
            path = path[:level] + [heading]; buffer = []
        else: buffer.append(line)
    if buffer and "\n".join(buffer).strip(): result.append((" > ".join(path), "\n".join(buffer).strip()))
    return result


__all__ = ["parse_html_sections", "parse_markdown_sections", "pymupdf_extract"]
