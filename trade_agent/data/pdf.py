"""Safe PDF extraction with truthful MinerU capability reporting."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True, slots=True)
class PdfExtraction:
    units: list[dict]
    parser: str
    mode: str
    degraded_components: tuple[str, ...] = ()


class _MalformedCandidate(ValueError):
    pass


class _InvalidItem(ValueError):
    pass


def _strict_page(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _InvalidItem("page_idx must be a nonnegative 0-based integer")
    return value


def _strict_block(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _InvalidItem("block index must be a nonnegative integer")
    return value


def _bbox(value: object) -> list[int | float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise _InvalidItem("bbox must contain four coordinates")
    result: list[int | float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise _InvalidItem("bbox coordinates must be numeric")
        if not math.isfinite(coordinate) or not 0 <= coordinate <= 1000:
            raise _InvalidItem("bbox coordinates must be within 0-1000")
        result.append(coordinate)
    if result[2] < result[0] or result[3] < result[1]:
        raise _InvalidItem("bbox coordinates must be ordered")
    return result


def _confidence(item: dict) -> float | None:
    value = item.get("confidence", item.get("score"))
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise _InvalidItem("confidence must be finite and within 0-1")
    return float(value)


def _caption(value: object) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        return " ".join(" ".join(part.split()) for part in value if part.strip())
    return ""


def _table_text(item: dict) -> str:
    caption = _caption(item.get("table_caption", item.get("caption")))
    body = item.get("table_body", item.get("body"))
    if not isinstance(body, str) or not body.strip():
        body = item.get("table_html", item.get("html"))
        if isinstance(body, str) and body.strip():
            try:
                from bs4 import BeautifulSoup

                body = BeautifulSoup(body, "html.parser").get_text(" ", strip=True)
            except Exception:
                body = ""
    body = body.strip() if isinstance(body, str) else ""
    return "\n".join(part for part in (caption, body) if part)


def _normal_text(item: dict) -> str:
    value = item.get("text", item.get("content"))
    return " ".join(value.split()) if isinstance(value, str) else ""


class MinerUAdapter:
    def __init__(
        self,
        command: tuple[str, ...] = ("mineru",),
        timeout: int = 45,
        runner: Callable = subprocess.run,
    ):
        if not command or not all(isinstance(value, str) and value for value in command):
            raise ValueError("MinerU command must be a nonempty argv tuple")
        if type(timeout) is not int or timeout <= 0:
            raise ValueError("MinerU timeout must be a positive integer")
        self.command = command
        self.timeout = timeout
        self.runner = runner
        self.last_status = "not_run"

    def extract(self, path: Path) -> PdfExtraction | None:
        try:
            with tempfile.TemporaryDirectory(prefix="trade-mineru-") as temp:
                out = Path(temp) / "out"
                out.mkdir()
                argv = [*self.command, "-p", str(path), "-o", str(out), "-m", "auto"]
                result = self.runner(
                    argv,
                    shell=False,
                    timeout=self.timeout,
                    capture_output=True,
                    text=True,
                )
                if getattr(result, "returncode", 1) != 0:
                    self.last_status = "nonzero_exit"
                    return None

                saw_output = False
                saw_malformed = False
                stages = (
                    ("*_content_list_v2.json", self._content_list_strict, "structured_v2", "content_list"),
                    ("*_content_list.json", self._content_list_strict, "structured_v1", "content_list"),
                    ("*_middle.json", self._middle_strict, "middle_json", "middle_json"),
                )
                for pattern, parser, status, mode in stages:
                    for candidate in sorted(out.rglob(pattern)):
                        saw_output = True
                        try:
                            units = parser(candidate)
                        except Exception:
                            saw_malformed = True
                            continue
                        if units:
                            self.last_status = status
                            return PdfExtraction(units, "mineru", mode)

                for candidate in sorted(out.rglob("*.md")):
                    saw_output = True
                    try:
                        markdown = candidate.read_text(encoding="utf-8").strip()
                    except (OSError, UnicodeError):
                        saw_malformed = True
                        continue
                    if markdown:
                        self.last_status = "markdown_degraded"
                        locator = {"page": 1, "block": 0, "raw": {"kind": "markdown"}}
                        return PdfExtraction(
                            [{"text": markdown, "kind": "markdown", "locator": locator}],
                            "mineru",
                            "markdown",
                            ("structured_output_unavailable",),
                        )

                if not saw_output:
                    self.last_status = "no_output"
                elif saw_malformed:
                    self.last_status = "malformed_output"
                else:
                    self.last_status = "no_usable_output"
                return None
        except FileNotFoundError:
            self.last_status = "unavailable"
            return None
        except subprocess.TimeoutExpired:
            self.last_status = "timeout"
            return None
        except OSError:
            self.last_status = "execution_error"
            return None

    @staticmethod
    def _read_json(path: Path) -> object:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise _MalformedCandidate("invalid JSON candidate") from exc

    @classmethod
    def _content_list_strict(cls, path: Path) -> list[dict]:
        data = cls._read_json(path)
        if isinstance(data, list):
            values = data
        elif isinstance(data, dict) and isinstance(data.get("content_list"), list):
            values = data["content_list"]
        else:
            raise _MalformedCandidate("content-list must be a list")

        units: list[dict] = []
        invalid_items = 0
        for index, item in enumerate(values):
            if not isinstance(item, dict):
                continue
            try:
                page_index = _strict_page(item.get("page_idx", item.get("page_no", 0)))
                block_index = _strict_block(item.get("block_idx", index))
                bbox = _bbox(item.get("bbox"))
                confidence = _confidence(item)
                kind = item.get("type", "text")
                if not isinstance(kind, str) or not kind:
                    raise _InvalidItem("type must be a string")
                text = _table_text(item) if kind == "table" else _normal_text(item)
                if not text:
                    continue
                raw = {"bbox": bbox, "kind": kind}
                if item.get("text_level") is not None:
                    text_level = item["text_level"]
                    if isinstance(text_level, bool) or not isinstance(text_level, (int, str)):
                        raise _InvalidItem("text_level must be a string or integer")
                    raw["text_level"] = text_level
                locator: dict = {"page": page_index + 1, "block": block_index, "raw": raw}
                if kind == "table":
                    locator["table"] = f"table-{page_index + 1}-{block_index}"
                unit = {"text": text, "kind": kind, "locator": locator}
                if confidence is not None:
                    unit["confidence"] = confidence
                units.append(unit)
            except _InvalidItem:
                invalid_items += 1
                continue
        if invalid_items and not units:
            raise _MalformedCandidate("content-list contains only invalid items")
        return units

    @classmethod
    def _content_list(cls, path: Path) -> list[dict]:
        try:
            return cls._content_list_strict(path)
        except Exception:
            return []

    @classmethod
    def _middle_strict(cls, path: Path) -> list[dict]:
        data = cls._read_json(path)
        if isinstance(data, list):
            pages = data
        elif isinstance(data, dict):
            pages = data.get("pdf_info", data.get("pages"))
        else:
            pages = None
        if not isinstance(pages, list):
            raise _MalformedCandidate("middle JSON must contain page list")

        units: list[dict] = []
        invalid_items = 0
        for page_position, page in enumerate(pages):
            if not isinstance(page, dict):
                continue
            try:
                page_index = _strict_page(page.get("page_idx", page_position))
            except _InvalidItem:
                invalid_items += 1
                continue
            blocks = page.get("para_blocks", page.get("blocks", []))
            tables = page.get("tables", [])
            if not isinstance(blocks, list) or not isinstance(tables, list):
                continue
            for block_position, item in enumerate([*blocks, *tables]):
                if not isinstance(item, dict):
                    continue
                try:
                    block_index = _strict_block(item.get("block_idx", block_position))
                    bbox = _bbox(item.get("bbox"))
                    confidence = _confidence(item)
                    kind = item.get("type", "text")
                    if not isinstance(kind, str) or not kind:
                        raise _InvalidItem("type must be a string")
                    if kind == "table" or any(key in item for key in ("table_body", "table_html")):
                        kind = "table"
                        text = _table_text(item)
                    else:
                        text = _normal_text(item)
                        if not text:
                            lines = item.get("lines", [])
                            if not isinstance(lines, list):
                                raise _InvalidItem("lines must be a list")
                            fragments: list[str] = []
                            for line in lines:
                                spans = line.get("spans", []) if isinstance(line, dict) else []
                                if not isinstance(spans, list):
                                    continue
                                for span in spans:
                                    if not isinstance(span, dict):
                                        continue
                                    value = span.get("content", span.get("text", ""))
                                    if isinstance(value, str):
                                        fragments.append(value)
                            text = " ".join("".join(fragments).split())
                    if not text:
                        continue
                    raw = {"bbox": bbox, "kind": kind}
                    locator: dict = {"page": page_index + 1, "block": block_index, "raw": raw}
                    if kind == "table":
                        locator["table"] = f"table-{page_index + 1}-{block_index}"
                    unit = {"text": text, "kind": kind, "locator": locator}
                    if confidence is not None:
                        unit["confidence"] = confidence
                    units.append(unit)
                except _InvalidItem:
                    invalid_items += 1
                    continue
        if invalid_items and not units:
            raise _MalformedCandidate("middle JSON contains only invalid items")
        return units

    @classmethod
    def _middle(cls, path: Path) -> list[dict]:
        try:
            return cls._middle_strict(path)
        except Exception:
            return []


def pymupdf_extract(path: Path, *, mineru_status: str = "not_run") -> PdfExtraction:
    import pymupdf

    document = pymupdf.open(path)
    units: list[dict] = []
    try:
        for page_no, page in enumerate(document, 1):
            for block_no, block in enumerate(page.get_text("blocks")):
                text = str(block[4]).strip()
                if text:
                    units.append(
                        {
                            "text": text,
                            "kind": "text",
                            "locator": {
                                "page": page_no,
                                "block": block_no,
                                "raw": {"bbox": list(block[:4]), "kind": "text"},
                            },
                        }
                    )
    finally:
        document.close()
    return PdfExtraction(units, "pymupdf", "text", (f"mineru_{mineru_status}",))
