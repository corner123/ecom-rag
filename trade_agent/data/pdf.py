"""Safe PDF extraction, including an optional MinerU CLI adapter."""
from __future__ import annotations

import json
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


class MinerUAdapter:
    def __init__(self, command: tuple[str, ...] = ("mineru",), timeout: int = 45, runner: Callable = subprocess.run):
        self.command, self.timeout, self.runner, self.last_status = command, timeout, runner, "not_run"

    def extract(self, path: Path) -> PdfExtraction | None:
        try:
            with tempfile.TemporaryDirectory(prefix="trade-mineru-") as temp:
                out = Path(temp) / "out"; out.mkdir()
                argv = [*self.command, "-p", str(path), "-o", str(out), "-m", "auto"]
                result = self.runner(argv, shell=False, timeout=self.timeout, capture_output=True, text=True)
                if getattr(result, "returncode", 1) != 0:
                    self.last_status = "nonzero_exit"
                    return None
                for candidate in sorted(out.rglob("*_content_list_v2.json")):
                    units = self._content_list(candidate)
                    if units:
                        self.last_status = "structured_v2"
                        return PdfExtraction(units, "mineru", "content_list")
                for candidate in sorted(out.rglob("*_content_list.json")):
                    units = self._content_list(candidate)
                    if units:
                        self.last_status = "structured_v1"
                        return PdfExtraction(units, "mineru", "content_list")
                for candidate in sorted(out.rglob("*_middle.json")):
                    units = self._middle(candidate)
                    if units:
                        self.last_status = "middle_json"
                        return PdfExtraction(units, "mineru", "middle_json")
                for candidate in sorted(out.rglob("*.md")):
                    text = candidate.read_text(encoding="utf-8", errors="replace").strip()
                    if text:
                        self.last_status = "markdown_degraded"
                        return PdfExtraction([{"text": text, "locator": {"page": 1, "block": 0}}], "mineru", "markdown", ("structured_output_unavailable",))
        except FileNotFoundError:
            self.last_status = "unavailable"
            return None
        except subprocess.TimeoutExpired:
            self.last_status = "timeout"
            return None
        except OSError:
            self.last_status = "execution_error"
            return None
        self.last_status = "no_usable_output"
        return None

    @staticmethod
    def _content_list(path: Path) -> list[dict]:
        try: data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): return []
        values = data if isinstance(data, list) else data.get("content_list", [])
        units = []
        for index, item in enumerate(values):
            if not isinstance(item, dict): continue
            text = str(item.get("text") or item.get("content") or "").strip()
            if not text: continue
            page = item.get("page_idx", item.get("page_no", 0))
            locator = {"page": int(page) + 1, "block": index, "raw": {"bbox": item.get("bbox"), "kind": item.get("type")}}
            if item.get("type") == "table": locator["table"] = f"table-{index}"
            units.append({"text": text, "kind": item.get("type", "text"), "locator": locator, "confidence": item.get("score")})
        return units

    @staticmethod
    def _middle(path: Path) -> list[dict]:
        try: data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): return []
        values = data if isinstance(data, list) else data.get("pdf_info", data.get("pages", []))
        units = []
        for page_index, page in enumerate(values):
            blocks = page.get("para_blocks", page.get("blocks", [])) if isinstance(page, dict) else []
            for index, item in enumerate(blocks):
                lines = item.get("lines", []) if isinstance(item, dict) else []
                nested = " ".join(str(span.get("content", span.get("text", ""))) for line in lines for span in line.get("spans", []))
                text = str(item.get("text") or item.get("content") or nested or "").strip()
                if text:
                    locator = {"page": int(page.get("page_idx", page_index)) + 1, "block": index, "raw": {"bbox": item.get("bbox")}}
                    if item.get("type") == "table": locator["table"] = f"table-{index}"
                    units.append({"text": text, "kind": item.get("type", "text"), "locator": locator, "confidence": item.get("score")})
        return units


def pymupdf_extract(path: Path) -> PdfExtraction:
    import fitz
    document = fitz.open(path); units: list[dict] = []
    try:
        for page_no, page in enumerate(document, 1):
            for block_no, block in enumerate(page.get_text("blocks")):
                text = str(block[4]).strip()
                if text: units.append({"text": text, "kind": "text", "locator": {"page": page_no, "block": block_no, "raw": {"bbox": list(block[:4])}}})
    finally:
        document.close()
    return PdfExtraction(units, "pymupdf", "text", ("mineru_unavailable",))
