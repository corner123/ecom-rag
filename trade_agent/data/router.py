"""Fail-closed physical and business source routing."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator, model_validator
from trade_agent.schemas.source import _aware, _validate_json_value

from trade_agent.data.pdf import MinerUAdapter, pymupdf_extract
from trade_agent.data.quarantine import QuarantineRecord, sanitize_diagnostic
from trade_agent.schemas.source import DocumentRecord, FileType, SourceType, content_sha256, stable_id

_EXTENSIONS = {FileType.HTML: {".html", ".htm"}, FileType.MARKDOWN: {".md", ".markdown"}, FileType.JSON: {".json"}, FileType.JSONL: {".jsonl"}, FileType.PDF: {".pdf"}, FileType.GENERATED_PROFILE: {".json"}}
_PAIRS = {SourceType.OFFICIAL_WEBSITE: {FileType.HTML, FileType.MARKDOWN}, SourceType.B2B: {FileType.JSON, FileType.HTML}, SourceType.INDUSTRY_NEWS: {FileType.JSON, FileType.HTML, FileType.PDF}, SourceType.SOCIAL: {FileType.JSONL}, SourceType.REGULATOR: {FileType.PDF}, SourceType.CUSTOMS_PROFILE: {FileType.GENERATED_PROFILE}}


class SourceInput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    path: Path
    file_type: FileType
    source_type: SourceType
    source_id: StrictStr
    title: StrictStr
    language: StrictStr
    fetched_at: datetime
    is_synthetic: StrictBool
    source_url: AnyUrl | None = None
    canonical_url: AnyUrl | None = None
    publish_time: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    license_scope: StrictStr | None = None
    manifest_attributes: dict[str, Any] = Field(default_factory=dict)
    @field_validator("source_id", "title", "language")
    @classmethod
    def nonblank(cls, value):
        if not value.strip(): raise ValueError("must not be blank")
        return value
    _times = field_validator("fetched_at", "publish_time", "valid_from", "valid_to")(_aware)
    @field_validator("license_scope")
    @classmethod
    def nonblank_optional(cls, value):
        if value is not None and not value.strip(): raise ValueError("must not be blank")
        return value
    @model_validator(mode="after")
    def ranges_and_attrs(self):
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to: raise ValueError("valid_from must not follow valid_to")
        _validate_json_value(self.manifest_attributes, "manifest_attributes")
        return self


class DocumentRouter:
    def __init__(self, *, max_bytes: int = 20_000_000, max_documents: int = 10_000, mineru: MinerUAdapter | None = None):
        self.max_bytes, self.max_documents, self.mineru = max_bytes, max_documents, mineru or MinerUAdapter()
        self.quarantines: list[QuarantineRecord] = []

    def _quarantine(self, code: str, source: SourceInput, parser: str, diagnostic: object) -> list[DocumentRecord]:
        self.quarantines.append(QuarantineRecord(code, str(source.path), parser, sanitize_diagnostic(diagnostic)))
        return []

    def load(self, source: SourceInput) -> list[DocumentRecord]:
        try:
            size = source.path.stat().st_size
        except OSError as exc: return self._quarantine("SOURCE_UNREADABLE", source, "router", exc)
        if size > self.max_bytes: return self._quarantine("INPUT_TOO_LARGE", source, "router", size)
        try: raw = source.path.read_bytes()
        except OSError as exc: return self._quarantine("SOURCE_UNREADABLE", source, "router", exc)
        if source.path.suffix.lower() not in _EXTENSIONS[source.file_type]: return self._quarantine("FILE_TYPE_MISMATCH", source, "router", "extension conflicts with declared file type")
        if source.file_type is FileType.PDF and not raw.startswith(b"%PDF"): return self._quarantine("FILE_TYPE_MISMATCH", source, "pdf", "PDF signature missing")
        if source.file_type is not FileType.PDF and raw.startswith(b"%PDF"): return self._quarantine("FILE_TYPE_MISMATCH", source, "router", "PDF signature conflicts with declared file type")
        if source.file_type in {FileType.JSON, FileType.JSONL, FileType.GENERATED_PROFILE} and raw.lstrip()[:1] not in {b"{", b"["}: return self._quarantine("FILE_TYPE_MISMATCH", source, "json", "JSON signature missing")
        if source.file_type is FileType.HTML and b"<" not in raw[:1024]: return self._quarantine("FILE_TYPE_MISMATCH", source, "html", "HTML signature missing")
        if source.file_type not in _PAIRS.get(source.source_type, set()): return self._quarantine("UNSUPPORTED_SOURCE_FILE_PAIR", source, "router", "declared source/file pair is not supported")
        try:
            if source.file_type is FileType.PDF: docs = self._pdf(source)
            elif source.file_type is FileType.HTML: docs = self._sections(source, raw.decode("utf-8"), html=True)
            elif source.file_type is FileType.MARKDOWN: docs = self._sections(source, raw.decode("utf-8"), html=False)
            elif source.file_type is FileType.JSONL: docs = self._jsonl(source, raw.decode("utf-8"))
            else: docs = self._json(source, raw.decode("utf-8"))
        except UnicodeDecodeError as exc: return self._quarantine("TEXT_DECODE_FAILED", source, "parser", exc)
        except json.JSONDecodeError as exc: return self._quarantine("MALFORMED_JSON", source, "json", exc)
        except Exception as exc: return self._quarantine("PARSE_FAILED", source, "parser", exc)
        if len(docs) > self.max_documents: return self._quarantine("DOCUMENT_COUNT_LIMIT", source, "parser", len(docs))
        return docs

    def _attrs(self, source: SourceInput, item: dict[str, Any]) -> dict[str, Any]:
        attrs = {**source.manifest_attributes, "publish_time": source.publish_time.isoformat() if source.publish_time else None, "valid_from": source.valid_from.isoformat() if source.valid_from else None, "valid_to": source.valid_to.isoformat() if source.valid_to else None, "license_scope": source.license_scope, **{k:v for k,v in item.items() if v is not None and v != ""}}
        attrs.setdefault("source_weight", 0.5); attrs.setdefault("source_weight_version", "task6-v1-provisional")
        return attrs
    def _record(self, source, title, content, attributes, units, identity):
        return DocumentRecord(document_id=stable_id("doc", source.source_id, identity), source_id=source.source_id, file_type=source.file_type, source_type=source.source_type, title=title, language=source.language, content=content, content_hash=content_sha256(content), fetched_at=source.fetched_at, is_synthetic=source.is_synthetic, attributes=attributes, units=units, source_url=attributes.get("url", source.source_url), canonical_url=attributes.get("canonical_url", source.canonical_url))
    def _json(self, source, text):
        payload = json.loads(text)
        if source.file_type is FileType.GENERATED_PROFILE:
            content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            attrs = self._attrs(source, payload); attrs["aggregation_info"] = {k: payload[k] for k in ("aggregation_grain", "aggregation_window", "calendar_month", "source_record_count", "raw_record_summary") if k in payload}
            return [self._record(source, payload.get("company", source.title), content, attrs, [{"text": content, "locator": {"profile": "monthly_company_hs"}}], payload.get("calendar_month", source.path.name))]
        values = payload.get("products") if source.source_type is SourceType.B2B else payload.get("stories")
        if not isinstance(values, list): raise ValueError("expected products or stories list")
        docs=[]
        for row, item in enumerate(values, 1):
            title = item.get("product_name") or item.get("headline")
            content = item.get("description") or item.get("body") or json.dumps(item, sort_keys=True)
            locator = {"row": row, "raw": {"item_id": item.get("product_id") or item.get("id")}}
            docs.append(self._record(source, title, content, self._attrs(source, item), [{"text": content, "locator": locator}], str(item.get("product_id") or item.get("id") or row)))
        return docs
    def _jsonl(self, source, text):
        docs=[]
        for row, line in enumerate(text.splitlines(), 1):
            if not line.strip(): continue
            item=json.loads(line); content=item["text"]; post=item["post_id"]
            docs.append(self._record(source, post, content, self._attrs(source, item), [{"text": content, "locator": {"post_id": post, "row": row}}], post))
        return docs
    def _sections(self, source, text, html):
        sections=[]
        if html:
            soup=BeautifulSoup(text, "html.parser"); current=""
            dom_sections = soup.find_all("section")
            for section in dom_sections:
                heading = section.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])
                value = section.get_text(" ", strip=True)
                if value: sections.append(((heading.get_text(" ", strip=True) if heading else section.get("id") or source.title), value))
            if not dom_sections:
                for element in soup.find_all(["h1","h2","h3","h4","h5","h6","p","li"]):
                    value=element.get_text(" ", strip=True)
                    if element.name.startswith("h"): current=value
                    elif value: sections.append((current or source.title, value))
        else:
            current=source.title; buffer=[]
            for line in text.splitlines():
                if line.startswith("#"):
                    if buffer: sections.append((current, "\n".join(buffer).strip())); buffer=[]
                    current=line.lstrip("#").strip()
                else: buffer.append(line)
            if buffer: sections.append((current, "\n".join(buffer).strip()))
        return [self._record(source, heading, body, self._attrs(source, {}), [{"text": body, "locator": {"section": heading}}], f"{heading}:{i}") for i,(heading,body) in enumerate(sections) if body]
    def _pdf(self, source):
        extraction=self.mineru.extract(source.path) or pymupdf_extract(source.path)
        if not extraction.units: return self._quarantine("SCANNED_PDF_OCR_UNAVAILABLE", source, extraction.parser, "no usable OCR or text units")
        content="\n\n".join(unit["text"] for unit in extraction.units)
        attrs=self._attrs(source, {"parser": extraction.parser, "parser_mode": extraction.mode, "mineru_status": self.mineru.last_status, "degraded_components": list(extraction.degraded_components)})
        return [self._record(source, source.title, content, attrs, extraction.units, source.path.name)]


def probe_manifest(corpus_root: Path, manifest_path: Path) -> dict[str, Any]:
    """Route every manifest record deterministically without touching any SQL ledger."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    router = DocumentRouter()
    documents = 0
    for record in sorted(payload["records"], key=lambda item: item["path"]):
        source_type = SourceType(record["source_type"])
        if source_type is SourceType.TRADE_LEDGER:
            continue
        when = datetime.fromisoformat(record["ingested_at"])
        source = SourceInput(
            path=corpus_root / record["path"], file_type=FileType(record["file_type"]), source_type=source_type,
            source_id=stable_id("source", record["path"]), title=record.get("expected_entity", record["path"]),
            language="en", fetched_at=when, is_synthetic=record["is_synthetic"], source_url="https://ingest.example/manifest",
            publish_time=datetime.fromisoformat(record["publish_time"]), valid_from=datetime.fromisoformat(record["valid_from"]),
            manifest_attributes={key: value for key, value in record.items() if key not in {"path", "file_type", "source_type", "is_synthetic"}},
        )
        documents += len(router.load(source))
    return {"documents": documents, "quarantines": [{"error_code": item.error_code, "path": str(Path(item.source_path).relative_to(corpus_root))} for item in router.quarantines]}
