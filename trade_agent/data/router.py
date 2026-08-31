"""Fail-closed physical routing and source-specific normalization."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from trade_agent.data.parsers import (
    ParseFailure,
    json_value,
    jsonl_values,
    parse_html_sections,
    parse_markdown_sections,
)
from trade_agent.data.pdf import MinerUAdapter, pymupdf_extract
from trade_agent.data.quarantine import QuarantineRecord, sanitize_diagnostic
from trade_agent.schemas.source import (
    DocumentRecord,
    FileType,
    SourceType,
    _aware,
    _reserved_synthetic_url,
    _validate_json_value,
    content_sha256,
    stable_id,
)

_EXTENSIONS = {
    FileType.HTML: {".html", ".htm"},
    FileType.MARKDOWN: {".md", ".markdown"},
    FileType.JSON: {".json"},
    FileType.JSONL: {".jsonl"},
    FileType.PDF: {".pdf"},
    FileType.GENERATED_PROFILE: {".json"},
}
_PAIRS = {
    SourceType.OFFICIAL_WEBSITE: {FileType.HTML, FileType.MARKDOWN},
    SourceType.B2B: {FileType.JSON, FileType.HTML},
    SourceType.INDUSTRY_NEWS: {FileType.JSON, FileType.HTML, FileType.PDF},
    SourceType.SOCIAL: {FileType.JSONL},
    SourceType.REGULATOR: {FileType.PDF},
    SourceType.CUSTOMS_PROFILE: {FileType.GENERATED_PROFILE},
}
_URL = TypeAdapter(AnyUrl)
_HTML_DOCUMENT = re.compile(
    r"(?is)^\s*(?:<!--.*?-->\s*)*(?:<!doctype\s+html\b|<(?:html|head|body|main|article|section|h[1-6]|p|div|nav|aside|table|ul|ol|dl|header|footer|blockquote|pre)\b)"
)
_XML_OR_SVG_DOCUMENT = re.compile(
    r"(?is)^\s*(?:<!--.*?-->\s*)*(?:(?:<\?xml\b|<!doctype\s+(?!html\b)|<svg\b).*|"
    r"<(?P<root>[A-Za-z_][\w:.-]*)(?:\s+[^<>]*)?(?:/>|>.*?</(?P=root)\s*>))\s*$"
)
_PROFILE_MARKERS = frozenset({"company", "company_id", "country_code", "hs_code", "calendar_month", "aggregation_grain"})
_HS_CODE = re.compile(r"^[0-9]{4,10}$")
_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_SYSTEM_ATTRIBUTE_KEYS = {
    "feed_source_url", "feed_canonical_url", "item_source_url", "item_canonical_url",
    "source_weight", "source_weight_version", "fact_type", "publish_time", "published_at",
    "valid_from", "valid_to", "license_scope", "aggregation_info", "ocr_confidence",
    "source_payload",
}


class MediaKind(str, Enum):
    """Physical media categories accepted by the bounded router sniff."""

    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    JSONL = "jsonl"
    GENERATED_PROFILE = "generated_profile"
    TEXT = "text"
    XML_OR_SVG = "xml_or_svg"
    ARCHIVE = "archive"
    IMAGE = "image"
    BINARY = "binary"


_ARCHIVE_SIGNATURES = (
    b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08", b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00",
    b"7z\xbc\xaf\x27\x1c", b"Rar!\x1a\x07\x00", b"Rar!\x1a\x07\x01\x00",
)
_IMAGE_SIGNATURES = (
    b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"II*\x00", b"MM\x00*",
)


def _is_archive(raw: bytes) -> bool:
    return raw.startswith(_ARCHIVE_SIGNATURES) or (len(raw) >= 262 and raw[257:262] == b"ustar")


def _is_image(raw: bytes) -> bool:
    if raw.startswith(_IMAGE_SIGNATURES) or (raw.startswith(b"RIFF") and raw[8:12] == b"WEBP"):
        return True
    if raw.startswith(b"BM") and len(raw) >= 26:
        declared_size = int.from_bytes(raw[2:6], "little")
        dib_size = int.from_bytes(raw[14:18], "little")
        return 26 <= declared_size <= len(raw) and dib_size >= 12
    if raw.startswith(b"\x00\x00\x01\x00") and len(raw) >= 6:
        image_count = int.from_bytes(raw[4:6], "little")
        return 1 <= image_count <= 10_000 and len(raw) >= 6 + image_count * 16
    return False


def _json_media_kind(text: str) -> MediaKind:
    """Classify only unambiguous JSON records; malformed JSON stays parser-owned."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) > 1:
            try:
                first = json.loads(lines[0])
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(first, dict):
                    return MediaKind.JSONL
        # A malformed JSON candidate is still physically JSON-like.  Keeping
        # it in this family makes Markdown reject it while the declared JSON
        # parser retains responsibility for its MALFORMED_JSON diagnosis.
        return MediaKind.JSONL if '"post_id"' in text else MediaKind.JSON
    if isinstance(value, dict):
        if _PROFILE_MARKERS.issubset(value):
            return MediaKind.GENERATED_PROFILE
        if "post_id" in value:
            # A single JSONL record is byte-identical to a JSON object; its
            # social record shape is the only deterministic discriminator.
            return MediaKind.JSONL
    return MediaKind.JSON


def sniff_media_kind(raw: bytes) -> MediaKind:
    """Return a deterministic physical-media kind from bounded ingress bytes.

    This is deliberately a media boundary, not a replacement for JSON or
    business-shape parsing.  Text that is merely malformed remains available
    to the declared structured parser so it can report its stable typed error.
    ``DocumentRouter`` supplies only the already bounded ``max_bytes + 1``
    read, and this function performs no filesystem I/O.
    """
    sample = raw
    if sample.startswith(b"%PDF"):
        return MediaKind.PDF
    if _is_archive(sample):
        return MediaKind.ARCHIVE
    if _is_image(sample):
        return MediaKind.IMAGE
    try:
        text = sample.decode("utf-8")
    except UnicodeDecodeError:
        return MediaKind.BINARY
    if "\x00" in text or any(ord(char) < 32 and char not in "\t\n\r\f" for char in text):
        return MediaKind.BINARY

    stripped = text.lstrip("\ufeff")
    if _HTML_DOCUMENT.match(stripped):
        return MediaKind.HTML
    if _XML_OR_SVG_DOCUMENT.match(stripped):
        return MediaKind.XML_OR_SVG
    if stripped.lstrip().startswith(("{", "[")):
        return _json_media_kind(stripped)
    return MediaKind.TEXT


_MEDIA_COMPATIBILITY = {
    FileType.PDF: {MediaKind.PDF},
    FileType.HTML: {MediaKind.HTML},
    FileType.MARKDOWN: {MediaKind.TEXT},
    # Text is admitted here only to retain parser-owned malformed-JSON errors.
    FileType.JSON: {MediaKind.JSON, MediaKind.TEXT},
    FileType.JSONL: {MediaKind.JSONL, MediaKind.TEXT},
    FileType.GENERATED_PROFILE: {MediaKind.GENERATED_PROFILE, MediaKind.JSON, MediaKind.TEXT},
}


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

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False) -> "SourceInput":
        if update is None:
            return super().model_copy(deep=deep)
        data = self.model_dump(mode="python")
        if deep:
            data = deepcopy(data)
        data.update(update)
        return type(self).model_validate(data)

    @field_validator("source_id", "title", "language")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    _times = field_validator("fetched_at", "publish_time", "valid_from", "valid_to")(_aware)

    @field_validator("license_scope")
    @classmethod
    def nonblank_optional(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def ranges_and_attrs(self) -> "SourceInput":
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to:
            raise ValueError("valid_from must not follow valid_to")
        _validate_json_value(self.manifest_attributes, "manifest_attributes")
        if self.is_synthetic and any(
            url is not None and not _reserved_synthetic_url(url)
            for url in (self.source_url, self.canonical_url)
        ):
            raise ValueError("synthetic source inputs must use reserved example hosts")
        return self


def _required_str(item: dict[str, Any], key: str, parser: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ParseFailure("INVALID_DOCUMENT_SHAPE", parser, f"{key} must be a nonblank string")
    return value


def _required_int(item: dict[str, Any], key: str, parser: str, *, minimum: int = 0) -> int:
    value = item.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ParseFailure("INVALID_DOCUMENT_SHAPE", parser, f"{key} must be an integer >= {minimum}")
    return value


def _required_url(item: dict[str, Any], key: str, parser: str) -> str:
    value = _required_str(item, key, parser)
    try:
        return str(_URL.validate_python(value))
    except ValidationError as exc:
        raise ParseFailure("INVALID_DOCUMENT_SHAPE", parser, f"{key} must be a URL") from exc


def _required_time(item: dict[str, Any], key: str, parser: str) -> str:
    value = _required_str(item, key, parser)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ParseFailure("INVALID_DOCUMENT_SHAPE", parser, f"{key} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ParseFailure("INVALID_DOCUMENT_SHAPE", parser, f"{key} must include a timezone")
    return value


class DocumentRouter:
    def __init__(
        self,
        *,
        max_bytes: int = 20_000_000,
        max_documents: int = 10_000,
        mineru: MinerUAdapter | None = None,
    ):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if type(max_documents) is not int or max_documents <= 0:
            raise ValueError("max_documents must be a positive integer")
        self.max_bytes = max_bytes
        self.max_documents = max_documents
        self.mineru = mineru or MinerUAdapter()
        self.quarantines: list[QuarantineRecord] = []

    def _quarantine(self, code: str, source: SourceInput, parser: str, diagnostic: object) -> list[DocumentRecord]:
        source_path = str(source.path)
        self.quarantines.append(
            QuarantineRecord(
                error_code=code,
                source_path=source_path,
                parser=parser,
                diagnostic=sanitize_diagnostic(diagnostic, source_path=source_path),
            )
        )
        return []

    def load(self, source: SourceInput) -> list[DocumentRecord]:
        try:
            size = source.path.stat().st_size
        except OSError as exc:
            return self._quarantine("SOURCE_UNREADABLE", source, "router", exc)
        if size > self.max_bytes:
            return self._quarantine("INPUT_TOO_LARGE", source, "router", f"stat size {size} exceeds limit")
        try:
            with source.path.open("rb") as handle:
                raw = handle.read(self.max_bytes + 1)
        except OSError as exc:
            return self._quarantine("SOURCE_UNREADABLE", source, "router", exc)
        if len(raw) > self.max_bytes:
            return self._quarantine("INPUT_TOO_LARGE", source, "router", "bounded read exceeded limit")

        if source.path.suffix.lower() not in _EXTENSIONS[source.file_type]:
            return self._quarantine("FILE_TYPE_MISMATCH", source, "router", "extension conflicts with declared file type")
        if source.file_type not in _PAIRS.get(source.source_type, set()):
            return self._quarantine("UNSUPPORTED_SOURCE_FILE_PAIR", source, "router", "declared source/file pair is not supported")
        media_kind = sniff_media_kind(raw)
        if media_kind not in _MEDIA_COMPATIBILITY[source.file_type]:
            return self._quarantine(
                "FILE_TYPE_MISMATCH",
                source,
                "router",
                f"content media {media_kind.value} conflicts with declared {source.file_type.value}",
            )

        try:
            if source.file_type is FileType.PDF:
                docs = self._pdf(source)
            else:
                text = raw.decode("utf-8")
                if source.file_type is FileType.HTML:
                    docs = self._sections(source, text, html=True)
                elif source.file_type is FileType.MARKDOWN:
                    docs = self._sections(source, text, html=False)
                elif source.file_type is FileType.JSONL:
                    docs = self._jsonl(source, text)
                else:
                    docs = self._json(source, text)
        except UnicodeDecodeError as exc:
            return self._quarantine("TEXT_DECODE_FAILED", source, "text", exc)
        except ParseFailure as exc:
            return self._quarantine(exc.code, source, exc.parser, exc.diagnostic)
        except ValidationError as exc:
            return self._quarantine("INVALID_DOCUMENT_SHAPE", source, "schema", exc)
        except Exception as exc:  # Unknown implementation failures remain visible and fail closed.
            return self._quarantine("PARSE_FAILED", source, "parser", exc)
        if len(docs) > self.max_documents:
            return self._quarantine("DOCUMENT_COUNT_LIMIT", source, "parser", "document limit reached")
        return docs

    @staticmethod
    def _feed_urls(source: SourceInput) -> tuple[str | None, str | None]:
        return (
            str(source.source_url) if source.source_url else None,
            str(source.canonical_url) if source.canonical_url else None,
        )

    def _attrs(
        self,
        source: SourceInput,
        item: dict[str, Any],
        *,
        item_source_url: str | None = None,
        item_canonical_url: str | None = None,
    ) -> dict[str, Any]:
        feed_source_url, feed_canonical_url = self._feed_urls(source)
        catalog = source.manifest_attributes
        attrs = {
            key: value
            for key, value in item.items()
            if key not in _SYSTEM_ATTRIBUTE_KEYS and key not in catalog
        }
        if item:
            attrs["source_payload"] = deepcopy(item)
        attrs.update(catalog)
        attrs.update({
            "feed_source_url": feed_source_url,
            "feed_canonical_url": feed_canonical_url,
            "item_source_url": item_source_url or feed_source_url,
            "item_canonical_url": item_canonical_url or feed_canonical_url or item_source_url or feed_source_url,
            "publish_time": (
                source.publish_time.isoformat()
                if source.publish_time
                else catalog.get("publish_time")
                or (item.get("published_at") if source.source_type in {SourceType.INDUSTRY_NEWS, SourceType.SOCIAL} else None)
            ),
            "valid_from": source.valid_from.isoformat() if source.valid_from else catalog.get("valid_from"),
            "valid_to": source.valid_to.isoformat() if source.valid_to else catalog.get("valid_to"),
            "license_scope": source.license_scope if source.license_scope is not None else catalog.get("license_scope"),
            "source_weight": catalog.get("source_weight", 0.5),
            "source_weight_version": catalog.get("source_weight_version", "task6-v1-provisional"),
            "fact_type": catalog.get("fact_type"),
        })
        attrs = {key: value for key, value in attrs.items() if value is not None and value != ""}
        _validate_json_value(attrs, "document_attributes")
        return attrs

    @staticmethod
    def _b2b_evidence_text(item: dict[str, Any]) -> str:
        narrative = next(
            (
                value.strip()
                for key in ("description", "body")
                if isinstance((value := item.get(key)), str) and value.strip()
            ),
            None,
        )
        if narrative is None:
            return json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

        narrative_folded = narrative.casefold()

        def appears_standalone(value: str) -> bool:
            return re.search(
                rf"(?<!\w){re.escape(value.casefold())}(?!\w)",
                narrative_folded,
            ) is not None

        seen: set[str] = set()
        parts: list[str] = []
        for label, key in (
            ("Supplier", "supplier"),
            ("Product", "product_name"),
            ("Category", "category"),
            ("SKU", "sku"),
            ("HS", "hs_code"),
        ):
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            value = value.strip()
            folded = value.casefold()
            if folded in seen or appears_standalone(value):
                continue
            seen.add(folded)
            parts.append(f"{label}: {value}")
        parts.append(narrative)
        return "\n".join(parts)

    def _raw_provenance(
        self,
        source: SourceInput,
        *,
        item_source_url: str | None,
        item_canonical_url: str | None,
        values: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        feed_source_url, feed_canonical_url = self._feed_urls(source)
        raw = {
            "feed_source_url": feed_source_url,
            "feed_canonical_url": feed_canonical_url,
            "item_source_url": item_source_url or feed_source_url,
            "item_canonical_url": item_canonical_url or feed_canonical_url or item_source_url or feed_source_url,
            "manifest_locator": source.manifest_attributes.get("locator"),
            "manifest_claim_ids": source.manifest_attributes.get("reference_claim_ids"),
            **(values or {}),
        }
        return {key: value for key, value in raw.items() if value is not None and value != ""}

    @staticmethod
    def _record(
        source: SourceInput,
        title: str,
        content: str,
        attributes: dict[str, Any],
        units: list[dict[str, Any]],
        identity: str,
    ) -> DocumentRecord:
        return DocumentRecord(
            document_id=stable_id("doc", source.source_id, identity),
            source_id=source.source_id,
            file_type=source.file_type,
            source_type=source.source_type,
            title=title,
            language=source.language,
            content=content,
            content_hash=content_sha256(content),
            fetched_at=source.fetched_at,
            is_synthetic=source.is_synthetic,
            attributes=attributes,
            units=units,
            source_url=attributes.get("item_source_url"),
            canonical_url=attributes.get("item_canonical_url"),
        )

    def _validate_profile(self, payload: dict[str, Any]) -> None:
        parser = "generated_profile"
        for key in ("company", "country_code", "hs_code", "calendar_month", "aggregation_grain", "currency"):
            _required_str(payload, key, parser)
        _required_int(payload, "company_id", parser, minimum=1)
        _required_int(payload, "source_record_count", parser)
        if not re.fullmatch(r"[A-Z]{2}", payload["country_code"]):
            raise ParseFailure("INVALID_DOCUMENT_SHAPE", parser, "country_code must be ISO alpha-2")
        if not _HS_CODE.fullmatch(payload["hs_code"]):
            raise ParseFailure("INVALID_DOCUMENT_SHAPE", parser, "hs_code must be 4-10 digits")
        if not _MONTH.fullmatch(payload["calendar_month"]):
            raise ParseFailure("INVALID_DOCUMENT_SHAPE", parser, "calendar_month must be YYYY-MM")
        if not isinstance(payload.get("aggregation_window"), dict) or not isinstance(payload.get("raw_record_summary"), dict):
            raise ParseFailure("INVALID_DOCUMENT_SHAPE", parser, "aggregation window and raw record summary must be objects")
        if not isinstance(payload.get("roles_included"), list) or not all(isinstance(value, str) and value for value in payload["roles_included"]):
            raise ParseFailure("INVALID_DOCUMENT_SHAPE", parser, "roles_included must be a string list")
        for key in ("total_amount_usd", "total_quantity_kg", "import_amount_usd", "export_amount_usd", "import_quantity_kg", "export_quantity_kg"):
            _required_str(payload, key, parser)

    def _validate_b2b(self, item: dict[str, Any]) -> None:
        parser = "b2b"
        product_id = item.get("product_id")
        if isinstance(product_id, bool) or not isinstance(product_id, (int, str)) or not str(product_id).strip():
            raise ParseFailure("INVALID_DOCUMENT_SHAPE", parser, "product_id must be a string or integer")
        for key in ("product_name", "sku", "supplier", "hs_code", "reference_claim_id"):
            _required_str(item, key, parser)
        if not _HS_CODE.fullmatch(item["hs_code"]):
            raise ParseFailure("INVALID_DOCUMENT_SHAPE", parser, "hs_code must be 4-10 digits")
        _required_url(item, "url", parser)

    def _validate_news(self, item: dict[str, Any]) -> None:
        parser = "industry_news"
        for key in (
            "id", "headline", "body", "entity", "publisher", "canonical_story_id",
            "dedupe_cluster_id", "reference_claim_id",
        ):
            _required_str(item, key, parser)
        _required_url(item, "url", parser)
        _required_time(item, "published_at", parser)
        if item.get("syndicated_from") is not None and (not isinstance(item["syndicated_from"], str) or not item["syndicated_from"].strip()):
            raise ParseFailure("INVALID_DOCUMENT_SHAPE", parser, "syndicated_from must be null or nonblank string")

    def _validate_social(self, item: dict[str, Any]) -> None:
        parser = "social"
        for key in ("post_id", "text", "company", "reference_claim_id"):
            _required_str(item, key, parser)
        _required_url(item, "url", parser)
        _required_time(item, "published_at", parser)

    def _json(self, source: SourceInput, text: str) -> list[DocumentRecord]:
        payload = json_value(text)
        if not isinstance(payload, dict):
            raise ParseFailure("INVALID_DOCUMENT_SHAPE", "json", "top-level payload must be object")
        if source.file_type is FileType.GENERATED_PROFILE:
            self._validate_profile(payload)
            content = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            item_url = str(source.source_url) if source.source_url else None
            canonical_url = str(source.canonical_url) if source.canonical_url else item_url
            attrs = self._attrs(source, payload, item_source_url=item_url, item_canonical_url=canonical_url)
            attrs["aggregation_info"] = {
                key: payload[key]
                for key in ("aggregation_grain", "aggregation_window", "calendar_month", "source_record_count", "raw_record_summary")
            }
            raw = self._raw_provenance(
                source,
                item_source_url=item_url,
                item_canonical_url=canonical_url,
                values={key: payload[key] for key in ("company_id", "country_code", "hs_code", "calendar_month", "aggregation_grain", "source_record_count")},
            )
            locator = {"profile": "monthly_company_hs", "raw": raw}
            identity = f"{payload['company_id']}:{payload['country_code']}:{payload['hs_code']}:{payload['calendar_month']}"
            return [self._record(source, payload["company"], content, attrs, [{"text": content, "locator": locator}], identity)]

        key = "products" if source.source_type is SourceType.B2B else "stories"
        values = payload.get(key)
        if not isinstance(values, list):
            raise ParseFailure("INVALID_DOCUMENT_SHAPE", "json", f"expected {key} list")
        if not values:
            raise ParseFailure("INVALID_DOCUMENT_SHAPE", "json", f"{key} list must not be empty")
        if len(values) > self.max_documents:
            raise ParseFailure("DOCUMENT_COUNT_LIMIT", "json", "document limit reached")

        canonical_urls: dict[str, str] = {}
        seen_identities: set[str] = set()
        for item in values:
            if not isinstance(item, dict):
                raise ParseFailure("INVALID_DOCUMENT_SHAPE", "json", "items must be objects")
            if source.source_type is SourceType.B2B:
                self._validate_b2b(item)
                identity = str(item["product_id"])
                parser = "b2b"
                label = "product"
            else:
                self._validate_news(item)
                identity = item["id"]
                parser = "industry_news"
                label = "story"
                canonical_urls[identity] = item["url"]
            if identity in seen_identities:
                raise ParseFailure("INVALID_DOCUMENT_SHAPE", parser, f"duplicate {label} ID")
            seen_identities.add(identity)

        docs: list[DocumentRecord] = []
        for row, item in enumerate(values, 1):
            if row > self.max_documents:
                raise ParseFailure("DOCUMENT_COUNT_LIMIT", "json", "document limit reached")
            if not isinstance(item, dict):
                raise ParseFailure("INVALID_DOCUMENT_SHAPE", "json", "items must be objects")
            if source.source_type is SourceType.B2B:
                identity = str(item["product_id"])
                title = item["product_name"]
                content = self._b2b_evidence_text(item)
                item_url = item["url"]
                canonical_url = item.get("canonical_url") or item_url
                raw_values = {
                    "product_id": item["product_id"], "claim_id": item["reference_claim_id"],
                    "sku": item["sku"], "hs_code": item["hs_code"], "supplier": item["supplier"],
                }
            else:
                identity = item["id"]
                title = item["headline"]
                content = item["body"]
                item_url = item["url"]
                canonical_id = item["canonical_story_id"]
                if canonical_id not in canonical_urls:
                    raise ParseFailure("INVALID_DOCUMENT_SHAPE", "industry_news", "canonical_story_id does not resolve")
                canonical_url = canonical_urls[canonical_id]
                raw_values = {
                    "story_id": item["id"], "claim_id": item["reference_claim_id"],
                    "canonical_story_id": canonical_id, "syndicated_from": item.get("syndicated_from"),
                    "dedupe_cluster_id": item["dedupe_cluster_id"],
                }
            attrs = self._attrs(source, item, item_source_url=item_url, item_canonical_url=canonical_url)
            raw = self._raw_provenance(source, item_source_url=item_url, item_canonical_url=canonical_url, values=raw_values)
            docs.append(self._record(source, title, content, attrs, [{"text": content, "locator": {"row": row, "raw": raw}}], identity))
        return docs

    def _jsonl(self, source: SourceInput, text: str) -> list[DocumentRecord]:
        values = jsonl_values(text, self.max_documents)
        if not values:
            raise ParseFailure("INVALID_DOCUMENT_SHAPE", "social", "JSONL feed must contain at least one post")
        seen_post_ids: set[str] = set()
        for _, item in values:
            self._validate_social(item)
            post_id = item["post_id"]
            if post_id in seen_post_ids:
                raise ParseFailure("INVALID_DOCUMENT_SHAPE", "social", "duplicate post ID")
            seen_post_ids.add(post_id)

        docs: list[DocumentRecord] = []
        for row, item in values:
            item_url = item["url"]
            canonical_url = item.get("canonical_url") or item_url
            attrs = self._attrs(source, item, item_source_url=item_url, item_canonical_url=canonical_url)
            raw = self._raw_provenance(
                source,
                item_source_url=item_url,
                item_canonical_url=canonical_url,
                values={"post_id": item["post_id"], "claim_id": item["reference_claim_id"], "company": item["company"]},
            )
            locator = {"post_id": item["post_id"], "row": row, "raw": raw}
            docs.append(self._record(source, item["post_id"], item["text"], attrs, [{"text": item["text"], "locator": locator}], item["post_id"]))
        return docs

    def _sections(self, source: SourceInput, text: str, html: bool) -> list[DocumentRecord]:
        parser = parse_html_sections if html else parse_markdown_sections
        sections = parser(text, source.title, self.max_documents)
        item_url = str(source.source_url) if source.source_url else None
        canonical_url = str(source.canonical_url) if source.canonical_url else item_url
        docs: list[DocumentRecord] = []
        for index, (heading, body) in enumerate(sections):
            attrs = self._attrs(source, {}, item_source_url=item_url, item_canonical_url=canonical_url)
            raw = self._raw_provenance(source, item_source_url=item_url, item_canonical_url=canonical_url, values={"section_index": index})
            docs.append(self._record(source, heading, body, attrs, [{"text": body, "locator": {"section": heading, "raw": raw}}], f"{heading}:{index}"))
        return docs

    def _pdf(self, source: SourceInput) -> list[DocumentRecord]:
        try:
            extraction = self.mineru.extract(source.path)
            if extraction is None:
                extraction = pymupdf_extract(source.path, mineru_status=self.mineru.last_status)
        except Exception as exc:
            raise ParseFailure("MALFORMED_PDF", "pdf", "PDF extraction failed") from exc
        if not extraction.units:
            diagnostic = f"mineru_status={self.mineru.last_status}; pymupdf=no_usable_text"
            return self._quarantine("SCANNED_PDF_OCR_UNAVAILABLE", source, extraction.parser, diagnostic)

        item_url = str(source.source_url) if source.source_url else None
        canonical_url = str(source.canonical_url) if source.canonical_url else item_url
        units: list[dict[str, Any]] = []
        for unit in extraction.units:
            copied = {**unit, "locator": dict(unit["locator"])}
            locator = copied["locator"]
            existing_raw = locator.get("raw")
            raw_values = dict(existing_raw) if isinstance(existing_raw, dict) else ({"parser_raw": existing_raw} if existing_raw is not None else {})
            locator["raw"] = self._raw_provenance(source, item_source_url=item_url, item_canonical_url=canonical_url, values=raw_values)
            units.append(copied)
        content = "\n\n".join(unit["text"] for unit in units)
        attrs = self._attrs(
            source,
            {
                "parser": extraction.parser,
                "parser_mode": extraction.mode,
                "mineru_status": self.mineru.last_status,
                "degraded_components": list(extraction.degraded_components),
            },
            item_source_url=item_url,
            item_canonical_url=canonical_url,
        )
        return [self._record(source, source.title, content, attrs, units, source.path.name)]


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
            path=corpus_root / record["path"],
            file_type=FileType(record["file_type"]),
            source_type=source_type,
            source_id=stable_id("source", record["path"]),
            title=record.get("expected_entity", record["path"]),
            language="en",
            fetched_at=when,
            is_synthetic=record["is_synthetic"],
            source_url="https://ingest.example/manifest",
            publish_time=datetime.fromisoformat(record["publish_time"]),
            valid_from=datetime.fromisoformat(record["valid_from"]),
            manifest_attributes={key: value for key, value in record.items() if key not in {"path", "file_type", "source_type", "is_synthetic"}},
        )
        documents += len(router.load(source))
    return {
        "documents": documents,
        "quarantines": [
            {"error_code": item.error_code, "path": str(Path(item.source_path).relative_to(corpus_root))}
            for item in router.quarantines
        ],
    }
