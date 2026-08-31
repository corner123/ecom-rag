"""Source-aware chunking with exact budgets and lossless locators."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite
from typing import Callable

from trade_agent.schemas.source import (
    ChunkMetadata,
    ChunkRecord,
    DocumentRecord,
    FileType,
    SourceLocator,
    SourceType,
    content_sha256,
    stable_id,
)


def fallback_tokens(text: str) -> int:
    """Count stable lexical tokens while keeping trade identifiers atomic."""
    return len(re.findall(r"[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)*|[^\s]", text))


_PROTECTED = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"https?://[A-Za-z0-9._/?=&%+-]+"
    r"|[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)+"
    r"|(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]+"
    r"|\d{4,10}"
    r")(?![A-Za-z0-9])"
)
_PARAGRAPH_END = re.compile(r"\n[ \t]*\n")
_SENTENCE_END = re.compile(r"[.!?。！？；](?:[\"')\]”’》】）]*)[ \t]*(?:\n|(?=\s)|$)?")
_SAFE_END = re.compile(r"\s+|[,;:，、；：]\s*")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


@dataclass(frozen=True, slots=True)
class SplitPiece:
    content: str
    locator: dict
    confidence: float | int | None = None


class BaseChunker:
    def __init__(
        self,
        max_tokens: int = 500,
        overlap_tokens: int = 64,
        token_counter: Callable[[str], int] | None = None,
    ):
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if type(overlap_tokens) is not int or not 0 <= overlap_tokens < max_tokens:
            raise ValueError("require integer 0 <= overlap < max")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.counter = token_counter or fallback_tokens
        self._count("")

    def _count(self, text: str) -> int:
        value = self.counter(text)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("token counter must return a nonnegative integer")
        return value

    @staticmethod
    def _protected_spans(text: str) -> list[tuple[int, int]]:
        return [match.span() for match in _PROTECTED.finditer(text)]

    @staticmethod
    def _safe(position: int, spans: list[tuple[int, int]]) -> bool:
        return not any(start < position < end for start, end in spans)

    def _fit_end(self, text: str, start: int, prefix: str, spans: list[tuple[int, int]]) -> int:
        categories: tuple[list[int], list[int], list[int]] = (
            [match.end() for match in _PARAGRAPH_END.finditer(text, start)],
            [match.end() for match in _SENTENCE_END.finditer(text, start)],
            [match.end() for match in _SAFE_END.finditer(text, start)],
        )
        for positions in categories:
            for end in reversed(positions):
                if end <= start or not self._safe(end, spans):
                    continue
                if self._count(prefix + text[start:end]) <= self.max_tokens:
                    return end
        if self._count(prefix + text[start:]) <= self.max_tokens:
            return len(text)
        codepoint_end = self._codepoint_end(text, start, prefix, spans)
        if codepoint_end is not None:
            return codepoint_end
        next_boundaries = sorted({end for positions in categories for end in positions if end > start and self._safe(end, spans)})
        if not next_boundaries:
            raise ValueError("atomic token cannot fit within chunk budget")
        first = next_boundaries[0]
        if self._count(prefix + text[start:first]) > self.max_tokens:
            protected = next((text[left:right] for left, right in spans if left <= start < right or start <= left < first), None)
            if protected:
                raise ValueError(f"atomic identifier {protected!r} cannot fit within chunk budget")
            raise ValueError("atomic token cannot fit within chunk budget")
        raise ValueError("token counter cannot make progress at a safe boundary")

    def _codepoint_end(
        self,
        text: str,
        start: int,
        prefix: str,
        spans: list[tuple[int, int]],
    ) -> int | None:
        """Find a bounded codepoint cut for CJK prose when no lexical boundary fits."""
        if _CJK.search(text, start) is None:
            return None
        low, high = start + 1, len(text)
        best: int | None = None
        while low <= high:
            middle = (low + high) // 2
            if self._count(prefix + text[start:middle]) <= self.max_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best is None:
            return None
        for left, right in spans:
            if left < best < right:
                if self._count(prefix + text[start:right]) <= self.max_tokens:
                    best = right
                else:
                    best = left
                break
        return best if best > start and self._safe(best, spans) else None

    def _overlap_start(self, text: str, start: int, end: int, spans: list[tuple[int, int]]) -> int:
        if self.overlap_tokens == 0:
            return end
        candidates = {match.end() for match in _SAFE_END.finditer(text, start, end)}
        candidates.update(match.end() for match in _SENTENCE_END.finditer(text, start, end))
        candidates.update(match.end() for match in _PARAGRAPH_END.finditer(text, start, end))
        for candidate in sorted(candidates):
            if candidate <= start or candidate >= end or not self._safe(candidate, spans):
                continue
            if self._count(text[candidate:end]) <= self.overlap_tokens:
                return candidate
        if _CJK.search(text, start, end) is not None:
            low, high = start + 1, end - 1
            best: int | None = None
            while low <= high:
                middle = (low + high) // 2
                if self._count(text[middle:end]) <= self.overlap_tokens:
                    best = middle
                    high = middle - 1
                else:
                    low = middle + 1
            if best is not None:
                for left, right in spans:
                    if left < best < right:
                        best = right
                        break
                if start < best < end and self._safe(best, spans) and self._count(text[best:end]) <= self.overlap_tokens:
                    return best
        return end

    def split_text(self, text: str, prefix: str = "") -> list[str]:
        if not text:
            raise ValueError("chunk text must not be empty")
        if self._count(prefix + text) <= self.max_tokens:
            return [prefix + text]

        spans = self._protected_spans(text)
        for left, right in spans:
            if self._count(prefix + text[left:right]) > self.max_tokens:
                raise ValueError(f"atomic identifier {text[left:right]!r} cannot fit within chunk budget")

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = self._fit_end(text, start, prefix, spans)
            if end <= start:
                raise ValueError("token counter cannot make progress")
            content = prefix + text[start:end]
            if self._count(content) > self.max_tokens:
                raise ValueError("chunk exceeds configured budget")
            chunks.append(content)
            if end == len(text):
                break
            next_start = self._overlap_start(text, start, end, spans)
            start = next_start if next_start > start else end
        return chunks

    @staticmethod
    def _first_locator(document: DocumentRecord, fallback: dict) -> tuple[dict, float | int | None]:
        if not document.units:
            return fallback, None
        unit = document.units[0]
        locator = unit.get("locator")
        return (dict(locator) if isinstance(locator, dict) else fallback), unit.get("confidence")


class WebsiteSectionChunker(BaseChunker):
    def split(self, document: DocumentRecord) -> list[SplitPiece]:
        locator, confidence = self._first_locator(document, {"section": document.title})
        heading = locator.get("section") or document.title
        return [SplitPiece(content, locator, confidence) for content in self.split_text(document.content, f"{heading}\n")]


class B2BProductChunker(BaseChunker):
    def split(self, document: DocumentRecord) -> list[SplitPiece]:
        locator, confidence = self._first_locator(document, {"row": 1})
        prefix = f"{document.title}\n" if document.file_type is FileType.HTML else ""
        return [SplitPiece(content, locator, confidence) for content in self.split_text(document.content, prefix)]


class NewsParagraphChunker(BaseChunker):
    def split(self, document: DocumentRecord) -> list[SplitPiece]:
        locator, confidence = self._first_locator(document, {"row": 1})
        return [SplitPiece(content, locator, confidence) for content in self.split_text(document.content, f"{document.title}\n")]


class SocialPostChunker(BaseChunker):
    def split(self, document: DocumentRecord) -> list[SplitPiece]:
        fallback = {key: document.attributes[key] for key in ("post_id", "row") if key in document.attributes}
        locator, confidence = self._first_locator(document, fallback or {"post_id": "unknown"})
        return [SplitPiece(content, locator, confidence) for content in self.split_text(document.content)]


class PdfLayoutChunker(BaseChunker):
    def split(self, document: DocumentRecord) -> list[SplitPiece]:
        pieces: list[SplitPiece] = []
        prefix = f"{document.title}\n" if document.source_type is SourceType.INDUSTRY_NEWS else ""
        for unit in document.units:
            locator = unit.get("locator")
            if not isinstance(locator, dict):
                raise ValueError("PDF unit requires a structured locator")
            confidence = unit.get("confidence")
            pieces.extend(SplitPiece(content, dict(locator), confidence) for content in self.split_text(unit["text"], prefix))
        return pieces


class CustomsProfileChunker(BaseChunker):
    def split(self, document: DocumentRecord) -> list[SplitPiece]:
        locator, confidence = self._first_locator(document, {"profile": "monthly_company_hs"})
        return [SplitPiece(content, locator, confidence) for content in self.split_text(document.content)]


class ChunkRouter:
    def __init__(self, max_tokens: int = 500, overlap_tokens: int = 64, token_counter: Callable[[str], int] | None = None):
        args = (max_tokens, overlap_tokens, token_counter)
        self.by = {
            SourceType.OFFICIAL_WEBSITE: WebsiteSectionChunker(*args),
            SourceType.B2B: B2BProductChunker(*args),
            SourceType.INDUSTRY_NEWS: NewsParagraphChunker(*args),
            SourceType.SOCIAL: SocialPostChunker(*args),
            SourceType.REGULATOR: PdfLayoutChunker(*args),
            SourceType.CUSTOMS_PROFILE: CustomsProfileChunker(*args),
        }

    def chunk(self, document: DocumentRecord) -> list[ChunkRecord]:
        configured = self.by[document.source_type]
        selected = (
            PdfLayoutChunker(configured.max_tokens, configured.overlap_tokens, configured.counter)
            if document.file_type.value == "pdf"
            else configured
        )
        attrs = document.attributes
        chunks: list[ChunkRecord] = []
        for index, piece in enumerate(selected.split(document)):
            locator = SourceLocator.model_validate(piece.locator)
            content = piece.content
            unit_confidence = None
            normalized_locator = locator.model_dump(mode="json", exclude_none=True)
            for unit in document.units:
                candidate = unit.get("locator")
                if not isinstance(candidate, dict):
                    continue
                try:
                    candidate_locator = SourceLocator.model_validate(candidate)
                except Exception:
                    continue
                if candidate_locator.model_dump(mode="json", exclude_none=True) == normalized_locator:
                    if "confidence" in unit:
                        unit_confidence = unit["confidence"]
                    break
            if unit_confidence is not None:
                if isinstance(unit_confidence, bool) or not isinstance(unit_confidence, (int, float)) or not isfinite(unit_confidence):
                    raise ValueError("document unit confidence must be a finite number")
                unit_confidence = float(unit_confidence)
            entity_id = attrs.get("entity_id")
            if entity_id is not None and not isinstance(entity_id, str):
                entity_id = str(entity_id)
            metadata = ChunkMetadata(
                chunk_id=stable_id("chunk", document.document_id, str(index), content),
                chunk_index=index,
                document_id=document.document_id,
                entity_id=entity_id,
                company_name=attrs.get("company") or attrs.get("supplier") or attrs.get("entity"),
                normalized_name=attrs.get("normalized_name"),
                country_code=attrs.get("country_code"),
                region=attrs.get("region"),
                hs_code=attrs.get("hs_code"),
                product_name=attrs.get("product_name"),
                sku=attrs.get("sku"),
                file_type=document.file_type,
                source_type=document.source_type,
                source_weight=attrs.get("source_weight", 0.5),
                fact_type=attrs.get("fact_type"),
                publish_time=attrs.get("publish_time"),
                valid_from=attrs.get("valid_from"),
                valid_to=attrs.get("valid_to"),
                ingested_at=document.fetched_at,
                source_url=document.source_url,
                canonical_url=document.canonical_url,
                source_locator=locator,
                raw_record_id=attrs.get("raw_record_id"),
                aggregation_info=attrs.get("aggregation_info"),
                content_hash=content_sha256(content),
                parent_document_hash=document.content_hash,
                language=document.language,
                ocr_confidence=unit_confidence if unit_confidence is not None else attrs.get("ocr_confidence"),
                is_synthetic=document.is_synthetic,
                license_scope=attrs.get("license_scope"),
                dedupe_cluster_id=attrs.get("dedupe_cluster_id"),
            )
            chunks.append(ChunkRecord(content=content, metadata=metadata))
        return chunks
