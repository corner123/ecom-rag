"""Source-aware chunks with one counter governing every size decision."""
from __future__ import annotations
import re
from typing import Callable
from trade_agent.schemas.source import ChunkMetadata, ChunkRecord, DocumentRecord, SourceLocator, SourceType, content_sha256, stable_id

def fallback_tokens(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*|[^\s]", text))

class BaseChunker:
    def __init__(self, max_tokens=500, overlap_tokens=64, token_counter: Callable[[str], int] | None=None):
        if max_tokens <= 0 or not 0 <= overlap_tokens < max_tokens: raise ValueError("require 0 <= overlap < max")
        self.max_tokens, self.overlap_tokens, self.counter = max_tokens, overlap_tokens, token_counter or fallback_tokens
    def _end(self, text: str, start: int, budget: int) -> int:
        lo, hi = start, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.counter(text[start:mid]) <= budget: lo = mid
            else: hi = mid - 1
        bounded = lo
        for match in re.finditer(r"(?:[.!?](?:\s|$)|\n\s*\n|\s+)", text[start:bounded]): bounded = start + match.end()
        return bounded if bounded > start else lo
    def _overlap_start(self, text: str, start: int, end: int) -> int:
        if not self.overlap_tokens: return end
        lo, hi = start, end
        while lo < hi:
            mid = (lo + hi) // 2
            if self.counter(text[mid:end]) <= self.overlap_tokens: hi = mid
            else: lo = mid + 1
        return lo
    def split_text(self, text: str, prefix: str="") -> list[str]:
        cost = self.counter(prefix)
        if cost >= self.max_tokens: raise ValueError("prefix consumes chunk budget")
        if self.counter(prefix + text) <= self.max_tokens: return [prefix + text]
        budget, out, start = self.max_tokens - cost, [], 0
        while start < len(text):
            end = self._end(text, start, budget)
            if end <= start: raise ValueError("token counter cannot make progress")
            out.append(prefix + text[start:end])
            if end == len(text): break
            next_start = self._overlap_start(text, start, end)
            start = next_start if next_start > start else end
        return out

class WebsiteSectionChunker(BaseChunker):
    def split(self,d):
        heading=(d.units[0].get("locator",{}).get("section",d.title) if d.units else d.title)
        return [(x,{"section":heading}) for x in self.split_text(d.content, f"{heading}\n")]
class B2BProductChunker(BaseChunker):
    def split(self,d): return [(x,{"row":d.units[0].get("locator",{}).get("row",1) if d.units else 1}) for x in self.split_text(d.content)]
class NewsParagraphChunker(BaseChunker):
    def split(self,d): return [(x,{"row":d.units[0].get("locator",{}).get("row",1) if d.units else 1}) for x in self.split_text(d.content,f"{d.title}\n")]
class SocialPostChunker(BaseChunker):
    def split(self,d):
        loc=d.units[0].get("locator",{}) if d.units else d.attributes
        return [(x,{"post_id":loc.get("post_id"),"row":loc.get("row")}) for x in self.split_text(d.content)]
class PdfLayoutChunker(BaseChunker):
    def split(self,d): return [(x,u["locator"]) for u in d.units for x in self.split_text(u["text"])]
class CustomsProfileChunker(BaseChunker):
    def split(self,d): return [(x,{"profile":"monthly_company_hs"}) for x in self.split_text(d.content)]

class ChunkRouter:
    def __init__(self,max_tokens=500,overlap_tokens=64,token_counter=None):
        args=(max_tokens,overlap_tokens,token_counter); self.by={SourceType.OFFICIAL_WEBSITE:WebsiteSectionChunker(*args),SourceType.B2B:B2BProductChunker(*args),SourceType.INDUSTRY_NEWS:NewsParagraphChunker(*args),SourceType.SOCIAL:SocialPostChunker(*args),SourceType.REGULATOR:PdfLayoutChunker(*args),SourceType.CUSTOMS_PROFILE:CustomsProfileChunker(*args)}
    def chunk(self,d:DocumentRecord):
        selected=PdfLayoutChunker(self.by[d.source_type].max_tokens,self.by[d.source_type].overlap_tokens,self.by[d.source_type].counter) if d.file_type.value=="pdf" else self.by[d.source_type]
        chunks=[]
        for i,(content,loc) in enumerate(selected.split(d)):
            attrs=d.attributes; unit=next((u for u in d.units if u.get("locator")==loc),{}); locator=SourceLocator.model_validate(loc)
            meta=ChunkMetadata(chunk_id=stable_id("chunk",d.document_id,str(i),content),document_id=d.document_id,entity_id=attrs.get("entity_id"),company_name=attrs.get("company") or attrs.get("supplier") or attrs.get("entity"),normalized_name=attrs.get("normalized_name"),country_code=attrs.get("country_code"),region=attrs.get("region"),hs_code=attrs.get("hs_code"),product_name=attrs.get("product_name"),sku=attrs.get("sku"),file_type=d.file_type,source_type=d.source_type,source_weight=attrs.get("source_weight",.5),fact_type=attrs.get("fact_type"),publish_time=attrs.get("published_at") or attrs.get("publish_time"),valid_from=attrs.get("valid_from"),valid_to=attrs.get("valid_to"),ingested_at=d.fetched_at,source_url=d.source_url,canonical_url=d.canonical_url or attrs.get("url"),source_locator=locator,raw_record_id=attrs.get("raw_record_id"),aggregation_info=attrs.get("aggregation_info"),content_hash=content_sha256(content),parent_document_hash=d.content_hash,language=d.language,ocr_confidence=unit.get("confidence",attrs.get("ocr_confidence")),is_synthetic=d.is_synthetic,license_scope=attrs.get("license_scope"),dedupe_cluster_id=attrs.get("dedupe_cluster_id"))
            chunks.append(ChunkRecord(content=content,metadata=meta))
        return chunks
