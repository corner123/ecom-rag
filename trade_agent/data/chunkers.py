"""Source-aware deterministic chunking."""
from __future__ import annotations
import re
from datetime import datetime, timezone
from typing import Callable
from trade_agent.schemas.source import ChunkMetadata, ChunkRecord, DocumentRecord, SourceLocator, SourceType, content_sha256, stable_id

def _tokens(text): return re.findall(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*|[^\s]", text)
def _count(text): return len(_tokens(text))
class BaseChunker:
    def __init__(self,max_tokens=500,overlap_tokens=64,token_counter:Callable[[str],int]|None=None):
        if max_tokens <= 0 or not 0 <= overlap_tokens < max_tokens: raise ValueError("require 0 <= overlap < max")
        self.max_tokens,self.overlap_tokens,self.counter=max_tokens,overlap_tokens,token_counter or _count
    def split_text(self,text):
        words=_tokens(text)
        if self.counter(text)<=self.max_tokens:return [text]
        output=[]; start=0
        while start<len(words):
            end=min(len(words),start+self.max_tokens); output.append(" ".join(words[start:end]));
            if end==len(words):break
            start=end-self.overlap_tokens
        return output
class WebsiteSectionChunker(BaseChunker):
    def split(self,d):
        heading=(d.units[0].get("locator",{}).get("section",d.title) if d.units else d.title)
        return [(f"{heading}\n{x}",{"section": heading}) for x in self.split_text(d.content)]
class B2BProductChunker(BaseChunker):
    def split(self,d): return [(x,{"row": d.units[0].get("locator",{}).get("row",1) if d.units else 1}) for x in self.split_text(d.content)]
class NewsParagraphChunker(BaseChunker):
    def split(self,d): return [(f"{d.title}\n{x}",{"row":(d.units[0].get("locator",{}).get("row",1) if d.units else 1)}) for x in self.split_text(d.content)]
class SocialPostChunker(BaseChunker):
    def split(self,d):
        base=d.units[0].get("locator",{}) if d.units else d.attributes;return [(x,{"post_id":base.get("post_id"),"row":base.get("row")}) for x in self.split_text(d.content)]
class PdfLayoutChunker(BaseChunker):
    def split(self,d):
        out=[]
        for unit in d.units:
            for text in self.split_text(unit["text"]): out.append((text,unit["locator"]))
        return out
class CustomsProfileChunker(BaseChunker):
    def split(self,d): return [(x,{"profile":"monthly_company_hs"}) for x in self.split_text(d.content)]
class ChunkRouter:
    def __init__(self,max_tokens=500,overlap_tokens=64,token_counter=None):
        args=(max_tokens,overlap_tokens,token_counter);self.by={SourceType.OFFICIAL_WEBSITE:WebsiteSectionChunker(*args),SourceType.B2B:B2BProductChunker(*args),SourceType.INDUSTRY_NEWS:NewsParagraphChunker(*args),SourceType.SOCIAL:SocialPostChunker(*args),SourceType.REGULATOR:PdfLayoutChunker(*args),SourceType.CUSTOMS_PROFILE:CustomsProfileChunker(*args)}
    def chunk(self,d:DocumentRecord):
        chunks=[]
        chunker = PdfLayoutChunker(self.by[d.source_type].max_tokens, self.by[d.source_type].overlap_tokens, self.by[d.source_type].counter) if d.file_type.value == "pdf" else self.by[d.source_type]
        for i,(content,loc) in enumerate(chunker.split(d)):
            attrs=d.attributes; locator=SourceLocator.model_validate(loc)
            meta=ChunkMetadata(chunk_id=stable_id("chunk",d.document_id,str(i),content),document_id=d.document_id,entity_id=attrs.get("entity_id"),company_name=attrs.get("company") or attrs.get("supplier") or attrs.get("entity"),normalized_name=attrs.get("normalized_name"),country_code=attrs.get("country_code"),region=attrs.get("region"),hs_code=attrs.get("hs_code"),product_name=attrs.get("product_name"),sku=attrs.get("sku"),file_type=d.file_type,source_type=d.source_type,source_weight=attrs.get("source_weight",0.5),fact_type=attrs.get("fact_type"),publish_time=attrs.get("published_at") or attrs.get("publish_time"),valid_from=attrs.get("valid_from"),valid_to=attrs.get("valid_to"),ingested_at=d.fetched_at,source_url=d.source_url,canonical_url=d.canonical_url or attrs.get("url"),source_locator=locator,raw_record_id=attrs.get("raw_record_id"),aggregation_info=attrs.get("aggregation_info"),content_hash=content_sha256(content),parent_document_hash=d.content_hash,language=d.language,ocr_confidence=attrs.get("ocr_confidence"),is_synthetic=d.is_synthetic,license_scope=attrs.get("license_scope"),dedupe_cluster_id=attrs.get("dedupe_cluster_id"))
            chunks.append(ChunkRecord(content=content,metadata=meta))
        return chunks
