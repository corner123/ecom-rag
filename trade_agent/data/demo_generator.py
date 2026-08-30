"""Build a small, deterministic, explicitly synthetic trade-intelligence corpus."""
from __future__ import annotations

import hashlib
import io
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from trade_agent.db.seed import TradeSeedBundle, generate_trade_seed

_STAMP = "2026-08-30T00:00:00+00:00"
_MARKER = ".trade-intel-demo-owned"
_BANNER = "SYNTHETIC DEMONSTRATION ONLY — FICTIONAL DATA; NOT FOR PRODUCTION USE."


@dataclass(frozen=True)
class CorpusSummary:
    source_types: set[str]
    file_types: set[str]
    website_sections: int
    b2b_products: int
    news_stories: int
    social_posts: int
    text_pdf_count: int
    scanned_pdf_count: int
    customs_profiles: int
    manifest_path: Path


def _json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _safe_root(output: Path) -> Path:
    root = output.resolve()
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    forbidden = {Path(root.anchor), cwd, home}
    if root in forbidden or root.parent == root or root == cwd.parent:
        raise ValueError("refusing broad output root")
    return root


def ensure_safe_output(output: Path, *, clean: bool = False) -> Path:
    """Prepare an owned output directory without ever deleting an arbitrary tree."""
    root = _safe_root(output)
    if root.exists() and not root.is_dir():
        raise ValueError("output must be a directory")
    if root.exists() and any(root.iterdir()) and not (root / _MARKER).is_file():
        raise ValueError("refusing to write into a nonempty directory not owned by this demo")
    if clean and root.exists():
        marker = root / _MARKER
        if not marker.is_file() or marker.read_text(encoding="utf-8") != "synthetic-trade-intel-demo-v1\n":
            raise ValueError("refusing to clean a nonempty directory not owned by this demo")
        for child in sorted(root.iterdir()):
            if child.name in {_MARKER, "README.md"}:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / _MARKER
    if marker.exists() and marker.read_text(encoding="utf-8") != "synthetic-trade-intel-demo-v1\n":
        raise ValueError("output ownership marker is invalid")
    marker.write_text("synthetic-trade-intel-demo-v1\n", encoding="utf-8")
    return root


def _write(root: Path, relative: str, payload: bytes, *, source_type: str, file_type: str,
           expected_entity: str, fact_type: str, locator: dict[str, Any], claim_id: str,
           records: list[dict[str, Any]]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    records.append({
        "path": relative, "content_hash": hashlib.sha256(payload).hexdigest(),
        "source_type": source_type, "file_type": file_type, "expected_entity": expected_entity,
        "fact_type": fact_type, "publish_time": _STAMP, "valid_from": _STAMP,
        "ingested_at": _STAMP, "reference_claim_ids": [claim_id], "locator": locator,
        "is_synthetic": True,
    })


def _text_pdf(title: str, lines: list[str]) -> bytes:
    buffer = io.BytesIO()
    pdf = Canvas(buffer, pagesize=(612, 792), invariant=1, pageCompression=1)
    pdf.setTitle(title); pdf.setAuthor("Synthetic Trade Intelligence Demo")
    pdf.setFont("Helvetica-Bold", 14); pdf.drawString(54, 740, title)
    pdf.setFont("Helvetica", 10)
    for index, line in enumerate([_BANNER, *lines]):
        pdf.drawString(54, 710 - index * 24, line)
    pdf.showPage(); pdf.save()
    return buffer.getvalue()


def _scanned_pdf() -> bytes:
    # A tiny embedded raster image makes this a genuinely image-only PDF.
    png = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
           b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb1\x00\x00\x00\x00IEND\xaeB`\x82")
    buffer = io.BytesIO()
    pdf = Canvas(buffer, pagesize=(612, 792), invariant=1, pageCompression=1)
    pdf.setTitle("Synthetic scanned regulator notice")
    pdf.drawImage(ImageReader(io.BytesIO(png)), 36, 36, width=540, height=720, mask="auto")
    pdf.showPage(); pdf.save()
    return buffer.getvalue()


def _website(root: Path, bundle: TradeSeedBundle, records: list[dict[str, Any]]) -> int:
    for index in range(12):
        company = bundle.companies[index]
        name = company.company_name
        claim = f"CLAIM-WEB-{index + 1:02d}"
        text = f"{_BANNER}\n{name} is a fictional supplier profile for HS demonstration data."
        html = f"<!doctype html><html><body><aside>{_BANNER}</aside><h1>{name}</h1><section id='overview'>{text}</section></body></html>"
        _write(root, f"website/section-{index + 1:02d}.html", html.encode(), source_type="official_website", file_type="html", expected_entity=name, fact_type="company_status", locator={"section": "overview", "url": company.website}, claim_id=claim, records=records)
    markdown = f"# Synthetic website methodology\n\n{_BANNER}\n\nAll company pages use fictional `.example` domains.\n"
    _write(root, "website/methodology.md", markdown.encode(), source_type="official_website", file_type="markdown", expected_entity="Synthetic Trade Intelligence Demo", fact_type="company_status", locator={"section": "methodology"}, claim_id="CLAIM-WEB-METHOD", records=records)
    return 12


def _b2b(root: Path, bundle: TradeSeedBundle, records: list[dict[str, Any]]) -> int:
    products = []
    for index in range(18):
        product = bundle.products[index]
        company = bundle.companies[index]
        products.append({"synthetic_notice": _BANNER, "product_id": product.id, "sku": product.sku, "product_name": product.product_name, "hs_code": bundle.hs_codes[product.hs_code_id - 1].hs_code, "supplier": company.company_name, "url": f"https://marketplace.example/products/{product.sku.lower()}"})
    payload = _json({"synthetic_notice": _BANNER, "products": products})
    _write(root, "b2b/products.json", payload, source_type="b2b", file_type="json", expected_entity="Synthetic B2B Marketplace", fact_type="product_offering", locator={"table": "products", "rows": 18}, claim_id="CLAIM-B2B-CATALOG", records=records)
    return len(products)


def _news(root: Path, bundle: TradeSeedBundle, records: list[dict[str, Any]]) -> int:
    stories = []
    for index in range(16):
        company = bundle.companies[index]
        syndicated_from = None if index < 12 else f"NEWS-{index - 11:03d}"
        stories.append({"id": f"NEWS-{index + 1:03d}", "synthetic_notice": _BANNER, "headline": f"Fictional trade signal for {company.company_name}", "body": f"{_BANNER} This fictional report describes a demo-only market signal.", "url": f"https://newsroom.example/story/{index + 1:03d}", "syndicated_from": syndicated_from, "published_at": _STAMP})
    _write(root, "news/stories.json", _json({"synthetic_notice": _BANNER, "stories": stories}), source_type="industry_news", file_type="json", expected_entity="Synthetic Industry Newswire", fact_type="market_signal", locator={"table": "stories", "rows": 16, "syndication_control": "NEWS-013..016 mirror NEWS-001..004"}, claim_id="CLAIM-NEWS-SET", records=records)
    html = f"<html><body><p>{_BANNER}</p><article><h1>{stories[0]['headline']}</h1><p>{stories[0]['body']}</p></article></body></html>"
    _write(root, "news/representative-story.html", html.encode(), source_type="industry_news", file_type="html", expected_entity=bundle.companies[0].company_name, fact_type="market_signal", locator={"section": "article", "story_id": "NEWS-001"}, claim_id="CLAIM-NEWS-001", records=records)
    return len(stories)


def _social(root: Path, bundle: TradeSeedBundle, records: list[dict[str, Any]]) -> int:
    posts = [{"post_id": f"POST-{index + 1:03d}", "synthetic_notice": _BANNER, "company": bundle.companies[index].company_name, "text": f"{_BANNER} Fictional update about a sample shipment.", "url": f"https://social.example/posts/{index + 1:03d}", "published_at": _STAMP} for index in range(12)]
    _write(root, "social/posts.jsonl", b"".join(_json(post) for post in posts), source_type="social", file_type="jsonl", expected_entity="Synthetic Social Feed", fact_type="market_signal", locator={"post_id": "POST-001..POST-012", "rows": 12}, claim_id="CLAIM-SOCIAL-SET", records=records)
    return len(posts)


def _pdfs(root: Path, records: list[dict[str, Any]]) -> tuple[int, int]:
    _write(root, "pdf/regulator-notice.pdf", _text_pdf("Synthetic regulator notice", ["A demo-only compliance date is 2026-09-01."]), source_type="regulator", file_type="pdf", expected_entity="Synthetic Trade Standards Office", fact_type="regulation", locator={"page": 1, "kind": "text_pdf"}, claim_id="CLAIM-REG-001", records=records)
    _write(root, "pdf/market-bulletin.pdf", _text_pdf("Synthetic market bulletin", ["This report contains fictional aggregate indicators only."]), source_type="industry_news", file_type="pdf", expected_entity="Synthetic Market Bulletin", fact_type="market_signal", locator={"page": 1, "kind": "text_pdf"}, claim_id="CLAIM-PDF-001", records=records)
    _write(root, "pdf/scanned-regulator-notice.pdf", _scanned_pdf(), source_type="regulator", file_type="pdf", expected_entity="Synthetic Trade Standards Office", fact_type="regulation", locator={"page": 1, "kind": "image_only_scanned_pdf"}, claim_id="CLAIM-REG-SCAN-001", records=records)
    return 2, 1


def _profiles(root: Path, bundle: TradeSeedBundle, records: list[dict[str, Any]]) -> int:
    chosen = [record for record in bundle.trade_records if record.importer_id in {1, 2, 3}][:12]
    for index, record in enumerate(chosen, 1):
        company = bundle.companies[record.importer_id - 1]
        hs = bundle.hs_codes[record.hs_code_id - 1]
        profile = {"synthetic_notice": _BANNER, "company": company.company_name, "company_id": company.id, "hs_code": hs.hs_code, "month": record.trade_date.isoformat(), "trade_amount_usd": str(record.trade_amount), "aggregation": "one deterministic monthly company/HS tuple from Task 3 seed aggregates", "source_record_count": 1}
        _write(root, f"customs_profiles/{company.id:03d}-{hs.hs_code}-{record.trade_date:%Y-%m}.json", _json(profile), source_type="customs_profile", file_type="generated_profile", expected_entity=company.company_name, fact_type="trade_activity", locator={"profile": "monthly_company_hs", "company_id": company.id, "hs_code": hs.hs_code, "month": record.trade_date.isoformat(), "aggregate_rows": 1}, claim_id=f"CLAIM-CUSTOMS-{index:02d}", records=records)
    return len(chosen)


def generate_demo_corpus(output: Path, seed: int = 20260830, clean: bool = False) -> CorpusSummary:
    root = ensure_safe_output(Path(output), clean=clean)
    records: list[dict[str, Any]] = []
    bundle = generate_trade_seed(seed)
    website_sections = _website(root, bundle, records)
    b2b_products = _b2b(root, bundle, records)
    news_stories = _news(root, bundle, records)
    social_posts = _social(root, bundle, records)
    text_pdfs, scanned_pdfs = _pdfs(root, records)
    profiles = _profiles(root, bundle, records)
    records.sort(key=lambda record: record["path"])
    manifest = {"synthetic_notice": _BANNER, "seed": seed, "generated_at": _STAMP, "records": records, "counts": dict(sorted(Counter(record["source_type"] for record in records).items()))}
    manifest_path = root / "manifests" / "corpus_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(_json(manifest))
    return CorpusSummary(set(manifest["counts"]), {record["file_type"] for record in records}, website_sections, b2b_products, news_stories, social_posts, text_pdfs, scanned_pdfs, profiles, manifest_path)
