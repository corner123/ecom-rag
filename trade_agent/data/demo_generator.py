"""Build a small, deterministic, explicitly synthetic trade-intelligence corpus."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from trade_agent.db.seed import TradeSeedBundle, generate_trade_seed

_STAMP = "2026-08-30T00:00:00+00:00"
_MARKER = ".trade-intel-demo-owned"
_BANNER = "SYNTHETIC DEMONSTRATION ONLY — FICTIONAL DATA; NOT FOR PRODUCTION USE."
_SCAN_WARNING = "SYNTHETIC DEMONSTRATION ONLY - FICTIONAL DATA - NOT FOR PRODUCTION USE."


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
    root = output.absolute()
    cwd = Path.cwd().absolute()
    home = Path.home().absolute()
    forbidden = {Path(root.anchor), cwd, home}
    if root in forbidden or root.parent == root or root == cwd.parent:
        raise ValueError("refusing broad output root")
    return root


def _trusted_baseline(root: Path) -> Path:
    """Find a lexical safe starting point without resolving a user output path."""
    candidates = {Path.cwd().absolute(), Path.home().absolute(), Path(tempfile.gettempdir()).absolute()}
    candidates.update(candidate.resolve() for candidate in tuple(candidates))
    # macOS commonly exposes /var as an alias to /private/var; it is a system baseline,
    # not an untrusted portion of a tempfile output path.
    for system_alias in (Path("/var"), Path("/tmp")):
        if system_alias.is_symlink():
            candidates.add(system_alias)
    matching = [candidate for candidate in candidates if root.is_relative_to(candidate)]
    return max(matching, key=lambda candidate: len(candidate.parts)) if matching else Path(root.anchor)


def _assert_safe_output_ancestors(root: Path) -> None:
    """Reject symlink components from a trusted lexical baseline through ``root``."""
    baseline = _trusted_baseline(root)
    relative = root.relative_to(baseline)
    current = baseline
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("refusing symlink output ancestor")


def _assert_no_symlinks(root: Path) -> None:
    """Reject symlinks in the owned output tree without resolving ordinary parents."""
    if root.is_symlink():
        raise ValueError("refusing symlink output root")
    if not root.exists():
        return
    for directory, directories, filenames in os.walk(root, followlinks=False):
        for name in [*directories, *filenames]:
            if (Path(directory) / name).is_symlink():
                raise ValueError("refusing symlink inside output tree")


def _owned_marker(root: Path) -> Path:
    marker = root / _MARKER
    if marker.is_symlink():
        raise ValueError("refusing symlink ownership marker")
    return marker


def ensure_safe_output(output: Path, *, clean: bool = False) -> Path:
    """Prepare an owned output directory without ever deleting an arbitrary tree."""
    root = _safe_root(output)
    _assert_safe_output_ancestors(root)
    _assert_no_symlinks(root)
    if root.exists() and not root.is_dir():
        raise ValueError("output must be a directory")
    if root.exists() and any(root.iterdir()) and not _owned_marker(root).is_file():
        raise ValueError("refusing to write into a nonempty directory not owned by this demo")
    if clean and root.exists():
        marker = _owned_marker(root)
        if not marker.is_file() or marker.read_text(encoding="utf-8") != "synthetic-trade-intel-demo-v1\n":
            raise ValueError("refusing to clean a nonempty directory not owned by this demo")
        for child in sorted(root.iterdir()):
            if child.name in {_MARKER, "README.md"}:
                continue
            if child.is_symlink():
                raise ValueError("refusing symlink inside output tree")
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    root.mkdir(parents=True, exist_ok=True)
    _assert_safe_output_ancestors(root)
    _assert_no_symlinks(root)
    marker = _owned_marker(root)
    if marker.exists() and marker.read_text(encoding="utf-8") != "synthetic-trade-intel-demo-v1\n":
        raise ValueError("output ownership marker is invalid")
    marker.write_text("synthetic-trade-intel-demo-v1\n", encoding="utf-8")
    return root


def _write(root: Path, relative: str, payload: bytes, *, source_type: str, file_type: str,
           expected_entity: str, fact_type: str, locator: dict[str, Any], claim_id: str,
           records: list[dict[str, Any]]) -> None:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts or relative_path.as_posix() != relative:
        raise ValueError("generated path must be a safe relative POSIX path")
    _assert_no_symlinks(root)
    _assert_safe_output_ancestors(root)
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_safe_output_ancestors(root)
    _assert_no_symlinks(root)
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
    image = Image.new("RGB", (1200, 1550), "white")
    draw = ImageDraw.Draw(image)
    heading = ImageFont.load_default(size=42)
    body = ImageFont.load_default(size=27)
    table_font = ImageFont.load_default(size=22)
    warning_font = ImageFont.load_default(size=20)
    draw.rectangle((35, 35, 1165, 1515), outline="black", width=5)
    draw.rectangle((55, 70, 1145, 155), fill="black")
    draw.text((75, 102), _SCAN_WARNING, fill="white", font=warning_font)
    draw.text((75, 205), "SYNTHETIC TRADE STANDARDS OFFICE", fill="black", font=heading)
    draw.text((75, 270), "FICTIONAL SCANNED REGULATOR NOTICE / DEMO ONLY", fill="black", font=body)
    draw.text((75, 345), "NOTICE TABLE", fill="black", font=heading)
    rows = [("Rule", "Synthetic compliance window", "2026-09-01"), ("Scope", "Fictional HS sample", "010121"), ("Status", "Demo-only notice", "NOT PRODUCTION")]
    y = 420
    for row in [("Field", "Description", "Value"), *rows]:
        draw.rectangle((75, y, 1125, y + 90), outline="black", width=2)
        draw.line((350, y, 350, y + 90), fill="black", width=2)
        draw.line((800, y, 800, y + 90), fill="black", width=2)
        for x, value in zip((90, 365, 815), row):
            draw.text((x, y + 32), value, fill="black", font=table_font)
        y += 90
    draw.text((75, 850), _SCAN_WARNING, fill="black", font=warning_font)
    png_buffer = io.BytesIO()
    image.save(png_buffer, format="PNG", optimize=False, compress_level=9)
    buffer = io.BytesIO()
    pdf = Canvas(buffer, pagesize=(612, 792), invariant=1, pageCompression=1)
    pdf.setTitle("Synthetic scanned regulator notice")
    pdf.drawImage(ImageReader(io.BytesIO(png_buffer.getvalue())), 36, 36, width=540, height=720, mask="auto")
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
        products.append({"synthetic_notice": _BANNER, "reference_claim_id": f"CLAIM-B2B-{index + 1:03d}", "product_id": product.id, "sku": product.sku, "product_name": product.product_name, "hs_code": bundle.hs_codes[product.hs_code_id - 1].hs_code, "supplier": company.company_name, "url": f"https://marketplace.example/products/{product.sku.lower()}"})
    payload = _json({"synthetic_notice": _BANNER, "products": products})
    _write(root, "b2b/products.json", payload, source_type="b2b", file_type="json", expected_entity="Synthetic B2B Marketplace", fact_type="product_offering", locator={"table": "products", "items": [{"row": index + 1, "claim_id": product["reference_claim_id"]} for index, product in enumerate(products)]}, claim_id="CLAIM-B2B-CATALOG", records=records)
    records[-1]["reference_claim_ids"] = [product["reference_claim_id"] for product in products]
    return len(products)


def _news(root: Path, bundle: TradeSeedBundle, records: list[dict[str, Any]]) -> int:
    stories: list[dict[str, Any]] = []
    for index in range(12):
        company = bundle.companies[index]
        story_id = f"NEWS-{index + 1:03d}"
        stories.append({"id": story_id, "synthetic_notice": _BANNER, "reference_claim_id": f"CLAIM-{story_id}", "entity": company.company_name, "headline": f"Fictional trade signal for {company.company_name}", "body": f"{_BANNER} This fictional report describes a demo-only market signal.", "url": f"https://newsroom.example/story/{index + 1:03d}", "publisher": "Synthetic Primary Newswire", "canonical_story_id": story_id, "syndicated_from": None, "dedupe_cluster_id": f"NEWS-CLUSTER-{index + 1:03d}", "published_at": _STAMP})
    for index in range(4):
        canonical = stories[index]
        story_id = f"NEWS-{index + 13:03d}"
        stories.append({**canonical, "id": story_id, "reference_claim_id": f"CLAIM-{story_id}", "url": f"https://syndication.example/story/{index + 13:03d}", "publisher": "Synthetic Syndication Mirror", "syndicated_from": canonical["id"], "published_at": _STAMP})
    _write(root, "news/stories.json", _json({"synthetic_notice": _BANNER, "stories": stories}), source_type="industry_news", file_type="json", expected_entity="Synthetic Industry Newswire", fact_type="market_signal", locator={"table": "stories", "items": [{"row": index + 1, "story_id": story["id"], "canonical_story_id": story["canonical_story_id"], "syndicated_from": story["syndicated_from"], "dedupe_cluster_id": story["dedupe_cluster_id"], "claim_id": story["reference_claim_id"]} for index, story in enumerate(stories)]}, claim_id="CLAIM-NEWS-SET", records=records)
    records[-1]["reference_claim_ids"] = [story["reference_claim_id"] for story in stories]
    html = f"<html><body><p>{_BANNER}</p><article><h1>{stories[0]['headline']}</h1><p>{stories[0]['body']}</p></article></body></html>"
    _write(root, "news/representative-story.html", html.encode(), source_type="industry_news", file_type="html", expected_entity=bundle.companies[0].company_name, fact_type="market_signal", locator={"section": "article", "story_id": "NEWS-001"}, claim_id="CLAIM-NEWS-001", records=records)
    return len(stories)


def _social(root: Path, bundle: TradeSeedBundle, records: list[dict[str, Any]]) -> int:
    posts = [{"post_id": f"POST-{index + 1:03d}", "reference_claim_id": f"CLAIM-POST-{index + 1:03d}", "synthetic_notice": _BANNER, "company": bundle.companies[index].company_name, "text": f"{_BANNER} Fictional update about a sample shipment.", "url": f"https://social.example/posts/{index + 1:03d}", "published_at": _STAMP} for index in range(12)]
    _write(root, "social/posts.jsonl", b"".join(_json(post) for post in posts), source_type="social", file_type="jsonl", expected_entity="Synthetic Social Feed", fact_type="market_signal", locator={"items": [{"row": index + 1, "post_id": post["post_id"], "claim_id": post["reference_claim_id"]} for index, post in enumerate(posts)]}, claim_id="CLAIM-SOCIAL-SET", records=records)
    records[-1]["reference_claim_ids"] = [post["reference_claim_id"] for post in posts]
    return len(posts)


def _pdfs(root: Path, records: list[dict[str, Any]]) -> tuple[int, int]:
    _write(root, "pdf/regulator-notice.pdf", _text_pdf("Synthetic regulator notice", ["A demo-only compliance date is 2026-09-01."]), source_type="regulator", file_type="pdf", expected_entity="Synthetic Trade Standards Office", fact_type="regulation", locator={"page": 1, "kind": "text_pdf"}, claim_id="CLAIM-REG-001", records=records)
    _write(root, "pdf/market-bulletin.pdf", _text_pdf("Synthetic market bulletin", ["This report contains fictional aggregate indicators only."]), source_type="industry_news", file_type="pdf", expected_entity="Synthetic Market Bulletin", fact_type="market_signal", locator={"page": 1, "kind": "text_pdf"}, claim_id="CLAIM-PDF-001", records=records)
    _write(root, "pdf/scanned-regulator-notice.pdf", _scanned_pdf(), source_type="regulator", file_type="pdf", expected_entity="Synthetic Trade Standards Office", fact_type="regulation", locator={"page": 1, "kind": "image_only_scanned_pdf"}, claim_id="CLAIM-REG-SCAN-001", records=records)
    return 2, 1


def _profiles(root: Path, bundle: TradeSeedBundle, records: list[dict[str, Any]]) -> int:
    selected = {1, 2, 3}
    aggregates: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for row in bundle.trade_records:
        for company_id, role in ((row.importer_id, "import"), (row.exporter_id, "export")):
            if company_id not in selected:
                continue
            hs = bundle.hs_codes[row.hs_code_id - 1]
            company = bundle.companies[company_id - 1]
            key = (company_id, company.country_id, hs.hs_code, row.trade_date.strftime("%Y-%m"))
            value = aggregates.setdefault(key, {"total_amount": Decimal(), "import_amount": Decimal(), "export_amount": Decimal(), "total_quantity": Decimal(), "import_quantity": Decimal(), "export_quantity": Decimal(), "raw_ids": []})
            value["total_amount"] += row.trade_amount; value[f"{role}_amount"] += row.trade_amount
            value["total_quantity"] += row.quantity; value[f"{role}_quantity"] += row.quantity
            value["raw_ids"].append(row.raw_record_id)
    for index, ((company_id, country_id, hs_code, month), value) in enumerate(sorted(aggregates.items()), 1):
        company = bundle.companies[company_id - 1]
        country = bundle.countries[country_id - 1]
        raw_ids = sorted(value["raw_ids"])
        profile = {"synthetic_notice": _BANNER, "company": company.company_name, "company_id": company_id, "country_code": country.country_code, "hs_code": hs_code, "calendar_month": month, "currency": "USD", "aggregation_grain": "company_country_hs_calendar_month", "roles_included": ["import", "export"], "aggregation_window": {"start": f"{month}-01", "calendar_month": month}, "import_amount_usd": str(value["import_amount"]), "export_amount_usd": str(value["export_amount"]), "total_amount_usd": str(value["total_amount"]), "import_quantity_kg": str(value["import_quantity"]), "export_quantity_kg": str(value["export_quantity"]), "total_quantity_kg": str(value["total_quantity"]), "source_record_count": len(raw_ids), "raw_record_summary": {"record_ids": raw_ids[:5], "record_id_hash": hashlib.sha256(_json(raw_ids)).hexdigest()}}
        _write(root, f"customs_profiles/{company_id:03d}-{hs_code}-{month}.json", _json(profile), source_type="customs_profile", file_type="generated_profile", expected_entity=company.company_name, fact_type="trade_activity", locator={"profile": "monthly_company_hs", "company_id": company_id, "country_code": country.country_code, "hs_code": hs_code, "calendar_month": month, "aggregation_grain": profile["aggregation_grain"], "aggregate_rows": len(raw_ids)}, claim_id=f"CLAIM-CUSTOMS-{index:03d}", records=records)
    return len(aggregates)


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
