from __future__ import annotations

import hashlib
import json
import tempfile
from fnmatch import fnmatch
from decimal import Decimal
from pathlib import Path

import pytest
import yaml
from pypdf import PdfReader

from trade_agent.data.demo_generator import generate_demo_corpus
from trade_agent.db.seed import generate_trade_seed


def _hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_demo_has_every_required_source_and_file_type(tmp_path: Path) -> None:
    summary = generate_demo_corpus(tmp_path, seed=20260830)

    assert summary.source_types == {
        "official_website", "b2b", "industry_news", "social", "regulator", "customs_profile",
    }
    assert {"html", "markdown", "json", "jsonl", "pdf", "generated_profile"} <= summary.file_types
    assert summary.website_sections >= 12
    assert summary.b2b_products >= 18
    assert summary.news_stories >= 16
    assert summary.social_posts >= 12
    assert summary.text_pdf_count >= 2
    assert summary.scanned_pdf_count == 1


def test_demo_manifest_is_deterministic_and_rehashable(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    generate_demo_corpus(first, seed=20260830)
    generate_demo_corpus(second, seed=20260830)
    assert _hashes(first) == _hashes(second)

    manifest = json.loads((first / "manifests" / "corpus_manifest.json").read_text())
    for record in manifest["records"]:
        payload = first / record["path"]
        assert payload.is_file()
        assert record["content_hash"] == hashlib.sha256(payload.read_bytes()).hexdigest()
        assert record["is_synthetic"] is True
        assert record["expected_entity"]
        assert record["reference_claim_ids"]
        assert record["locator"]


def test_clean_requires_a_owned_marker_and_rejects_broad_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="broad"):
        generate_demo_corpus(Path.cwd(), clean=True)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "unrelated.txt").write_text("keep")
    with pytest.raises(ValueError, match="owned"):
        generate_demo_corpus(occupied, clean=True)
    with pytest.raises(ValueError, match="owned"):
        generate_demo_corpus(occupied)
    assert (occupied / "unrelated.txt").read_text() == "keep"
    owned = tmp_path / "owned"
    generate_demo_corpus(owned)
    generate_demo_corpus(owned, clean=True)
    assert (owned / "manifests" / "corpus_manifest.json").is_file()


def test_symlinked_generated_child_is_rejected_without_touching_outside(tmp_path: Path) -> None:
    owned, outside = tmp_path / "owned", tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("do not modify")
    generate_demo_corpus(owned)
    for path in (owned / "website").iterdir():
        path.unlink()
    (owned / "website").rmdir()
    (owned / "website").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        generate_demo_corpus(owned)
    with pytest.raises(ValueError, match="symlink"):
        generate_demo_corpus(owned, clean=True)
    assert sentinel.read_text() == "do not modify"


def test_symlinked_generated_file_is_rejected_without_touching_outside(tmp_path: Path) -> None:
    owned, outside = tmp_path / "owned", tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("do not modify")
    generate_demo_corpus(owned)
    target = owned / "website" / "section-01.html"
    target.unlink()
    target.symlink_to(sentinel)

    with pytest.raises(ValueError, match="symlink"):
        generate_demo_corpus(owned)
    with pytest.raises(ValueError, match="symlink"):
        generate_demo_corpus(owned, clean=True)
    assert sentinel.read_text() == "do not modify"


def test_symlinked_output_ancestor_is_rejected_before_creating_child(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("do not modify")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)
    output = linked_parent / "child"

    with pytest.raises(ValueError, match="symlink"):
        generate_demo_corpus(output)
    with pytest.raises(ValueError, match="symlink"):
        generate_demo_corpus(output, clean=True)
    assert not (outside / "child").exists()
    assert sentinel.read_text() == "do not modify"


def test_normal_var_tempfile_descendant_is_not_rejected_as_a_system_alias() -> None:
    if not Path("/var").is_symlink():
        pytest.skip("platform has no /var system alias")
    with tempfile.TemporaryDirectory(prefix="trade-demo-") as temp_dir:
        physical = Path(temp_dir).resolve()
        if not physical.is_relative_to(Path("/private/var")):
            pytest.skip("tempfile is not backed by macOS /private/var")
        lexical_temp = Path("/var") / physical.relative_to(Path("/private/var"))
        output = lexical_temp / "child"
        generate_demo_corpus(output)
        assert (output / "manifests" / "corpus_manifest.json").is_file()


def test_customs_profiles_are_monthly_company_hs_aggregates(tmp_path: Path) -> None:
    generate_demo_corpus(tmp_path)
    bundle = generate_trade_seed(20260830)
    expected: dict[tuple[int, str, str], dict[str, Decimal | int]] = {}
    for row in bundle.trade_records:
        for company_id, role in ((row.importer_id, "import"), (row.exporter_id, "export")):
            if company_id not in {1, 2, 3}:
                continue
            hs = bundle.hs_codes[row.hs_code_id - 1].hs_code
            key = (company_id, hs, row.trade_date.strftime("%Y-%m"))
            values = expected.setdefault(key, {"total": Decimal(), "import": Decimal(), "export": Decimal(), "rows": 0})
            values["total"] += row.trade_amount
            values[role] += row.trade_amount
            values["rows"] += 1
    files = sorted((tmp_path / "customs_profiles").glob("*.json"))
    assert len(files) == len(expected) < len(bundle.trade_records)
    for file in files:
        profile = json.loads(file.read_text())
        values = expected[(profile["company_id"], profile["hs_code"], profile["calendar_month"])]
        assert Decimal(profile["total_amount_usd"]) == values["total"]
        assert Decimal(profile["import_amount_usd"]) == values["import"]
        assert Decimal(profile["export_amount_usd"]) == values["export"]
        assert profile["source_record_count"] == values["rows"]
        assert profile["aggregation_grain"] == "company_country_hs_calendar_month"


def test_scanned_pdf_is_nontrivial_image_only_raster(tmp_path: Path) -> None:
    generate_demo_corpus(tmp_path)
    page = PdfReader(str(tmp_path / "pdf" / "scanned-regulator-notice.pdf")).pages[0]
    assert page.extract_text().strip() == ""
    images = list(page.images)
    assert len(images) == 1
    assert images[0].image.width >= 1000
    assert images[0].image.height >= 1400
    assert len(images[0].data) > 1_000


def test_catalog_matches_each_manifest_record_once_and_news_syndication_is_traceable(tmp_path: Path) -> None:
    generate_demo_corpus(tmp_path)
    manifest = json.loads((tmp_path / "manifests" / "corpus_manifest.json").read_text())
    catalog = yaml.safe_load(Path("data/sources/trade_intel_demo.yaml").read_text())
    for record in manifest["records"]:
        matches = [rule for rule in catalog["sources"] if record["file_type"] in rule["file_types"] and any(fnmatch(record["path"], pattern) for pattern in rule["paths"])]
        assert len(matches) == 1
        assert matches[0]["source_type"] == record["source_type"]
    stories = json.loads((tmp_path / "news" / "stories.json").read_text())["stories"]
    by_id = {story["id"]: story for story in stories}
    for number in range(13, 17):
        mirror = by_id[f"NEWS-{number:03d}"]
        canonical = by_id[mirror["syndicated_from"]]
        assert (mirror["headline"], mirror["body"], mirror["entity"]) == (canonical["headline"], canonical["body"], canonical["entity"])
        assert mirror["canonical_story_id"] == canonical["id"]
        assert mirror["dedupe_cluster_id"] == canonical["dedupe_cluster_id"]
    news_record = next(row for row in manifest["records"] if row["path"] == "news/stories.json")
    assert len(news_record["reference_claim_ids"]) == 16
    assert len(news_record["locator"]["items"]) == 16
