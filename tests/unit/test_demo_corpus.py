from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from trade_agent.data.demo_generator import generate_demo_corpus


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
