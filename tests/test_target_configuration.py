from pathlib import Path
import subprocess

import pytest

from main import build_parser
from rag_core.engineering.target import (
    resolve_target_repository,
    resolve_target_source_id,
)
from rag_core.engineering.sufficiency import EvidenceSufficiencyGuard
from rag_core.retrieval.engineering import SourceIntent, SourceIntentRouter
from scripts.bootstrap_ecommerce_demo import create_demo_repository


def test_target_repository_precedence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KNOWLEDGE_TARGET_REPO", "generic-env")
    monkeypatch.setenv("MINI_NANOBOT_REPO", "legacy-env")

    assert resolve_target_repository("explicit", legacy_repo="legacy-explicit") == "explicit"
    assert resolve_target_repository(legacy_repo="legacy-explicit") == "generic-env"


def test_target_repository_legacy_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("KNOWLEDGE_TARGET_REPO", raising=False)
    monkeypatch.setenv("MINI_NANOBOT_REPO", "legacy-env")

    assert resolve_target_repository(legacy_repo=Path("legacy-explicit")) == Path(
        "legacy-explicit"
    )
    assert resolve_target_repository() == "legacy-env"


def test_target_source_id_is_configurable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENGINEERING_TARGET_SOURCE_ID", "ecommerce_demo")

    assert resolve_target_source_id() == "ecommerce_demo"
    assert resolve_target_source_id("explicit_source") == "explicit_source"
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_target_source_id("   ")


@pytest.mark.parametrize("flag", ["--target-repo", "--mini-repo"])
def test_engineering_query_accepts_generic_and_legacy_cli_alias(flag: str):
    args = build_parser().parse_args(
        ["engineering-query", "where is reserve_stock", flag, "repo"]
    )

    assert args.target_repo == "repo"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("当前源码中 `OrderService.confirm_payment` 如何实现？", SourceIntent.IMPLEMENTATION),
        ("为什么库存采用 reservation 而不是直接扣减？", SourceIntent.DESIGN),
        ("根据 PayGate 官方规范，event_id 如何用于回调去重？", SourceIntent.OFFICIAL),
        ("当前实现与 PayGate 官方规范的回调去重是否一致？", SourceIntent.COMPARISON),
        ("生产环境当前支付回调 P99 是多少？", SourceIntent.DESIGN),
    ],
)
def test_ecommerce_queries_route_deterministically(query: str, expected: SourceIntent):
    assert SourceIntentRouter().classify(query) is expected


def test_bootstrap_creates_clean_independent_git_repository(tmp_path: Path):
    target = create_demo_repository(tmp_path / "commerce")

    assert (target / ".git").is_dir()
    assert "synthetic" in (target / "README.md").read_text(encoding="utf-8").casefold()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=target,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        create_demo_repository(target)


def test_production_metrics_fail_closed_with_specific_reason():
    sufficient, warnings, reason = EvidenceSufficiencyGuard().check(
        "生产环境当前支付回调 P99、SLA 和月均故障率分别是多少？",
        SourceIntent.DESIGN,
        [],
    )

    assert sufficient is False
    assert reason == "missing_production_telemetry"
    assert warnings
