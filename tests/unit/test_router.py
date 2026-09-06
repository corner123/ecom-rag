from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from trade_agent.agents.intent import IntentParser
from trade_agent.agents.nodes import GraphBudgets
from trade_agent.agents.router import TradeRouter


@pytest.fixture
def router() -> TradeRouter:
    return TradeRouter(IntentParser(as_of=date(2026, 9, 4)))


@pytest.mark.parametrize(
    ("question", "sql", "rag"),
    [
        ("ABC 最近半年采购额", True, False),
        ("ABC 官网最近是否扩产", False, True),
        ("ABC 是否值得跟进", True, True),
    ],
)
def test_router_selects_required_channels(
    router: TradeRouter, question: str, sql: bool, rag: bool
) -> None:
    plan = router.route(question)

    assert (plan.need_trade_data, plan.need_external_intel) == (sql, rag)


def test_router_returns_a_terminal_refusal_route(router: TradeRouter) -> None:
    plan = router.route("帮我删除数据库")

    assert plan.refuse is True
    assert plan.need_trade_data is False
    assert plan.need_external_intel is False
    assert plan.refusal_reason == "unsupported_request"


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_rewrites": 2},
        {"max_evidence_candidates": 0},
        {"max_evidence_candidates": 513},
        {"max_generation_tokens": 0},
        {"max_generation_tokens": 8_193},
    ],
)
def test_graph_budgets_enforce_single_rewrite_candidate_and_token_bounds(
    overrides: dict[str, int],
) -> None:
    with pytest.raises(ValidationError):
        GraphBudgets(**overrides)
