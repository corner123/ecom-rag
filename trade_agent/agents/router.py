"""Deterministic route selection for the trade graph."""
from __future__ import annotations

import re

from trade_agent.agents.intent import IntentParser, QueryIntent, UnresolvedIntentConstraint
from trade_agent.agents.state import RoutePlan
from trade_agent.retrieval.filters import RetrievalFilter


class TradeRouter:
    def __init__(self, parser: IntentParser) -> None:
        if not hasattr(parser, "parse"):
            raise TypeError("router requires an intent parser")
        self.parser = parser

    def route(
        self, question: str, explicit_filters: RetrievalFilter | None = None
    ) -> RoutePlan:
        try:
            return self.route_intent(self.parser.parse(question, explicit_filters))
        except UnresolvedIntentConstraint:
            # Task routing can safely identify the required channel without
            # inventing the unresolved semantic constraint. Downstream SQL
            # planning still requires a fully validated QueryIntent.
            if re.match(r"^\s*[A-Za-z][A-Za-z0-9 .&'_-]{1,127}\s+(?:最近|近|过去)", question) and any(
                token in question.casefold() for token in ("采购", "进口", "出口", "贸易", "金额", "数量")
            ):
                return RoutePlan(
                    need_trade_data=True,
                    need_external_intel=False,
                    required_branches=("sql",),
                )
            raise

    @staticmethod
    def route_intent(intent: QueryIntent) -> RoutePlan:
        if type(intent) is not QueryIntent:
            raise TypeError("router requires an exact QueryIntent")
        refuse = intent.kind == "out_of_scope"
        return RoutePlan(
            need_trade_data=intent.need_trade_data,
            need_external_intel=intent.need_external_intel,
            refuse=refuse,
            refusal_reason=intent.out_of_scope_reason if refuse else None,
            required_branches=tuple(
                branch
                for branch, enabled in (
                    ("rag", intent.need_external_intel),
                    ("sql", intent.need_trade_data),
                )
                if enabled
            ),
        )
