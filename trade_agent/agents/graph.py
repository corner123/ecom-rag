"""Compilation of the bounded SQL and RAG workflow with real LangGraph."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
import json
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from trade_agent.agents.nodes import (
    EvidenceBranch,
    EvidenceRepository,
    GraphBudgets,
    answer_draft_node,
    claim_guard_node,
    default_node_dependencies,
    entity_dedup_conflict_node,
    evidence_validator_node,
    finalizer_node,
    normalize_node,
    policy_gate_node,
    query_rewrite_node,
    rag_node,
    router_node,
    sql_node,
)
from trade_agent.agents.state import RoutePlan, TradeIntelState
from trade_agent.evidence.claim_guard import ClaimHallucinationGuard
from trade_agent.evidence.validator import EvidenceValidator, ValidationOutcome
from trade_agent.generation.base import AnswerGenerator


@dataclass(frozen=True)
class GraphDependencies:
    intent_parser: Any
    evidence_repository: EvidenceRepository
    as_of: date
    sql_branch: EvidenceBranch | None = None
    rag_branch: EvidenceBranch | None = None
    entity_bindings: Mapping[str, int] = field(default_factory=dict)
    sql_identity_hmac_key: bytes | None = None
    budgets: GraphBudgets = field(default_factory=GraphBudgets)
    evidence_validator: EvidenceValidator | None = None
    answer_generator: AnswerGenerator | None = None
    claim_guard: ClaimHallucinationGuard | None = None
    policy: Callable[[str], bool] | None = None
    query_rewriter: Callable[[str, int], str] | None = None


def build_trade_graph(
    deps: GraphDependencies, checkpointer=None
) -> CompiledStateGraph:
    """Build and compile the production graph; no local graph fake is used."""

    if type(deps) is not GraphDependencies:
        raise TypeError("deps must be exact GraphDependencies")
    node_deps = default_node_dependencies(
        intent_parser=deps.intent_parser,
        sql_branch=deps.sql_branch,
        rag_branch=deps.rag_branch,
        evidence_repository=deps.evidence_repository,
        entity_bindings=deps.entity_bindings,
        sql_identity_hmac_key=deps.sql_identity_hmac_key,
        budgets=deps.budgets,
        as_of=deps.as_of,
        evidence_validator=deps.evidence_validator,
        answer_generator=deps.answer_generator,
        claim_guard=deps.claim_guard,
        policy=deps.policy,
        query_rewriter=deps.query_rewriter,
    )
    graph = StateGraph(TradeIntelState)
    graph.add_node("policy_gate", policy_gate_node(node_deps))
    graph.add_node("router", router_node(node_deps))
    graph.add_node("sql_node", sql_node(node_deps))
    graph.add_node("rag_node", rag_node(node_deps))
    graph.add_node("normalize", normalize_node(node_deps))
    graph.add_node("entity_dedup_conflict", entity_dedup_conflict_node(node_deps))
    graph.add_node("evidence_validator", evidence_validator_node(node_deps))
    graph.add_node("query_rewrite", query_rewrite_node(node_deps))
    graph.add_node("answer_draft", answer_draft_node(node_deps))
    graph.add_node("claim_guard", claim_guard_node(node_deps))
    graph.add_node("finalizer", finalizer_node(node_deps))

    graph.add_edge(START, "policy_gate")
    graph.add_conditional_edges("policy_gate", _after_policy, {"router": "router", "finalizer": "finalizer"})
    graph.add_conditional_edges(
        "router",
        _after_router,
        {
            "sql_node": "sql_node",
            "rag_node": "rag_node",
            "normalize": "normalize",
            "finalizer": "finalizer",
        },
    )
    graph.add_edge("sql_node", "normalize")
    graph.add_edge("rag_node", "normalize")
    graph.add_conditional_edges("normalize", _after_regular_node, {"next": "entity_dedup_conflict", "finalizer": "finalizer"})
    graph.add_conditional_edges("entity_dedup_conflict", _after_regular_node, {"next": "evidence_validator", "finalizer": "finalizer"})
    graph.add_conditional_edges(
        "evidence_validator",
        lambda state: _after_validation(state, deps.budgets),
        {"rewrite": "query_rewrite", "answer": "answer_draft", "finalizer": "finalizer"},
    )
    graph.add_conditional_edges("query_rewrite", _after_rewrite, {"router": "router", "finalizer": "finalizer"})
    graph.add_conditional_edges("answer_draft", _after_regular_node, {"next": "claim_guard", "finalizer": "finalizer"})
    graph.add_conditional_edges("claim_guard", _after_regular_node, {"next": "finalizer", "finalizer": "finalizer"})
    graph.add_edge("finalizer", END)
    return graph.compile(checkpointer=checkpointer)


def _after_policy(state: TradeIntelState) -> str:
    return "finalizer" if state.get("terminal") else "router"


def _after_router(state: TradeIntelState) -> str | list[str]:
    if state.get("terminal"):
        return "finalizer"
    plan = RoutePlan.model_validate_json(json.dumps(state["route_plan"]))
    if plan.refuse:
        return "normalize"
    branches = [f"{branch}_node" for branch in plan.required_branches]
    return branches or "normalize"


def _after_regular_node(state: TradeIntelState) -> str:
    return "finalizer" if state.get("terminal") else "next"


def _after_validation(state: TradeIntelState, budgets: GraphBudgets) -> str:
    if state.get("terminal"):
        return "finalizer"
    outcome = ValidationOutcome.model_validate_json(json.dumps(state["validation"]))
    if (
        outcome.decision == "rewrite_once"
        and state.get("rewrite_count", 0) < budgets.max_rewrites
        and state.get("retry_count", 0) < budgets.max_retries
    ):
        return "rewrite"
    return "answer"


def _after_rewrite(state: TradeIntelState) -> str:
    return "finalizer" if state.get("terminal") else "router"
