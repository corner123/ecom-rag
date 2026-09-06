"""Bounded node implementations and the durable Evidence payload boundary."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import json
import os
from pathlib import Path
from queue import Empty, Queue
from threading import BoundedSemaphore, Thread
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

from trade_agent.agents.intent import QueryIntent
from trade_agent.agents.intent import to_retrieval_query_intent
from trade_agent.agents.state import EvidenceRef, RoutePlan, TradeIntelState
from trade_agent.data.manifest import BuildManifest
from trade_agent.db.registry import RegistrySnapshot
from trade_agent.db.sql_executor import (
    ReadOnlySqlExecutor,
    SqlExecutionError,
    SqlExecutionTimeout,
    SqlTransportError,
)
from trade_agent.db.sql_planner import SqlPlanner
from trade_agent.db.sql_renderer import SqlDataScope, SqlRenderer
from trade_agent.db.sql_validator import SqlValidator
from trade_agent.errors import EvidenceRepositoryError, GraphError
from trade_agent.evidence.claim_guard import ClaimHallucinationGuard, GuardOutcome
from trade_agent.evidence.models import Conflict, Evidence
from trade_agent.evidence.normalize import normalize_retrieval
from trade_agent.evidence.sql import build_sql_evidence
from trade_agent.evidence.validator import (
    BranchExecutionReport,
    EvidenceValidator,
    ValidationContext,
    ValidationOutcome,
)
from trade_agent.generation.base import AnswerGenerator, DraftAnswer
from trade_agent.generation.deterministic import DeterministicAnswerGenerator
from trade_agent.retrieval.filters import RetrievalFilter
from trade_agent.retrieval.service import RetrievalService


class GraphBudgets(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    max_steps: StrictInt = Field(default=16, ge=1, le=128)
    max_retries: StrictInt = Field(default=1, ge=0, le=4)
    max_rewrites: StrictInt = Field(default=1, ge=0, le=2)
    max_llm_calls: StrictInt = Field(default=1, ge=0, le=8)
    max_outstanding_calls: StrictInt = Field(default=16, ge=1, le=128)
    node_timeout_seconds: StrictFloat = Field(default=5.0, gt=0, le=120)


@runtime_checkable
class EvidenceRepository(Protocol):
    def put_many(self, evidence: Sequence[Evidence]) -> tuple[EvidenceRef, ...]: ...

    def get_many(self, refs: Sequence[EvidenceRef]) -> tuple[Evidence, ...]: ...


@runtime_checkable
class EvidenceBranch(Protocol):
    supports_finite_timeout: bool

    def run(
        self,
        intent: QueryIntent,
        *,
        timeout_seconds: float,
        identity_hmac_key: bytes | None = None,
        entity_bindings: Mapping[str, int] | None = None,
    ) -> tuple[Sequence[Evidence], Sequence[str]]: ...


@dataclass(frozen=True)
class ContractSqlBranch:
    """Compose the reviewed SQL contracts into one graph branch adapter."""

    registry: RegistrySnapshot
    scope: SqlDataScope
    connection_factory: Callable[[], Any]
    supports_finite_timeout: bool = True

    def run(
        self,
        intent: QueryIntent,
        *,
        timeout_seconds: float,
        identity_hmac_key: bytes | None = None,
        entity_bindings: Mapping[str, int] | None = None,
    ) -> tuple[Sequence[Evidence], Sequence[str]]:
        plan = SqlPlanner(entity_bindings=entity_bindings).plan(intent, self.registry)
        rendered = SqlRenderer(scope=self.scope).render(plan)
        validated = SqlValidator(scope=self.scope).validate(rendered, self.registry)
        timeout_ms = max(1, int(timeout_seconds * 1_000))
        connection = self.connection_factory()
        try:
            result = ReadOnlySqlExecutor(
                connection,
                identity_hmac_key=identity_hmac_key,
                max_execution_time_ms=timeout_ms,
                client_timeout_ms=timeout_ms,
            ).execute(validated)
        finally:
            close = getattr(connection, "close", None)
            if callable(close):
                close()
        return tuple(build_sql_evidence(result)), ()


@dataclass(frozen=True)
class ContractRetrievalBranch:
    """Use RetrievalService and the published manifest normalization boundary."""

    service: RetrievalService
    published_manifest: BuildManifest
    top_k: int = 10

    @property
    def supports_finite_timeout(self) -> bool:
        return self.service.supports_finite_transport_timeout

    def run(
        self,
        intent: QueryIntent,
        *,
        timeout_seconds: float,
        identity_hmac_key: bytes | None = None,
        entity_bindings: Mapping[str, int] | None = None,
    ) -> tuple[Sequence[Evidence], Sequence[str]]:
        del identity_hmac_key, entity_bindings
        outcome = self.service.search(
            to_retrieval_query_intent(intent),
            top_k=self.top_k,
            transport_timeout_seconds=timeout_seconds,
        )
        evidence = normalize_retrieval(
            outcome, published_manifest=self.published_manifest
        )
        degraded = tuple(
            component
            for component in sorted(set(outcome.degradation))
            if component in {
                "dense_unavailable",
                "reranker_unavailable",
                "sparse_unavailable",
                "structured_output_unavailable",
            }
        )
        return tuple(evidence), degraded


class FileEvidenceRepository:
    """Atomic file-backed Evidence repository suitable for durable resume."""

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path) or not root.is_absolute():
            raise ValueError("evidence repository root must be an absolute Path")
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _payload(item: Evidence) -> bytes:
        checked = Evidence.model_validate(item.model_dump(mode="python"))
        return checked.model_dump_json().encode("utf-8")

    def put_many(self, evidence: Sequence[Evidence]) -> tuple[EvidenceRef, ...]:
        if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
            raise EvidenceRepositoryError("Evidence payloads must be a sequence")
        refs: list[EvidenceRef] = []
        for item in evidence:
            if type(item) is not Evidence:
                raise EvidenceRepositoryError("repository accepts exact Evidence contracts")
            payload = self._payload(item)
            digest = sha256(payload).hexdigest()
            target = self.root / f"{item.evidence_id}.json"
            if target.exists():
                if target.read_bytes() != payload:
                    raise EvidenceRepositoryError("stored Evidence identity has a different payload")
            else:
                temporary = self.root / f".{item.evidence_id}.{os.getpid()}.tmp"
                try:
                    with temporary.open("xb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, target)
                except FileExistsError:
                    if not target.exists() or target.read_bytes() != payload:
                        raise EvidenceRepositoryError("concurrent Evidence write disagreed")
                finally:
                    temporary.unlink(missing_ok=True)
            refs.append(EvidenceRef(evidence_id=item.evidence_id, payload_sha256=digest))
        return tuple(sorted(refs, key=lambda item: item.evidence_id))

    def get_many(self, refs: Sequence[EvidenceRef]) -> tuple[Evidence, ...]:
        if isinstance(refs, (str, bytes)) or not isinstance(refs, Sequence):
            raise EvidenceRepositoryError("Evidence references must be a sequence")
        values: list[Evidence] = []
        for ref in refs:
            checked = EvidenceRef.model_validate(ref.model_dump(mode="python"))
            path = self.root / f"{checked.evidence_id}.json"
            try:
                payload = path.read_bytes()
            except OSError as error:
                raise EvidenceRepositoryError("referenced Evidence payload is unavailable") from error
            if sha256(payload).hexdigest() != checked.payload_sha256:
                raise EvidenceRepositoryError("referenced Evidence payload failed hash verification")
            try:
                item = Evidence.model_validate_json(payload)
            except Exception as error:
                raise EvidenceRepositoryError("referenced Evidence payload failed contract validation") from error
            if item.evidence_id != checked.evidence_id:
                raise EvidenceRepositoryError("referenced Evidence identity differs from payload")
            values.append(item)
        return tuple(values)


@dataclass(frozen=True)
class NodeDependencies:
    intent_parser: Any
    sql_branch: EvidenceBranch | None
    rag_branch: EvidenceBranch | None
    evidence_repository: EvidenceRepository
    entity_bindings: Mapping[str, int]
    sql_identity_hmac_key: bytes | None
    budgets: GraphBudgets
    as_of: date
    evidence_validator: EvidenceValidator
    answer_generator: AnswerGenerator
    claim_guard: ClaimHallucinationGuard
    policy: Callable[[str], bool]
    query_rewriter: Callable[[str, int], str] | None
    branch_call_runner: "BoundedCallRunner"


def _error(code: str, node: str, *, retryable: bool, detail: str) -> dict[str, object]:
    return GraphError(code=code, node=node, retryable=retryable, detail=detail).model_dump(mode="json")


def _begin(state: TradeIntelState, node: str, deps: NodeDependencies) -> dict[str, object] | None:
    if state.get("terminal"):
        return {"step_count": 0, "node_status": {node: "skipped"}}
    if state.get("step_count", 0) >= deps.budgets.max_steps:
        return {
            "step_count": 0,
            "terminal": True,
            "answer": None,
            "refusal_reason": "workflow_step_limit",
            "errors": [_error("step_limit_exceeded", node, retryable=False, detail="graph step budget exhausted")],
            "node_status": {node: "failed"},
        }
    return None


def _next_step(state: TradeIntelState) -> int:
    return state.get("step_count", 0) + 1


def _bounded_call(function: Callable[[], Any], timeout_seconds: float) -> Any:
    output: Queue[tuple[bool, Any]] = Queue(maxsize=1)

    def invoke() -> None:
        try:
            output.put((True, function()))
        except BaseException as error:  # classified by the owning node
            output.put((False, error))

    Thread(target=invoke, daemon=True, name="trade-graph-bounded-call").start()
    try:
        ok, value = output.get(timeout=timeout_seconds)
    except Empty as error:
        raise TimeoutError("node deadline exceeded") from error
    if ok:
        return value
    raise value


class BoundedCallRunner:
    """Cap timed-out branch work that remains alive below the transport seam."""

    def __init__(self, max_outstanding_calls: int) -> None:
        self._slots = BoundedSemaphore(max_outstanding_calls)

    def call(self, function: Callable[[], Any], timeout_seconds: float) -> Any:
        if not self._slots.acquire(blocking=False):
            raise TimeoutError("outstanding branch call capacity exhausted")
        output: Queue[tuple[bool, Any]] = Queue(maxsize=1)

        def invoke() -> None:
            try:
                output.put((True, function()))
            except BaseException as error:  # classified by the owning node
                output.put((False, error))
            finally:
                self._slots.release()

        worker = Thread(
            target=invoke,
            daemon=True,
            name="trade-graph-bounded-branch-call",
        )
        try:
            worker.start()
        except BaseException:
            self._slots.release()
            raise
        try:
            ok, value = output.get(timeout=timeout_seconds)
        except Empty as error:
            raise TimeoutError("node deadline exceeded") from error
        if ok:
            return value
        raise value


def _intent(state: TradeIntelState) -> QueryIntent:
    return _from_json_state(QueryIntent, state["intent"])


def _refs(values: Sequence[dict[str, object]]) -> tuple[EvidenceRef, ...]:
    return tuple(_from_json_state(EvidenceRef, item) for item in values)


def _from_json_state(contract, value):
    return contract.model_validate_json(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def policy_gate_node(deps: NodeDependencies):
    def node(state: TradeIntelState) -> dict[str, object]:
        limited = _begin(state, "policy_gate", deps)
        if limited:
            return limited
        question = state.get("current_question") or state.get("question", "")
        try:
            allowed = _bounded_call(
                lambda: deps.policy(question), deps.budgets.node_timeout_seconds
            )
        except Exception:
            allowed = False
        if not isinstance(question, str) or not question.strip() or len(question) > 4_000 or allowed is not True:
            return {
                "step_count": _next_step(state),
                "terminal": True,
                "answer": None,
                "refusal_reason": "policy_denied",
                "errors": [_error("policy_denied", "policy_gate", retryable=False, detail="request did not pass local policy")],
                "node_status": {"policy_gate": "failed"},
            }
        return {
            "step_count": _next_step(state),
            "current_question": question,
            "rewrite_count": state.get("rewrite_count", 0),
            "retry_count": state.get("retry_count", 0),
            "llm_calls": state.get("llm_calls", 0),
            "node_status": {"policy_gate": "completed"},
        }

    return node


def router_node(deps: NodeDependencies):
    def node(state: TradeIntelState) -> dict[str, object]:
        limited = _begin(state, "router", deps)
        if limited:
            return limited
        try:
            raw_filters = state.get("explicit_filters")
            filters = _from_json_state(RetrievalFilter, raw_filters) if raw_filters else None
            intent = _bounded_call(
                lambda: deps.intent_parser.parse(state["current_question"], filters),
                deps.budgets.node_timeout_seconds,
            )
            if type(intent) is not QueryIntent:
                raise TypeError("intent parser returned a foreign contract")
            plan = RoutePlan(
                need_trade_data=intent.need_trade_data,
                need_external_intel=intent.need_external_intel,
                refuse=intent.kind == "out_of_scope",
                refusal_reason=intent.out_of_scope_reason if intent.kind == "out_of_scope" else None,
                required_branches=tuple(
                    branch
                    for branch, enabled in (("rag", intent.need_external_intel), ("sql", intent.need_trade_data))
                    if enabled
                ),
            )
            return {
                "step_count": _next_step(state),
                "intent": intent.model_dump(mode="json"),
                "route_plan": plan.model_dump(mode="json"),
                "node_status": {"router": "completed"},
            }
        except TimeoutError:
            return {
                "step_count": _next_step(state),
                "terminal": True,
                "answer": None,
                "refusal_reason": "intent_timeout",
                "errors": [_error("rag_timeout", "router", retryable=False, detail="intent parsing deadline exceeded")],
                "node_status": {"router": "failed"},
            }
        except (TypeError, ValueError) as error:
            return {
                "step_count": _next_step(state),
                "terminal": True,
                "answer": None,
                "refusal_reason": "invalid_request",
                "errors": [_error("invalid_request", "router", retryable=False, detail=type(error).__name__)],
                "node_status": {"router": "failed"},
            }
        except Exception as error:
            return {
                "step_count": _next_step(state),
                "terminal": True,
                "answer": None,
                "refusal_reason": "workflow_failed",
                "errors": [_error("internal_error", "router", retryable=False, detail=type(error).__name__)],
                "node_status": {"router": "failed"},
            }

    return node


def _branch_node(deps: NodeDependencies, branch_name: str):
    branch = deps.sql_branch if branch_name == "sql" else deps.rag_branch
    refs_key = f"{branch_name}_evidence_refs"
    report_key = f"{branch_name}_report"

    def node(state: TradeIntelState) -> dict[str, object]:
        limited = _begin(state, f"{branch_name}_node", deps)
        if limited:
            return limited
        intent = _intent(state)
        if branch_name == "sql":
            entity_ids = tuple(sorted(set((*intent.filters.entity_ids, *intent.retrieval_filter.entity_ids))))
            bindings = deps.entity_bindings
            if any(item not in bindings or type(bindings[item]) is not int or bindings[item] <= 0 for item in entity_ids):
                report = BranchExecutionReport(branch="sql", attempted=False, completed=False, zero_hits=False, error_codes=("policy_denied",))
                return {
                    "step_count": _next_step(state),
                    refs_key: [],
                    report_key: report.model_dump(mode="json"),
                    "errors": [_error("missing_entity_binding", "sql_node", retryable=False, detail="entity has no reviewed positive company primary-key binding")],
                    "node_status": {"sql_node": "failed"},
                }
            if type(deps.sql_identity_hmac_key) is not bytes or not deps.sql_identity_hmac_key:
                report = BranchExecutionReport(branch="sql", attempted=False, completed=False, zero_hits=False, error_codes=("invalid_contract",))
                return {
                    "step_count": _next_step(state),
                    refs_key: [],
                    report_key: report.model_dump(mode="json"),
                    "errors": [_error("missing_sql_hmac_key", "sql_node", retryable=False, detail="runtime SQL identity HMAC key is absent")],
                    "node_status": {"sql_node": "failed"},
                }
        if branch is None:
            report = BranchExecutionReport(branch=branch_name, attempted=False, completed=False, zero_hits=False, error_codes=("unavailable",))
            return {
                "step_count": _next_step(state),
                refs_key: [],
                report_key: report.model_dump(mode="json"),
                "errors": [_error(f"{branch_name}_unavailable", f"{branch_name}_node", retryable=True, detail="required branch dependency is unavailable")],
                "node_status": {f"{branch_name}_node": "failed"},
            }
        if getattr(branch, "supports_finite_timeout", False) is not True:
            report = BranchExecutionReport(branch=branch_name, attempted=False, completed=False, zero_hits=False, error_codes=("invalid_contract",))
            return {
                "step_count": _next_step(state),
                refs_key: [],
                report_key: report.model_dump(mode="json"),
                "errors": [_error(
                    "retrieval_timeout_unsupported" if branch_name == "rag" else "sql_unavailable",
                    f"{branch_name}_node",
                    retryable=False,
                    detail="branch adapter cannot enforce a finite transport timeout",
                )],
                "node_status": {f"{branch_name}_node": "failed"},
            }
        try:
            result = deps.branch_call_runner.call(
                lambda: branch.run(
                    intent,
                    timeout_seconds=deps.budgets.node_timeout_seconds,
                    identity_hmac_key=deps.sql_identity_hmac_key if branch_name == "sql" else None,
                    entity_bindings=deps.entity_bindings if branch_name == "sql" else None,
                ),
                deps.budgets.node_timeout_seconds,
            )
            evidence, degraded = result
            if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
                raise TypeError("branch evidence must be a sequence")
            if any(type(item) is not Evidence for item in evidence):
                raise TypeError("branch evidence must use the exact Evidence contract")
            checked = tuple(Evidence.model_validate(item.model_dump(mode="python")) for item in evidence)
            refs = deps.evidence_repository.put_many(checked)
            partial = tuple(sorted(set(degraded)))
            report = BranchExecutionReport(
                branch=branch_name,
                attempted=True,
                completed=True,
                zero_hits=not checked,
                degraded_components=partial,
            )
            return {
                "step_count": _next_step(state),
                refs_key: [item.model_dump(mode="json") for item in refs],
                report_key: report.model_dump(mode="json"),
                "node_status": {f"{branch_name}_node": "completed"},
            }
        except TimeoutError:
            report = BranchExecutionReport(branch=branch_name, attempted=True, completed=False, zero_hits=False, error_codes=("timeout",))
            return {
                "step_count": _next_step(state),
                refs_key: [],
                report_key: report.model_dump(mode="json"),
                "errors": [_error(f"{branch_name}_timeout", f"{branch_name}_node", retryable=True, detail="branch deadline exceeded")],
                "node_status": {f"{branch_name}_node": "retryable_failed"},
            }
        except SqlExecutionTimeout:
            report = BranchExecutionReport(branch=branch_name, attempted=True, completed=False, zero_hits=False, error_codes=("timeout",))
            return {
                "step_count": _next_step(state),
                refs_key: [],
                report_key: report.model_dump(mode="json"),
                "errors": [_error("sql_timeout", "sql_node", retryable=True, detail="SQL execution deadline exceeded")],
                "node_status": {"sql_node": "retryable_failed"},
            }
        except (SqlTransportError, OSError):
            report = BranchExecutionReport(branch=branch_name, attempted=True, completed=False, zero_hits=False, error_codes=("transport",))
            return {
                "step_count": _next_step(state),
                refs_key: [],
                report_key: report.model_dump(mode="json"),
                "errors": [_error(f"{branch_name}_transport", f"{branch_name}_node", retryable=True, detail="branch transport failed")],
                "node_status": {f"{branch_name}_node": "retryable_failed"},
            }
        except SqlExecutionError:
            report = BranchExecutionReport(branch=branch_name, attempted=True, completed=False, zero_hits=False, error_codes=("unknown",))
            return {
                "step_count": _next_step(state),
                refs_key: [],
                report_key: report.model_dump(mode="json"),
                "errors": [_error("sql_unavailable", "sql_node", retryable=False, detail="SQL execution failed closed")],
                "node_status": {"sql_node": "failed"},
            }
        except EvidenceRepositoryError:
            report = BranchExecutionReport(branch=branch_name, attempted=True, completed=False, zero_hits=False, error_codes=("invalid_contract",))
            return {
                "step_count": _next_step(state),
                refs_key: [],
                report_key: report.model_dump(mode="json"),
                "errors": [_error("evidence_repository_error", f"{branch_name}_node", retryable=False, detail="Evidence persistence failed")],
                "node_status": {f"{branch_name}_node": "failed"},
            }
        except (TypeError, ValueError):
            report = BranchExecutionReport(branch=branch_name, attempted=True, completed=False, zero_hits=False, error_codes=("invalid_contract",))
            return {
                "step_count": _next_step(state),
                refs_key: [],
                report_key: report.model_dump(mode="json"),
                "errors": [_error("invalid_contract", f"{branch_name}_node", retryable=False, detail="branch returned an invalid contract")],
                "node_status": {f"{branch_name}_node": "failed"},
            }
        except Exception as error:
            report = BranchExecutionReport(branch=branch_name, attempted=True, completed=False, zero_hits=False, error_codes=("unknown",))
            return {
                "step_count": _next_step(state),
                refs_key: [],
                report_key: report.model_dump(mode="json"),
                "errors": [_error("internal_error", f"{branch_name}_node", retryable=False, detail=type(error).__name__)],
                "node_status": {f"{branch_name}_node": "failed"},
            }

    return node


def sql_node(deps: NodeDependencies):
    return _branch_node(deps, "sql")


def rag_node(deps: NodeDependencies):
    return _branch_node(deps, "rag")


def normalize_node(deps: NodeDependencies):
    def node(state: TradeIntelState) -> dict[str, object]:
        limited = _begin(state, "normalize", deps)
        if limited:
            return limited
        plan = _from_json_state(RoutePlan, state["route_plan"])
        refs: list[EvidenceRef] = []
        reports: list[BranchExecutionReport] = []
        for branch in plan.required_branches:
            refs.extend(_refs(state.get(f"{branch}_evidence_refs", [])))
            raw_report = state.get(f"{branch}_report")
            if raw_report is None:
                reports.append(BranchExecutionReport(branch=branch, attempted=False, completed=False, zero_hits=False, error_codes=("unknown",)))
            else:
                reports.append(_from_json_state(BranchExecutionReport, raw_report))
        unique = {item.evidence_id: item for item in refs}
        return {
            "step_count": _next_step(state),
            "evidence_refs": [unique[key].model_dump(mode="json") for key in sorted(unique)],
            "branch_reports": [item.model_dump(mode="json") for item in sorted(reports, key=lambda item: item.branch)],
            "node_status": {"normalize": "completed"},
        }

    return node


def entity_dedup_conflict_node(deps: NodeDependencies):
    def node(state: TradeIntelState) -> dict[str, object]:
        limited = _begin(state, "entity_dedup_conflict", deps)
        if limited:
            return limited
        # Retrieval performs lineage-aware deduplication before normalization.
        # This graph boundary preserves unique Evidence IDs and leaves exact
        # structured conflict decisions to the shared validator contract.
        return {"step_count": _next_step(state), "conflicts": [], "node_status": {"entity_dedup_conflict": "completed"}}

    return node


def evidence_validator_node(deps: NodeDependencies):
    def node(state: TradeIntelState) -> dict[str, object]:
        limited = _begin(state, "evidence_validator", deps)
        if limited:
            return limited
        try:
            evidence = deps.evidence_repository.get_many(_refs(state.get("evidence_refs", [])))
            reports = tuple(_from_json_state(BranchExecutionReport, item) for item in state.get("branch_reports", []))
            context = ValidationContext(branch_reports=reports)
            conflicts = tuple(_from_json_state(Conflict, item) for item in state.get("conflicts", []))
            outcome = _bounded_call(
                lambda: deps.evidence_validator.validate(
                    _intent(state), evidence, conflicts, deps.as_of, context=context
                ),
                deps.budgets.node_timeout_seconds,
            )
            return {
                "step_count": _next_step(state),
                "validation": outcome.model_dump(mode="json"),
                "node_status": {"evidence_validator": "completed"},
            }
        except TimeoutError:
            return {
                "step_count": _next_step(state),
                "terminal": True,
                "answer": None,
                "refusal_reason": "validation_timeout",
                "errors": [_error("internal_error", "evidence_validator", retryable=False, detail="validation deadline exceeded")],
                "node_status": {"evidence_validator": "failed"},
            }
        except EvidenceRepositoryError:
            return {
                "step_count": _next_step(state),
                "terminal": True,
                "answer": None,
                "refusal_reason": "evidence_repository_error",
                "errors": [_error("evidence_repository_error", "evidence_validator", retryable=False, detail="Evidence reload failed")],
                "node_status": {"evidence_validator": "failed"},
            }
        except (TypeError, ValueError):
            return {
                "step_count": _next_step(state),
                "terminal": True,
                "answer": None,
                "refusal_reason": "invalid_contract",
                "errors": [_error("invalid_contract", "evidence_validator", retryable=False, detail="validation contract failed")],
                "node_status": {"evidence_validator": "failed"},
            }
        except Exception as error:
            return {
                "step_count": _next_step(state),
                "terminal": True,
                "answer": None,
                "refusal_reason": "workflow_failed",
                "errors": [_error("internal_error", "evidence_validator", retryable=False, detail=type(error).__name__)],
                "node_status": {"evidence_validator": "failed"},
            }

    return node


def query_rewrite_node(deps: NodeDependencies):
    def node(state: TradeIntelState) -> dict[str, object]:
        limited = _begin(state, "query_rewrite", deps)
        if limited:
            return limited
        count = state.get("rewrite_count", 0)
        if count >= deps.budgets.max_rewrites or state.get("retry_count", 0) >= deps.budgets.max_retries:
            return {
                "step_count": _next_step(state),
                "terminal": True,
                "answer": None,
                "refusal_reason": "evidence_insufficient_after_rewrite",
                "errors": [_error("retry_limit_exceeded", "query_rewrite", retryable=False, detail="rewrite/retry budget exhausted")],
                "node_status": {"query_rewrite": "failed"},
            }
        if deps.query_rewriter is None:
            rewritten = f"{state['question']} [rewrite-{count + 1}]"
            llm_calls = state.get("llm_calls", 0)
        else:
            llm_calls = state.get("llm_calls", 0)
            if llm_calls >= deps.budgets.max_llm_calls:
                return {
                    "step_count": _next_step(state),
                    "terminal": True,
                    "answer": None,
                    "refusal_reason": "llm_budget_exhausted",
                    "errors": [_error("llm_limit_exceeded", "query_rewrite", retryable=False, detail="LLM call budget exhausted")],
                    "node_status": {"query_rewrite": "failed"},
                }
            llm_calls += 1
            try:
                rewritten = _bounded_call(
                    lambda: deps.query_rewriter(state["current_question"], count + 1),
                    deps.budgets.node_timeout_seconds,
                )
            except TimeoutError:
                return {
                    "step_count": _next_step(state),
                    "terminal": True,
                    "answer": None,
                    "llm_calls": llm_calls,
                    "refusal_reason": "rewrite_timeout",
                    "errors": [_error("rag_timeout", "query_rewrite", retryable=False, detail="query rewrite deadline exceeded")],
                    "node_status": {"query_rewrite": "failed"},
                }
            except Exception as error:
                return {
                    "step_count": _next_step(state),
                    "terminal": True,
                    "answer": None,
                    "llm_calls": llm_calls,
                    "refusal_reason": "workflow_failed",
                    "errors": [
                        _error(
                            "internal_error",
                            "query_rewrite",
                            retryable=False,
                            detail=type(error).__name__,
                        )
                    ],
                    "node_status": {"query_rewrite": "failed"},
                }
        if not isinstance(rewritten, str) or not rewritten.strip() or len(rewritten) > 4_000:
            return {
                "step_count": _next_step(state),
                "terminal": True,
                "answer": None,
                "refusal_reason": "invalid_rewrite",
                "errors": [_error("invalid_contract", "query_rewrite", retryable=False, detail="query rewriter returned invalid text")],
                "node_status": {"query_rewrite": "failed"},
            }
        return {
            "step_count": _next_step(state),
            "current_question": rewritten,
            "rewrite_count": count + 1,
            "retry_count": state.get("retry_count", 0) + 1,
            "llm_calls": llm_calls,
            "sql_evidence_refs": [],
            "rag_evidence_refs": [],
            "evidence_refs": [],
            "branch_reports": [],
            "conflicts": [],
            "node_status": {"query_rewrite": "completed"},
        }

    return node


def answer_draft_node(deps: NodeDependencies):
    def node(state: TradeIntelState) -> dict[str, object]:
        limited = _begin(state, "answer_draft", deps)
        if limited:
            return limited
        provider_call = type(deps.answer_generator) is not DeterministicAnswerGenerator
        llm_calls = state.get("llm_calls", 0)
        if provider_call and llm_calls >= deps.budgets.max_llm_calls:
            return {
                "step_count": _next_step(state),
                "terminal": True,
                "answer": None,
                "refusal_reason": "llm_budget_exhausted",
                "errors": [
                    _error(
                        "llm_limit_exceeded",
                        "answer_draft",
                        retryable=False,
                        detail="LLM call budget exhausted",
                    )
                ],
                "node_status": {"answer_draft": "failed"},
            }
        if provider_call:
            llm_calls += 1
        try:
            evidence = deps.evidence_repository.get_many(_refs(state.get("evidence_refs", [])))
            validation = _from_json_state(ValidationOutcome, state["validation"])
            draft = _bounded_call(
                lambda: deps.answer_generator.generate(_intent(state), evidence, validation),
                deps.budgets.node_timeout_seconds,
            )
            if type(draft) is not DraftAnswer:
                raise TypeError("answer generator returned a foreign contract")
            return {
                "step_count": _next_step(state),
                "llm_calls": llm_calls,
                "draft": draft.model_dump(mode="json"),
                "node_status": {"answer_draft": "completed"},
            }
        except TimeoutError:
            return {
                "step_count": _next_step(state), "terminal": True, "answer": None,
                "llm_calls": llm_calls,
                "refusal_reason": "generation_timeout",
                "errors": [_error("generation_timeout", "answer_draft", retryable=False, detail="answer generation deadline exceeded")],
                "node_status": {"answer_draft": "failed"},
            }
        except EvidenceRepositoryError:
            return {
                "step_count": _next_step(state), "terminal": True, "answer": None,
                "llm_calls": llm_calls,
                "refusal_reason": "evidence_repository_error",
                "errors": [_error("evidence_repository_error", "answer_draft", retryable=False, detail="Evidence reload failed")],
                "node_status": {"answer_draft": "failed"},
            }
        except (TypeError, ValueError):
            return {
                "step_count": _next_step(state), "terminal": True, "answer": None,
                "llm_calls": llm_calls,
                "refusal_reason": "invalid_contract",
                "errors": [_error("invalid_contract", "answer_draft", retryable=False, detail="answer generation contract failed")],
                "node_status": {"answer_draft": "failed"},
            }
        except Exception as error:
            return {
                "step_count": _next_step(state), "terminal": True, "answer": None,
                "llm_calls": llm_calls,
                "refusal_reason": "workflow_failed",
                "errors": [_error("internal_error", "answer_draft", retryable=False, detail=type(error).__name__)],
                "node_status": {"answer_draft": "failed"},
            }

    return node


def claim_guard_node(deps: NodeDependencies):
    def node(state: TradeIntelState) -> dict[str, object]:
        limited = _begin(state, "claim_guard", deps)
        if limited:
            return limited
        try:
            evidence = deps.evidence_repository.get_many(_refs(state.get("evidence_refs", [])))
            outcome = _bounded_call(
                lambda: deps.claim_guard.guard(
                    _from_json_state(DraftAnswer, state["draft"]),
                    evidence,
                    _intent(state),
                    _from_json_state(ValidationOutcome, state["validation"]),
                ),
                deps.budgets.node_timeout_seconds,
            )
            if type(outcome) is not GuardOutcome:
                raise TypeError("claim guard returned a foreign contract")
            return {"step_count": _next_step(state), "guard": outcome.model_dump(mode="json"), "node_status": {"claim_guard": "completed"}}
        except TimeoutError:
            return {
                "step_count": _next_step(state), "terminal": True, "answer": None,
                "refusal_reason": "guard_timeout",
                "errors": [_error("guard_timeout", "claim_guard", retryable=False, detail="claim guard deadline exceeded")],
                "node_status": {"claim_guard": "failed"},
            }
        except EvidenceRepositoryError:
            return {
                "step_count": _next_step(state), "terminal": True, "answer": None,
                "refusal_reason": "evidence_repository_error",
                "errors": [_error("evidence_repository_error", "claim_guard", retryable=False, detail="Evidence reload failed")],
                "node_status": {"claim_guard": "failed"},
            }
        except (TypeError, ValueError):
            return {
                "step_count": _next_step(state), "terminal": True, "answer": None,
                "refusal_reason": "invalid_contract",
                "errors": [_error("invalid_contract", "claim_guard", retryable=False, detail="claim guard contract failed")],
                "node_status": {"claim_guard": "failed"},
            }
        except Exception as error:
            return {
                "step_count": _next_step(state), "terminal": True, "answer": None,
                "refusal_reason": "workflow_failed",
                "errors": [_error("internal_error", "claim_guard", retryable=False, detail=type(error).__name__)],
                "node_status": {"claim_guard": "failed"},
            }

    return node


def finalizer_node(deps: NodeDependencies):
    def node(state: TradeIntelState) -> dict[str, object]:
        if state.get("terminal"):
            return {"step_count": 0, "node_status": {"finalizer": "completed"}}
        limited = _begin(state, "finalizer", deps)
        if limited:
            return limited
        raw = state.get("guard")
        if raw is None:
            return {
                "step_count": _next_step(state),
                "terminal": True,
                "answer": None,
                "refusal_reason": state.get("refusal_reason") or "workflow_failed",
                "claims": [],
                "node_status": {"finalizer": "completed"},
            }
        outcome = _from_json_state(GuardOutcome, raw)
        refusal = outcome.refusal_reason
        validation = _from_json_state(ValidationOutcome, state["validation"])
        if validation.decision == "rewrite_once" and state.get("rewrite_count", 0) >= deps.budgets.max_rewrites:
            refusal = "evidence_insufficient_after_rewrite"
        answer = (
            "\n".join(claim.text for claim in outcome.claims)
            if outcome.accepted
            else None
        )
        return {
            "step_count": _next_step(state),
            "terminal": True,
            # Public answer is reconstructed solely by ClaimHallucinationGuard.
            "answer": answer,
            "claims": [claim.model_dump(mode="json") for claim in outcome.claims],
            "refusal_reason": refusal,
            "node_status": {"finalizer": "completed"},
        }

    return node


def default_node_dependencies(
    *,
    intent_parser: Any,
    sql_branch: EvidenceBranch | None,
    rag_branch: EvidenceBranch | None,
    evidence_repository: EvidenceRepository,
    entity_bindings: Mapping[str, int],
    sql_identity_hmac_key: bytes | None,
    budgets: GraphBudgets,
    as_of: date,
    evidence_validator: EvidenceValidator | None = None,
    answer_generator: AnswerGenerator | None = None,
    claim_guard: ClaimHallucinationGuard | None = None,
    policy: Callable[[str], bool] | None = None,
    query_rewriter: Callable[[str, int], str] | None = None,
) -> NodeDependencies:
    return NodeDependencies(
        intent_parser=intent_parser,
        sql_branch=sql_branch,
        rag_branch=rag_branch,
        evidence_repository=evidence_repository,
        entity_bindings=dict(entity_bindings),
        sql_identity_hmac_key=sql_identity_hmac_key,
        budgets=budgets,
        as_of=as_of,
        evidence_validator=evidence_validator or EvidenceValidator(),
        answer_generator=answer_generator or DeterministicAnswerGenerator(),
        claim_guard=claim_guard or ClaimHallucinationGuard(),
        policy=policy or (lambda _question: True),
        query_rewriter=query_rewriter,
        branch_call_runner=BoundedCallRunner(budgets.max_outstanding_calls),
    )
