"""Strict contracts and shared support helpers for answer generation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from trade_agent.agents.intent import QueryIntent
from trade_agent.evidence.models import Claim, Evidence
from trade_agent.evidence.validator import ValidationOutcome


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def model_copy(self, *, update: Mapping[str, object] | None = None, deep: bool = False) -> Self:
        if update is None:
            return super().model_copy(deep=deep)
        values = self.model_dump(mode="python")
        if deep:
            values = deepcopy(values)
        values.update(update)
        return type(self).model_validate(values)


class DraftAnswer(_Contract):
    """An untrusted provider draft whose claims still require a guard decision."""

    answer: StrictStr | None
    claims: tuple[Claim, ...]
    refusal_reason: StrictStr | None
    core_claim_ids: tuple[StrictStr, ...] = ()

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if (self.answer is None) == (self.refusal_reason is None):
            raise ValueError("exactly one of answer or refusal_reason is required")
        if self.answer is not None and not self.answer.strip():
            raise ValueError("answer must not be blank")
        if self.refusal_reason is not None and not self.refusal_reason.strip():
            raise ValueError("refusal_reason must not be blank")
        claim_ids = tuple(item.claim_id for item in self.claims)
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("claim IDs must be unique")
        if self.core_claim_ids != tuple(sorted(set(self.core_claim_ids))):
            raise ValueError("core claim IDs must be sorted and unique")
        if not set(self.core_claim_ids).issubset(claim_ids):
            raise ValueError("core claim IDs must belong to draft claims")
        if self.refusal_reason is not None and self.claims:
            raise ValueError("refusal drafts must not contain claims")
        return self


class AnswerGenerator(ABC):
    """Provider boundary: only validated Evidence can enter generation."""

    @abstractmethod
    def generate(
        self,
        intent: QueryIntent,
        evidence: Sequence[Evidence],
        validation: ValidationOutcome,
    ) -> DraftAnswer:
        raise NotImplementedError


def generation_evidence(
    intent: QueryIntent,
    evidence: Sequence[Evidence],
    validation: ValidationOutcome,
) -> tuple[Evidence, ...]:
    """Return only revalidated Evidence that the validator retained."""
    if type(intent) is not QueryIntent:
        raise TypeError("intent must be an exact QueryIntent")
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise TypeError("evidence must be a sequence")
    if type(validation) is not ValidationOutcome:
        raise TypeError("validation must be an exact ValidationOutcome")
    checked_validation = ValidationOutcome.model_validate(validation.model_dump(mode="python"))
    if not checked_validation.can_answer:
        return ()
    checked: dict[str, Evidence] = {}
    for item in evidence:
        if type(item) is not Evidence:
            raise TypeError("generation accepts exact Evidence contracts")
        verified = Evidence.model_validate(item.model_dump(mode="python"))
        if verified.evidence_id in checked:
            raise ValueError("generation Evidence IDs must be unique")
        checked[verified.evidence_id] = verified
    expected = set(checked_validation.eligible_evidence_ids)
    if not expected or not expected.issubset(checked):
        raise ValueError("validation retained Evidence is unavailable to generation")
    return tuple(checked[evidence_id] for evidence_id in sorted(expected))


def claim_id_for(*, evidence_id: str, payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        {"evidence_id": evidence_id, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "claim_" + sha256(canonical.encode("utf-8")).hexdigest()


def render_sql_claim(evidence: Evidence, row: Mapping[str, object], metric: str) -> str:
    """Canonical public text for one atomic SQL result row."""
    value = _render_value(row[metric])
    currency = _row_dimension(row, "currency")
    unit = _row_dimension(row, "unit")
    entity = _row_entity(row)
    period = f"{_date_text(evidence.valid_from)} to {_date_text(evidence.valid_to)}"
    grain = ",".join(evidence.aggregation_grain) or "none"
    dimensions = []
    for key in sorted(row):
        if key != metric and row[key] is not None:
            dimensions.append(f"{key}={_render_value(row[key])}")
    suffix = "; ".join(dimensions)
    return (
        f"{metric}={value}" + (f" {currency}" if currency else "") + (f" {unit}" if unit else "")
        + f"; entity={entity}; period={period}; grain={grain}"
        + (f"; dimensions={suffix}" if suffix else "")
    )


def sql_claim_rows(claim: Claim, evidence: Evidence) -> tuple[Mapping[str, object], ...]:
    """Rows that exactly support a claim's structured SQL fields and text."""
    if evidence.locator.branch != "sql" or claim.fact_type is None:
        return ()
    try:
        payload = json.loads(evidence.content)
    except (TypeError, json.JSONDecodeError):
        return ()
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return ()
    supported: list[Mapping[str, object]] = []
    for row in rows:
        if not isinstance(row, dict) or claim.fact_type not in row:
            continue
        if claim.value is None or not _values_equal(claim.value, row[claim.fact_type]):
            continue
        if claim.currency != _row_dimension(row, "currency") or claim.unit != _row_dimension(row, "unit"):
            continue
        if claim.text != render_sql_claim(evidence, row, claim.fact_type):
            continue
        supported.append(row)
    return tuple(supported)


def _row_dimension(row: Mapping[str, object], key: str) -> str | None:
    value = row.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _row_entity(row: Mapping[str, object]) -> str:
    for key in ("importer_company", "exporter_company", "company_name", "entity_id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return "unattributed"


def _render_value(value: object) -> str:
    if isinstance(value, Decimal):
        return format(value, "f")
    return str(value)


def _values_equal(claim_value: str, row_value: object) -> bool:
    rendered = _render_value(row_value)
    if claim_value == rendered:
        return True
    try:
        return Decimal(claim_value) == Decimal(rendered)
    except (InvalidOperation, ValueError):
        return False


def _date_text(value: date | datetime | None) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    return value.isoformat() if value is not None else "unbounded"
