"""Normalize bounded aggregate SQL results into shared Evidence."""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from hashlib import sha256
import json

from trade_agent.db.sql_executor import SqlExecutionResult
from trade_agent.evidence.models import Evidence, EvidenceLocator, SqlProvenance


def build_sql_evidence(result: SqlExecutionResult) -> list[Evidence]:
    if type(result) is not SqlExecutionResult:
        raise TypeError("SQL evidence requires an exact SqlExecutionResult")
    content = json.dumps(
        {"metrics": result.metric_names, "rows": result.rows},
        default=_json_value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    evidence_id = "sql_" + sha256(
        f"{result.dataset_id}:{result.schema_fingerprint}:{result.query_id}:{result.result_hash}".encode("utf-8")
    ).hexdigest()
    provenance = SqlProvenance(
        query_id=result.query_id,
        normalized_sql=result.normalized_sql,
        bound_filter_names=result.bound_filter_names,
        schema_fingerprint=result.schema_fingerprint,
        dataset_id=result.dataset_id,
        is_synthetic=result.is_synthetic,
        effective_start_date=result.effective_start_date,
        effective_end_date=result.effective_end_date,
        aggregation_grain=result.aggregation_grain,
        time_grain=result.time_grain,
        execution_ms=result.execution_ms,
        row_count=result.row_count,
        result_hash=result.result_hash,
        estimated_scan_rows=result.estimated_scan_rows,
        max_execution_time_ms=result.max_execution_time_ms,
        client_timeout_ms=result.client_timeout_ms,
    )
    return [
        Evidence(
            evidence_id=evidence_id,
            fact_type=result.metric_names[0] if len(result.metric_names) == 1 else "trade_aggregate",
            source_type="sql",
            source_weight=1.0,
            content=content,
            locator=EvidenceLocator(
                query_id=result.query_id,
                raw_record_locators=result.raw_record_locators,
                raw_record_locators_truncated=result.raw_record_locators_truncated,
            ),
            valid_from=result.effective_start_date,
            valid_to=result.effective_end_date,
            confidence=1.0,
            is_synthetic=result.is_synthetic,
            sql_provenance=provenance,
        )
    ]
def _json_value(value: object) -> str:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"unsupported evidence value: {type(value).__name__}")
