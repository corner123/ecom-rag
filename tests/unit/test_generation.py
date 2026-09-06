from __future__ import annotations

from datetime import date

import pytest

from trade_agent.agents.intent import IntentParser
from trade_agent.db.sql_executor import ReadOnlySqlExecutor, SqlExecutionResult
from trade_agent.evidence.models import RawRecordLocator
from trade_agent.evidence.sql import build_sql_evidence
from trade_agent.evidence.validator import (
    BranchExecutionReport,
    EvidenceValidator,
    ValidationContext,
)
from trade_agent.generation.deterministic import DeterministicAnswerGenerator


AS_OF = date(2026, 9, 4)


def _evidence(amount: str = "12.30"):
    rows = ({"importer_company": "Acme", "currency": "USD", "trade_amount": amount},)
    result = SqlExecutionResult(
        query_id="sqlq_" + "a" * 64,
        normalized_sql=(
            "SELECT importer.company_name AS importer_company, trade_amount, "
            "import_country.country_code, hs.hs_code FROM trade_records AS tr "
            "JOIN companies AS importer ON tr.importer_id = importer.id "
            "JOIN countries AS import_country ON tr.import_country_id = import_country.id "
            "JOIN hs_codes AS hs ON tr.hs_code_id = hs.id "
            "JOIN data_sources AS data_scope ON tr.source_id = data_scope.id "
            "WHERE data_scope.is_synthetic = :policy_is_synthetic "
            "AND tr.trade_date BETWEEN :policy_start_date AND :policy_end_date "
            "AND import_country.country_code = :filter_0_0 "
            "AND hs.hs_code = :filter_1_0 "
            "AND tr.trade_date BETWEEN :filter_2_start AND :filter_2_end LIMIT 50"
        ),
        bound_filter_names=(
            "filter_0_0", "filter_1_0", "filter_2_end", "filter_2_start",
            "policy_end_date", "policy_is_synthetic", "policy_start_date",
        ),
        schema_fingerprint="b" * 64,
        dataset_id="trade-seed-v1",
        is_synthetic=True,
        effective_start_date=date(2026, 3, 4),
        effective_end_date=AS_OF,
        aggregation_grain=("currency", "importer_company"),
        time_grain="total",
        metric_names=("trade_amount",),
        rows=rows,
        row_count=1,
        result_hash=ReadOnlySqlExecutor._hash_rows(rows),
        raw_record_locators=(RawRecordLocator(source_id=7, raw_record_id="ROW-9"),),
        raw_record_locators_truncated=False,
        estimated_scan_rows=1,
        execution_ms=1.0,
        max_execution_time_ms=2_000,
        client_timeout_ms=3_000,
    )
    return build_sql_evidence(result)[0]


@pytest.fixture
def validated_context():
    evidence = _evidence()
    intent = IntentParser(as_of=AS_OF).parse("最近半年美国采购 HS850440 金额最高的 10 家公司")
    validation = EvidenceValidator().validate(
        intent,
        (evidence,),
        (),
        AS_OF,
        context=ValidationContext(branch_reports=(
            BranchExecutionReport(branch="sql", attempted=True, completed=True, zero_hits=False),
        )),
    )
    assert validation.can_answer
    return intent, evidence, validation


def test_deterministic_generation_uses_only_validator_retained_evidence(validated_context) -> None:
    intent, evidence, validation = validated_context
    excluded = _evidence("999.99")

    draft = DeterministicAnswerGenerator().generate(intent, (evidence, excluded), validation)

    assert draft.claims
    assert {evidence_id for claim in draft.claims for evidence_id in claim.evidence_ids} == {evidence.evidence_id}
    assert excluded.evidence_id not in draft.answer
    assert "12.30" in draft.answer


def test_deterministic_generation_refuses_nonanswer_validation(validated_context) -> None:
    intent, evidence, validation = validated_context
    refusal = EvidenceValidator().validate(
        intent,
        (),
        (),
        AS_OF,
        context=ValidationContext(branch_reports=(
            BranchExecutionReport(branch="sql", attempted=True, completed=True, zero_hits=True),
        )),
    )

    draft = DeterministicAnswerGenerator().generate(intent, (evidence,), refusal)

    assert draft.answer is None
    assert draft.refusal_reason == "evidence_insufficient"
    assert draft.claims == ()
