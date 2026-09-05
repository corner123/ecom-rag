from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DatabaseError

from trade_agent.agents.intent import IntentParser
from trade_agent.db.registry import SchemaRegistry
from trade_agent.db.seed import generate_trade_seed
from trade_agent.db.session import database_url_from_environment
from trade_agent.db.sql_executor import ReadOnlySqlExecutor, SqlScanBudgetExceeded
from trade_agent.db.sql_planner import SqlPlanner
from trade_agent.db.sql_renderer import SqlDataScope, SqlRenderer
from trade_agent.db.sql_validator import SqlValidator
from trade_agent.evidence.sql import build_sql_evidence


pytestmark = pytest.mark.integration


@pytest.fixture
def identity_hmac_key() -> bytes:
    return bytes(range(32))


def _scope() -> SqlDataScope:
    return SqlDataScope(
        dataset_id="synthetic-demo-v1",
        synthetic=True,
        start_date=date(2025, 3, 1),
        end_date=date(2026, 8, 31),
    )


def test_approved_aggregate_returns_independently_checked_decimals_and_replayable_evidence(
    identity_hmac_key: bytes,
) -> None:
    engine = create_engine(database_url_from_environment(role="query"), pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            registry = SchemaRegistry().refresh(connection)
            intent = IntentParser(as_of=date(2026, 9, 4)).parse("最近18个月美国采购金额最高的 10 家公司")
            plan = SqlPlanner().plan(intent, registry)
            rendered = SqlRenderer(scope=_scope()).render(plan)
            validated = SqlValidator(scope=_scope()).validate(rendered, registry)
            result = ReadOnlySqlExecutor(
                connection,
                identity_hmac_key=identity_hmac_key,
                max_execution_time_ms=2_000,
                max_scan_rows=5_000,
            ).execute(validated)
            assert connection.scalar(text("SELECT @@session.max_execution_time")) == 0
            assert connection.connection.driver_connection._read_timeout is None

        seed = generate_trade_seed()
        importer_names = {company.id: company.company_name for company in seed.companies}
        expected: dict[tuple[str, str], Decimal] = {}
        for record in seed.trade_records:
            if record.import_country_id != 2 or not (date(2025, 3, 4) <= record.trade_date <= date(2026, 8, 31)):
                continue
            key = (importer_names[record.importer_id], record.currency)
            expected[key] = expected.get(key, Decimal("0")) + record.trade_amount
        expected_rows = sorted(expected.items(), key=lambda item: item[1], reverse=True)[:10]

        assert [(row["importer_company"], row["currency"], row["trade_amount"]) for row in result.rows] == [
            (company, currency, total) for (company, currency), total in expected_rows
        ]
        assert all(isinstance(row["trade_amount"], Decimal) for row in result.rows)
        assert result.raw_record_locators
        assert result.row_count <= 10
        assert result.max_execution_time_ms == 2_000
        assert result.client_timeout_ms == 3_000

        evidence = build_sql_evidence(result)
        assert len(evidence) == 1
        assert evidence[0].locator.raw_record_locators
        assert evidence[0].locator.scope == "bounded_predicate_population"
        assert all(locator.source_id > 0 and locator.raw_record_id for locator in evidence[0].locator.raw_record_locators)
        assert evidence[0].sql_provenance.result_hash == result.result_hash
        assert evidence[0].sql_provenance.schema_fingerprint == registry.fingerprint
        assert set(evidence[0].sql_provenance.bound_filter_names) == set(validated.bound_filter_names)
        assert evidence[0].valid_from == date(2025, 3, 4)
        assert evidence[0].valid_to == _scope().end_date
        assert evidence[0].is_synthetic is True
        provenance_dump = evidence[0].sql_provenance.model_dump()
        assert "params" not in provenance_dump
        assert "parameter_digest" not in provenance_dump
        assert "raw_record_id" not in evidence[0].content
    finally:
        engine.dispose()


def test_monthly_quantity_uses_decimal_and_matches_independent_seed_calculation(
    identity_hmac_key: bytes,
) -> None:
    engine = create_engine(database_url_from_environment(role="query"), pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            registry = SchemaRegistry().refresh(connection)
            plan = SqlPlanner().plan(
                IntentParser(as_of=date(2026, 9, 4)).parse("中国出口 HS850440 的月度数量"), registry
            )
            rendered = SqlRenderer(scope=_scope()).render(plan)
            validated = SqlValidator(scope=_scope()).validate(rendered, registry)
            result = ReadOnlySqlExecutor(
                connection,
                identity_hmac_key=identity_hmac_key,
                max_execution_time_ms=2_000,
                max_scan_rows=5_000,
            ).execute(validated)

        expected: dict[tuple[str, str], Decimal] = {}
        for record in generate_trade_seed().trade_records:
            if record.export_country_id != 1 or record.hs_code_id != 11:
                continue
            key = (record.trade_date.strftime("%Y-%m"), record.unit)
            expected[key] = expected.get(key, Decimal("0")) + record.quantity
        expected_rows = sorted(expected.items(), key=lambda item: item[1], reverse=True)[:50]
        assert [(row["trade_date"], row["unit"], row["quantity"]) for row in result.rows] == [
            (month, unit, total) for (month, unit), total in expected_rows
        ]
        assert all(isinstance(row["quantity"], Decimal) for row in result.rows)
    finally:
        engine.dispose()


def test_query_user_cannot_write() -> None:
    engine = create_engine(database_url_from_environment(role="query"))
    try:
        with engine.connect() as query_connection:
            with pytest.raises(DatabaseError):
                query_connection.execute(text("UPDATE companies SET industry='x'"))
    finally:
        engine.dispose()


def test_explain_scan_budget_fails_closed(identity_hmac_key: bytes) -> None:
    engine = create_engine(database_url_from_environment(role="query"))
    try:
        with engine.connect() as connection:
            connection.execute(text("SET SESSION max_execution_time = 37"))
            original_transaction_read_only = connection.scalar(text("SELECT @@SESSION.transaction_read_only"))
            registry = SchemaRegistry().refresh(connection)
            plan = SqlPlanner().plan(
                IntentParser(as_of=date(2026, 9, 4)).parse("最近18个月美国采购金额最高的 10 家公司"), registry
            )
            validated = SqlValidator(scope=_scope()).validate(SqlRenderer(scope=_scope()).render(plan), registry)
            with pytest.raises(SqlScanBudgetExceeded):
                ReadOnlySqlExecutor(
                    connection,
                    identity_hmac_key=identity_hmac_key,
                    max_execution_time_ms=2_000,
                    max_scan_rows=1,
                ).execute(validated)
            assert connection.scalar(text("SELECT @@SESSION.max_execution_time")) == 37
            assert connection.scalar(text("SELECT @@SESSION.transaction_read_only")) == original_transaction_read_only
    finally:
        engine.dispose()


def test_executor_rejects_a_disabled_client_timeout(identity_hmac_key: bytes) -> None:
    engine = create_engine(database_url_from_environment(role="query"))
    try:
        with engine.connect() as connection:
            with pytest.raises(ValueError, match="client_timeout_ms"):
                ReadOnlySqlExecutor(
                    connection, identity_hmac_key=identity_hmac_key, client_timeout_ms=0
                )
    finally:
        engine.dispose()


def test_query_identity_uses_private_keyed_parameters_even_for_empty_results(
    identity_hmac_key: bytes,
) -> None:
    engine = create_engine(database_url_from_environment(role="query"), pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            registry = SchemaRegistry().refresh(connection)
            parser = IntentParser(as_of=date(2026, 9, 4))
            planner = SqlPlanner()
            renderer = SqlRenderer(scope=_scope())
            first = renderer.render(
                planner.plan(
                    parser.parse("美国采购 2025-03-02 到 2025-03-03 HS850440 金额最高的 10 家公司"),
                    registry,
                )
            )
            second = renderer.render(
                planner.plan(
                    parser.parse("中国采购 2025-03-02 到 2025-03-03 HS850440 金额最高的 10 家公司"),
                    registry,
                )
            )
            validator = SqlValidator(scope=_scope())
            executor = ReadOnlySqlExecutor(
                connection,
                identity_hmac_key=identity_hmac_key,
                max_execution_time_ms=2_000,
                max_scan_rows=5_000,
            )
            first_result = executor.execute(validator.validate(first, registry))
            second_result = executor.execute(validator.validate(second, registry))

        assert first_result.rows == second_result.rows == ()
        assert first_result.result_hash == second_result.result_hash
        assert first_result.query_id != second_result.query_id
        assert build_sql_evidence(first_result)[0].evidence_id != build_sql_evidence(second_result)[0].evidence_id
        assert not hasattr(first_result, "parameter_digest")
        assert "US" not in first_result.query_id
        assert "CN" not in second_result.query_id
    finally:
        engine.dispose()
