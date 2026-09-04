from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text

from trade_agent.db.contracts import QueryConstraints
from trade_agent.db.registry import SchemaDriftError, SchemaRegistry
from trade_agent.db.session import database_url_from_environment


pytestmark = pytest.mark.integration


def test_live_query_role_refreshes_the_reviewed_seven_table_schema() -> None:
    engine = create_engine(database_url_from_environment(role="query"))
    try:
        with engine.connect() as connection:
            registry = SchemaRegistry()
            snapshot = registry.refresh(connection)

        assert snapshot.tables == (
            "companies",
            "company_products",
            "countries",
            "data_sources",
            "hs_codes",
            "products",
            "trade_records",
        )
        assert len(snapshot.joins) == 11
        assert snapshot.aggregations["trade_amount"]["currency_column"] == "trade_records.currency"
        assert snapshot.aggregations["quantity"]["unit_column"] == "trade_records.unit"
        assert registry.link(QueryConstraints(metrics=["trade_amount"], dimensions=["importer_company"], filters=["import_country"])).ok
    finally:
        engine.dispose()


def test_live_unreviewed_column_fails_closed_and_is_removed() -> None:
    query_engine = create_engine(database_url_from_environment(role="query"))
    migration_engine = create_engine(database_url_from_environment(role="migration"))
    column_name = f"registry_drift_{uuid4().hex}"
    added = False
    try:
        with query_engine.connect() as query_connection:
            registry = SchemaRegistry()
            baseline = registry.refresh(query_connection)
        with migration_engine.connect() as migration_connection:
            migration_connection.execute(text(f"ALTER TABLE companies ADD COLUMN {column_name} VARCHAR(8) NULL"))
            added = True
        with query_engine.connect() as query_connection:
            with pytest.raises(SchemaDriftError) as raised:
                registry.refresh(query_connection)
        assert raised.value.observed_fingerprint != baseline.fingerprint
        assert "table or column set differs" in raised.value.differences
    finally:
        if added:
            with migration_engine.connect() as migration_connection:
                migration_connection.execute(text(f"ALTER TABLE companies DROP COLUMN {column_name}"))
        query_engine.dispose()
        migration_engine.dispose()
