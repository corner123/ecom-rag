from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError, OperationalError

from trade_agent.db.migrate import migrate_database
from trade_agent.db.models import Base
from trade_agent.db.seed import generate_trade_seed, seed_database
from trade_agent.db.session import database_url_from_environment

pytestmark = pytest.mark.integration


def _migration_url() -> str:
    return database_url_from_environment(role="migration")


def test_ddl_matches_sqlalchemy_metadata_and_seed_contract():
    engine = create_engine(_migration_url())
    migrate_database(engine)
    summary = seed_database(engine, generate_trade_seed())

    inspector = inspect(engine)
    assert set(inspector.get_table_names()) == {
        "countries", "companies", "hs_codes", "products", "data_sources", "company_products", "trade_records"
    }
    assert summary.trade_records >= 800
    assert summary.months == 18

    with engine.connect() as connection:
        for table in Base.metadata.sorted_tables:
            actual_columns = {column["name"] for column in inspector.get_columns(table.name)}
            assert actual_columns == {column.name for column in table.columns}
            index_rows = connection.execute(
            text(
                "SELECT index_name, non_unique, seq_in_index, column_name "
                "FROM information_schema.statistics "
                "WHERE table_schema = DATABASE() AND table_name = :table AND index_name <> 'PRIMARY' "
                "ORDER BY index_name, seq_in_index"
            ),
            {"table": table.name},
            ).all()
            actual_indexes: dict[str, tuple[bool, tuple[str, ...]]] = {}
            for row in index_rows:
                index_name, non_unique, _sequence, column_name = row
                current = actual_indexes.get(index_name, (not bool(non_unique), ()))
                actual_indexes[index_name] = (current[0], (*current[1], column_name))
            expected_indexes = {
                index.name: (bool(index.unique), tuple(column.name for column in index.columns))
                for index in table.indexes
                if index.name
            }
            expected_indexes.update(
                {
                    constraint.name: (True, tuple(column.name for column in constraint.columns))
                    for constraint in table.constraints
                    if constraint.name and constraint.__class__.__name__ == "UniqueConstraint"
                }
            )
            assert actual_indexes == expected_indexes
            actual_fks = {(fk["constrained_columns"][0], fk["referred_table"]) for fk in inspector.get_foreign_keys(table.name)}
            expected_fks = {(fk.parent.name, fk.column.table.name) for fk in table.foreign_keys}
            assert expected_fks == actual_fks

    with engine.begin() as connection:
        record = connection.execute(text("SELECT source_id, raw_record_id FROM trade_records LIMIT 1")).one()
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO trade_records "
                    "(raw_record_id, source_id, importer_id, exporter_id, product_id, hs_code_id, "
                    "import_country_id, export_country_id, trade_date, quantity, unit, trade_amount, currency) "
                    "SELECT raw_record_id, source_id, importer_id, exporter_id, product_id, hs_code_id, "
                    "import_country_id, export_country_id, trade_date, quantity, unit, trade_amount, currency "
                    "FROM trade_records WHERE source_id = :source AND raw_record_id = :raw"
                ),
                {"raw": record.raw_record_id, "source": record.source_id},
            )


@pytest.mark.parametrize("statement", ["INSERT INTO countries (country_code, country_name, region) VALUES ('ZZ', 'No Write', 'test')", "UPDATE countries SET region = 'test' WHERE 1 = 0", "DELETE FROM countries WHERE 1 = 0"])
def test_query_user_cannot_mutate_database(statement: str):
    url = database_url_from_environment(role="query")
    engine = create_engine(url)
    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM countries")).scalar_one() >= 8
        with pytest.raises(OperationalError):
            connection.execute(text(statement))
