from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.dialects import mysql
from sqlalchemy.exc import IntegrityError, OperationalError

from trade_agent.db.migrate import migrate_database
from trade_agent.db.models import Base
from trade_agent.db.seed import generate_trade_seed, seed_database
from trade_agent.db.session import database_url_from_environment

pytestmark = pytest.mark.integration


def _migration_url() -> str:
    return database_url_from_environment(role="migration")


def _ddl_columns() -> dict[str, dict[str, tuple[str, bool, str | None]]]:
    """Parse the authoritative migration, independently of ORM metadata."""
    result: dict[str, dict[str, tuple[str, bool, str | None]]] = {}
    table = None
    for line in Path("db/migrations/001_schema.sql").read_text().splitlines():
        stripped = line.strip().rstrip(",")
        if stripped.startswith("CREATE TABLE"):
            table = stripped.split()[5]
            result[table] = {}
        elif table and stripped and not stripped.startswith(("PRIMARY", "CONSTRAINT", "INDEX", ")")):
            parts = stripped.split()
            name, type_ = parts[0], parts[1].lower()
            if len(parts) > 2 and parts[2].upper() == "UNSIGNED":
                type_ += " unsigned"
            nullable = "NOT NULL" not in stripped
            default = "current_timestamp" if "DEFAULT CURRENT_TIMESTAMP" in stripped else None
            result[table][name] = ("tinyint(1)" if type_ == "boolean" else type_, nullable, default)
    return result


def _normal_type(value: str) -> str:
    return value.lower().replace(" unsigned", " unsigned")


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

    ddl_columns = _ddl_columns()
    with engine.connect() as connection:
        for table in Base.metadata.sorted_tables:
            live_columns = connection.execute(text("SELECT column_name, column_type, is_nullable, column_default FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name=:table ORDER BY ordinal_position"), {"table": table.name}).all()
            actual_columns = {name: (_normal_type(column_type), nullable == "YES", None if default is None else str(default).lower().replace("()", "")) for name, column_type, nullable, default in live_columns}
            assert actual_columns == ddl_columns[table.name]
            orm_columns = {column.name: (_normal_type(column.type.compile(dialect=mysql.dialect())), column.nullable, "current_timestamp" if column.server_default is not None else None) for column in table.columns}
            assert orm_columns == ddl_columns[table.name]
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
            fk_rows = connection.execute(text("SELECT k.column_name, k.referenced_table_name, k.referenced_column_name, r.update_rule, r.delete_rule FROM information_schema.key_column_usage k JOIN information_schema.referential_constraints r ON r.constraint_schema=k.constraint_schema AND r.constraint_name=k.constraint_name AND r.table_name=k.table_name WHERE k.table_schema=DATABASE() AND k.table_name=:table AND k.referenced_table_name IS NOT NULL"), {"table": table.name}).all()
            actual_fks = {(row[0], row[1], row[2], row[3], row[4]) for row in fk_rows}
            expected_fks = {(fk.parent.name, fk.column.table.name, fk.column.name, "RESTRICT", "RESTRICT") for fk in table.foreign_keys}
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


def test_seed_rejects_invalid_fk_and_conflicting_existing_content_atomically():
    engine = create_engine(_migration_url())
    migrate_database(engine)
    bundle = generate_trade_seed()
    with engine.connect() as connection:
        before_countries = connection.execute(text("SELECT COUNT(*) FROM countries")).scalar_one()
    invalid = bundle.model_copy(update={"companies": [bundle.companies[0].model_copy(update={"country_id": 999999}), *bundle.companies[1:]]})
    with pytest.raises(IntegrityError):
        seed_database(engine, invalid)
    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM countries")).scalar_one() == before_countries

    seed_database(engine, bundle)
    conflicting = bundle.model_copy(update={"countries": [bundle.countries[0].model_copy(update={"country_name": "Conflict"}), *bundle.countries[1:]]})
    with pytest.raises(IntegrityError):
        seed_database(engine, conflicting)
