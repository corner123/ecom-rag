from __future__ import annotations

from collections.abc import Mapping

import pytest

from trade_agent.db.contracts import QueryConstraints
from trade_agent.db.registry import SchemaDriftError, SchemaRegistry


class _Mappings:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def all(self) -> list[dict[str, object]]:
        return self._rows


class _Result:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def mappings(self) -> _Mappings:
        return _Mappings(self._rows)


class FakeConnection:
    def __init__(self, columns: list[dict[str, object]], keys: list[dict[str, object]] | None = None) -> None:
        self.columns = columns
        self.keys = keys or []

    def execute(self, statement: object, parameters: Mapping[str, object]) -> _Result:
        query = str(statement)
        if "information_schema.table_constraints" in query:
            return _Result(self.keys)
        if "information_schema.columns" in query:
            return _Result(self.columns)
        if "information_schema.key_column_usage" in query:
            return _Result([])
        raise AssertionError(f"unexpected discovery query: {query}")


@pytest.fixture
def reviewed_columns() -> list[dict[str, object]]:
    return [
        {
            "table_name": "countries",
            "column_name": "id",
            "data_type": "bigint",
            "column_type": "bigint unsigned",
            "is_nullable": "NO",
            "column_default": None,
            "ordinal_position": 1,
        },
    ]


def test_registry_rejects_unknown_business_field() -> None:
    registry = SchemaRegistry()

    result = registry.link(QueryConstraints(metrics=["profit_margin"]))

    assert result.ok is False
    assert result.error_code == "schema_not_registered"
    assert result.missing == ("profit_margin",)


def test_default_semantics_preserve_decimal_precision_and_scale() -> None:
    semantics = SchemaRegistry()._load_semantics()

    columns = semantics["tables"]["trade_records"]["columns"]
    assert columns["quantity"]["type"] == "decimal(18,3)"
    assert columns["trade_amount"]["type"] == "decimal(18,2)"


def test_registry_refresh_rejects_an_unreviewed_physical_column(
    reviewed_columns: list[dict[str, object]],
) -> None:
    registry = SchemaRegistry(semantic_path=None)
    registry._load_semantics = lambda: {
        "database": "foreign_trade_db",
        "tables": {"countries": {"columns": {"id": {"description": "Identifier", "type": "bigint unsigned", "nullable": False, "default": None}}}},
        "joins": [],
        "aliases": {},
        "dimensions": {},
        "filters": {},
        "aggregations": {},
        "sensitive_fields": [],
        "max_result_rows": 50,
    }

    first = registry.refresh(FakeConnection(reviewed_columns))

    assert first.tables == ("countries",)
    assert first.columns == ("countries.id",)
    assert first.max_result_rows == 50

    unexpected = [*reviewed_columns, {**reviewed_columns[0], "column_name": "unexpected_col", "ordinal_position": 2}]
    with pytest.raises(SchemaDriftError) as raised:
        registry.refresh(FakeConnection(unexpected))

    assert raised.value.observed_fingerprint != first.fingerprint


def test_registry_refresh_rejects_primary_key_contract_drift(
    reviewed_columns: list[dict[str, object]],
) -> None:
    registry = SchemaRegistry(semantic_path=None)
    registry._load_semantics = lambda: {
        "database": "foreign_trade_db",
        "tables": {
            "countries": {
                "columns": {"id": {"description": "Identifier", "type": "bigint unsigned", "nullable": False, "default": None}},
                "primary_key": ["id"],
                "unique_keys": [],
            }
        },
        "joins": [],
        "aliases": {},
        "dimensions": {},
        "filters": {},
        "aggregations": {},
        "sensitive_fields": [],
        "max_result_rows": 50,
    }

    with pytest.raises(SchemaDriftError) as raised:
        registry.refresh(FakeConnection(reviewed_columns, keys=[]))

    assert "key contract differs" in raised.value.differences
