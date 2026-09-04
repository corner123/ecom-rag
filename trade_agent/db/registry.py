"""Live INFORMATION_SCHEMA discovery guarded by a reviewed semantic contract."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml
from sqlalchemy import Connection, text

from trade_agent.db.contracts import QueryConstraints


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return _freeze_value(value)


def _canonical_default(value: object) -> str | None:
    if value is None:
        return None
    return str(value).lower().replace("()", "")


def _canonical_type(value: object) -> str:
    normalized = str(value).lower().replace("numeric", "decimal").replace(", ", ",")
    return "tinyint(1)" if normalized in {"bool", "boolean"} else normalized


@dataclass(frozen=True)
class _PhysicalColumn:
    table: str
    name: str
    type: str
    nullable: bool
    default: str | None
    position: int

    def as_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "name": self.name,
            "type": self.type,
            "nullable": self.nullable,
            "default": self.default,
            "position": self.position,
        }


@dataclass(frozen=True)
class _PhysicalJoin:
    name: str
    left: str
    right: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "left": self.left, "right": self.right}


@dataclass(frozen=True)
class RegistryJoin:
    """A reviewed foreign-key route, including its business-role traversal."""

    name: str
    left: str
    right: str
    roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class _PhysicalKey:
    table: str
    name: str
    kind: str
    columns: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {"table": self.table, "name": self.name, "kind": self.kind, "columns": self.columns}


@dataclass(frozen=True)
class _PhysicalSchema:
    columns: tuple[_PhysicalColumn, ...]
    joins: tuple[_PhysicalJoin, ...]
    keys: tuple[_PhysicalKey, ...]

    @property
    def tables(self) -> tuple[str, ...]:
        return tuple(sorted({column.table for column in self.columns}))

    @property
    def fingerprint(self) -> str:
        payload = {
            "columns": [column.as_dict() for column in self.columns],
            "joins": [join.as_dict() for join in self.joins],
            "keys": [key.as_dict() for key in self.keys],
        }
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RegistrySnapshot:
    fingerprint: str
    tables: tuple[str, ...]
    columns: tuple[str, ...]
    joins: tuple[RegistryJoin, ...]
    aliases: Mapping[str, str]
    aggregations: Mapping[str, Mapping[str, Any]]
    dimensions: Mapping[str, str]
    filters: Mapping[str, str]
    sensitive_fields: tuple[str, ...]
    max_result_rows: int


@dataclass(frozen=True)
class SchemaLinkResult:
    ok: bool
    error_code: str | None = None
    missing: tuple[str, ...] = ()
    snapshot: RegistrySnapshot | None = None


class SchemaDriftError(RuntimeError):
    """The live schema differs from reviewed semantics and cannot be planned against."""

    def __init__(self, observed_fingerprint: str, differences: tuple[str, ...]) -> None:
        self.observed_fingerprint = observed_fingerprint
        self.differences = differences
        super().__init__("live schema does not match the reviewed contract")


class SchemaRegistry:
    def __init__(self, semantic_path: Path | None = None) -> None:
        self.semantic_path = semantic_path
        self._snapshot: RegistrySnapshot | None = None

    def refresh(self, connection: Connection) -> RegistrySnapshot:
        semantics = self._load_semantics()
        physical = self._inspect_information_schema(connection, str(semantics["database"]))
        differences = self._differences(physical, semantics)
        if differences:
            self._snapshot = None
            raise SchemaDriftError(physical.fingerprint, differences)
        snapshot = RegistrySnapshot(
            fingerprint=physical.fingerprint,
            tables=physical.tables,
            columns=tuple(f"{column.table}.{column.name}" for column in physical.columns),
            joins=tuple(
                RegistryJoin(
                    name=str(join["name"]),
                    left=str(join["left"]),
                    right=str(join["right"]),
                    roles=tuple(str(role) for role in join.get("roles", [])),
                )
                for join in semantics["joins"]
            ),
            aliases=_freeze_mapping(semantics["aliases"]),
            aggregations=MappingProxyType(
                {name: _freeze_mapping(definition) for name, definition in semantics["aggregations"].items()}
            ),
            dimensions=_freeze_mapping(semantics["dimensions"]),
            filters=_freeze_mapping(semantics["filters"]),
            sensitive_fields=tuple(semantics["sensitive_fields"]),
            max_result_rows=int(semantics["max_result_rows"]),
        )
        self._snapshot = snapshot
        return snapshot

    def link(self, constraints: QueryConstraints) -> SchemaLinkResult:
        snapshot = self._snapshot
        if snapshot is None:
            semantics = self._load_semantics()
            allowed = self._allowed_fields(
                aggregations=semantics["aggregations"],
                dimensions=semantics["dimensions"],
                filters=semantics["filters"],
                aliases=semantics["aliases"],
            )
        else:
            allowed = self._allowed_fields(
                aggregations=snapshot.aggregations,
                dimensions=snapshot.dimensions,
                filters=snapshot.filters,
                aliases=snapshot.aliases,
            )
        missing = tuple(
            field
            for category, fields in (
                ("metrics", constraints.metrics),
                ("dimensions", constraints.dimensions),
                ("filters", constraints.filters),
            )
            for field in fields
            if field not in allowed[category]
        )
        if missing:
            return SchemaLinkResult(ok=False, error_code="schema_not_registered", missing=missing)
        if snapshot is None:
            return SchemaLinkResult(ok=False, error_code="registry_not_refreshed")
        return SchemaLinkResult(ok=True, snapshot=snapshot)

    @staticmethod
    def _allowed_fields(
        *,
        aggregations: Mapping[str, Any],
        dimensions: Mapping[str, Any],
        filters: Mapping[str, Any],
        aliases: Mapping[str, Any],
    ) -> dict[str, set[str]]:
        return {
            "metrics": set(aggregations),
            "dimensions": set(dimensions) | set(aliases),
            "filters": set(filters) | set(aliases),
        }

    def _load_semantics(self) -> Mapping[str, Any]:
        path = self.semantic_path
        if path is None:
            path = Path(str(files("trade_agent.config").joinpath("schema_registry.yaml")))
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("schema registry semantics must be a mapping")
        required = {"database", "tables", "joins", "aliases", "dimensions", "filters", "aggregations", "sensitive_fields", "max_result_rows"}
        if set(loaded) != required:
            raise ValueError("schema registry semantics have unexpected keys")
        if not isinstance(loaded["tables"], dict) or not loaded["tables"]:
            raise ValueError("schema registry must register tables")
        if not isinstance(loaded["max_result_rows"], int) or loaded["max_result_rows"] <= 0:
            raise ValueError("max_result_rows must be a positive integer")
        return loaded

    @staticmethod
    def _inspect_information_schema(connection: Connection, database: str) -> _PhysicalSchema:
        columns = tuple(
            _PhysicalColumn(
                table=str(row["table_name"]),
                name=str(row["column_name"]),
                type=_canonical_type(row["column_type"]),
                nullable=str(row["is_nullable"]).upper() == "YES",
                default=_canonical_default(row["column_default"]),
                position=int(row["ordinal_position"]),
            )
            for row in connection.execute(
                text(
                    "SELECT table_name AS table_name, column_name AS column_name, "
                    "column_type AS column_type, is_nullable AS is_nullable, "
                    "column_default AS column_default, ordinal_position AS ordinal_position "
                    "FROM information_schema.columns "
                    "WHERE table_schema = :database ORDER BY table_name, ordinal_position"
                ),
                {"database": database},
            ).mappings().all()
        )
        joins = tuple(
            _PhysicalJoin(
                name=str(row["constraint_name"]),
                left=f"{row['left_table']}.{row['left_column']}",
                right=f"{row['right_table']}.{row['right_column']}",
            )
            for row in connection.execute(
                text(
                    "SELECT kcu.constraint_name AS constraint_name, kcu.table_name AS left_table, "
                    "kcu.column_name AS left_column, kcu.referenced_table_name AS right_table, "
                    "kcu.referenced_column_name AS right_column "
                    "FROM information_schema.key_column_usage AS kcu "
                    "WHERE kcu.table_schema = :database AND kcu.referenced_table_name IS NOT NULL "
                    "ORDER BY kcu.table_name, kcu.constraint_name, kcu.ordinal_position"
                ),
                {"database": database},
            ).mappings().all()
        )
        key_rows = connection.execute(
            text(
                "SELECT tc.table_name AS table_name, tc.constraint_name AS constraint_name, "
                "tc.constraint_type AS constraint_type, kcu.column_name AS column_name, "
                "kcu.ordinal_position AS ordinal_position "
                "FROM information_schema.table_constraints AS tc "
                "JOIN information_schema.key_column_usage AS kcu "
                "ON kcu.constraint_schema = tc.constraint_schema "
                "AND kcu.table_name = tc.table_name AND kcu.constraint_name = tc.constraint_name "
                "WHERE tc.constraint_schema = :database "
                "AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE') "
                "ORDER BY tc.table_name, tc.constraint_name, kcu.ordinal_position"
            ),
            {"database": database},
        ).mappings().all()
        grouped_keys: dict[tuple[str, str, str], list[str]] = {}
        for row in key_rows:
            grouped_keys.setdefault(
                (str(row["table_name"]), str(row["constraint_name"]), str(row["constraint_type"])), []
            ).append(str(row["column_name"]))
        keys = tuple(
            _PhysicalKey(table=table, name=name, kind=kind, columns=tuple(key_columns))
            for (table, name, kind), key_columns in grouped_keys.items()
        )
        return _PhysicalSchema(columns=columns, joins=joins, keys=keys)

    @staticmethod
    def _differences(physical: _PhysicalSchema, semantics: Mapping[str, Any]) -> tuple[str, ...]:
        expected_columns: dict[tuple[str, str], Mapping[str, Any]] = {
            (table, column): definition
            for table, table_definition in semantics["tables"].items()
            for column, definition in table_definition["columns"].items()
        }
        actual_columns = {(column.table, column.name): column for column in physical.columns}
        differences: list[str] = []
        if set(actual_columns) != set(expected_columns):
            differences.append("table or column set differs")
        for key in sorted(set(actual_columns) & set(expected_columns)):
            actual, expected = actual_columns[key], expected_columns[key]
            if _canonical_type(expected["type"]) != actual.type:
                differences.append(f"column type differs: {actual.table}.{actual.name}")
            if bool(expected["nullable"]) != actual.nullable:
                differences.append(f"column nullability differs: {actual.table}.{actual.name}")
            if _canonical_default(expected.get("default")) != actual.default:
                differences.append(f"column default differs: {actual.table}.{actual.name}")
        expected_joins = {
            (str(join["name"]), str(join["left"]), str(join["right"])) for join in semantics["joins"]
        }
        actual_joins = {(join.name, join.left, join.right) for join in physical.joins}
        if expected_joins != actual_joins:
            differences.append("foreign-key join contract differs")
        expected_keys = {
            (table, "PRIMARY", "PRIMARY KEY", tuple(definition["primary_key"]))
            for table, definition in semantics["tables"].items()
            if "primary_key" in definition
        }
        expected_keys.update(
            (table, str(key["name"]), "UNIQUE", tuple(key["columns"]))
            for table, definition in semantics["tables"].items()
            for key in definition.get("unique_keys", [])
        )
        actual_keys = {(key.table, key.name, key.kind, key.columns) for key in physical.keys}
        if expected_keys != actual_keys:
            differences.append("key contract differs")
        return tuple(differences)
