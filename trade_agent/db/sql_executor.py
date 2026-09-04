"""Budgeted execution of validated SQL on an explicitly read-only MySQL session."""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from hashlib import sha256
import json
from time import monotonic
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Connection, text
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
import sqlglot
from sqlglot import exp

from trade_agent.db.sql_validator import ValidatedSql


class SqlExecutionError(RuntimeError):
    """A database failure occurred without exposing SQL values or credentials."""

    error_code = "sql_execution_failed"


class SqlExecutionTimeout(SqlExecutionError):
    """MySQL terminated the SELECT at the configured session timeout."""

    error_code = "sql_timeout"


class SqlTransportError(SqlExecutionError):
    """The database connection failed in a way a workflow may retry."""

    error_code = "sql_transport_error"


class SqlScanBudgetExceeded(SqlExecutionError):
    """EXPLAIN estimated more examined rows than policy permits."""

    error_code = "sql_scan_budget_exceeded"


class SqlResultLimitExceeded(SqlExecutionError):
    """The driver returned more rows than the validated result bound."""

    error_code = "sql_result_limit_exceeded"


class SqlExecutionResult(BaseModel):
    """Internal aggregate result; downstream callers normalize it to Evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    query_id: str
    normalized_sql: str
    bound_filter_names: tuple[str, ...]
    schema_fingerprint: str
    dataset_id: str
    is_synthetic: bool
    effective_start_date: date
    effective_end_date: date
    aggregation_grain: tuple[str, ...]
    time_grain: str
    metric_names: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    row_count: int = Field(ge=0)
    result_hash: str
    raw_record_ids: tuple[str, ...]
    raw_record_ids_truncated: bool
    estimated_scan_rows: int = Field(ge=0)
    execution_ms: float = Field(ge=0)
    max_execution_time_ms: int = Field(gt=0)
    client_timeout_ms: int = Field(gt=0)


class ReadOnlySqlExecutor:
    """Own a short read-only transaction on an existing MySQL connection."""

    def __init__(
        self,
        connection: Connection,
        *,
        max_execution_time_ms: int = 2_000,
        client_timeout_ms: int | None = None,
        max_scan_rows: int = 100_000,
        max_locator_rows: int = 500,
    ) -> None:
        if not isinstance(connection, Connection):
            raise TypeError("connection must be a SQLAlchemy Connection")
        if type(max_execution_time_ms) is not int or max_execution_time_ms <= 0:
            raise ValueError("max_execution_time_ms must be a positive integer")
        if type(max_scan_rows) is not int or max_scan_rows <= 0:
            raise ValueError("max_scan_rows must be a positive integer")
        if type(max_locator_rows) is not int or max_locator_rows <= 0:
            raise ValueError("max_locator_rows must be a positive integer")
        self.connection = connection
        self.max_execution_time_ms = max_execution_time_ms
        self.client_timeout_ms = (
            max_execution_time_ms + 1_000 if client_timeout_ms is None else client_timeout_ms
        )
        if type(self.client_timeout_ms) is not int or self.client_timeout_ms <= 0:
            raise ValueError("client_timeout_ms must be a positive integer")
        self.max_scan_rows = max_scan_rows
        self.max_locator_rows = max_locator_rows

    def execute(self, validated: ValidatedSql) -> SqlExecutionResult:
        if type(validated) is not ValidatedSql:
            raise TypeError("executor requires an exact ValidatedSql")
        started = monotonic()
        connection = self.connection
        if connection.in_transaction():
            connection.rollback()
        estimated_scan_rows = 0
        driver = connection.connection.driver_connection
        if not hasattr(driver, "_read_timeout"):
            raise SqlExecutionError("database driver does not expose a bounded client read timeout")
        previous_read_timeout = driver._read_timeout
        previous_socket_timeout = driver._sock.gettimeout() if getattr(driver, "_sock", None) else None
        driver._read_timeout = self.client_timeout_ms / 1_000
        try:
            connection.execute(text("SET SESSION TRANSACTION READ ONLY"))
            connection.execute(
                text("SET SESSION max_execution_time = :timeout_ms"),
                {"timeout_ms": self.max_execution_time_ms},
            )
            connection.execute(text("START TRANSACTION READ ONLY"))
            explain_rows = connection.execute(
                text(f"EXPLAIN {validated.sql}"), dict(validated.params)
            ).mappings().all()
            estimated_scan_rows = self._estimated_rows(explain_rows)
            if estimated_scan_rows > self.max_scan_rows:
                raise SqlScanBudgetExceeded(
                    f"estimated scan rows {estimated_scan_rows} exceed configured budget"
                )

            result = connection.execute(text(validated.sql), dict(validated.params)).mappings()
            rows = tuple(dict(row) for row in result.fetchmany(validated.limit + 1))
            if len(rows) > validated.limit:
                raise SqlResultLimitExceeded("database returned more rows than the validated limit")
            locator_sql = self._locator_sql(validated.sql, self.max_locator_rows + 1)
            locator_result = connection.execute(text(locator_sql), dict(validated.params)).scalars()
            locator_values = tuple(str(value) for value in locator_result.fetchmany(self.max_locator_rows + 1))
            truncated = len(locator_values) > self.max_locator_rows
            raw_record_ids = locator_values[: self.max_locator_rows]
            result_hash = self._hash_rows(rows)
            query_id = sha256(
                json.dumps(
                    {
                        "sql": validated.sql,
                        "filters": validated.bound_filter_names,
                        "schema": validated.schema_fingerprint,
                        "dataset": validated.dataset_id,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            return SqlExecutionResult(
                query_id=query_id,
                normalized_sql=validated.sql,
                bound_filter_names=validated.bound_filter_names,
                schema_fingerprint=validated.schema_fingerprint,
                dataset_id=validated.dataset_id,
                is_synthetic=validated.is_synthetic,
                effective_start_date=validated.effective_start_date,
                effective_end_date=validated.effective_end_date,
                aggregation_grain=validated.aggregation_grain,
                time_grain=validated.time_grain,
                metric_names=validated.metric_names,
                rows=rows,
                row_count=len(rows),
                result_hash=result_hash,
                raw_record_ids=raw_record_ids,
                raw_record_ids_truncated=truncated,
                estimated_scan_rows=estimated_scan_rows,
                execution_ms=(monotonic() - started) * 1000,
                max_execution_time_ms=self.max_execution_time_ms,
                client_timeout_ms=self.client_timeout_ms,
            )
        except SqlExecutionError:
            raise
        except DBAPIError as error:
            code = self._mysql_error_code(error)
            message = str(error.orig).lower()
            if code in {1317, 3024} or (code == 2013 and "timed out" in message):
                raise SqlExecutionTimeout("read-only SELECT exceeded its execution timeout") from error
            if code in {2002, 2003, 2006, 2013}:
                raise SqlTransportError("database transport failed during read-only SQL execution") from error
            raise SqlExecutionError("read-only SQL execution failed") from error
        except SQLAlchemyError as error:
            raise SqlExecutionError("read-only SQL execution failed") from error
        finally:
            try:
                connection.exec_driver_sql("ROLLBACK")
                connection.exec_driver_sql("SET SESSION max_execution_time = 0")
                connection.rollback()
                driver._read_timeout = previous_read_timeout
                if getattr(driver, "_sock", None) is not None:
                    driver._sock.settimeout(previous_socket_timeout)
            except SQLAlchemyError:
                connection.invalidate()

    @staticmethod
    def _estimated_rows(rows: list[dict[str, Any]]) -> int:
        estimates = []
        prefix_rows = 1
        for row in rows:
            value = row.get("rows")
            if value is None:
                raise SqlScanBudgetExceeded("EXPLAIN did not provide a finite row estimate")
            try:
                table_rows = max(0, int(value))
                filtered = float(row.get("filtered", 100) or 100) / 100
                prefix_rows *= max(1, table_rows)
                estimates.append(max(1, int(prefix_rows * filtered)))
            except (TypeError, ValueError) as error:
                raise SqlScanBudgetExceeded("EXPLAIN row estimate was not an integer") from error
        if not estimates:
            raise SqlScanBudgetExceeded("EXPLAIN returned no query plan")
        return sum(estimates)

    @staticmethod
    def _locator_sql(sql: str, limit: int) -> str:
        tree = sqlglot.parse_one(sql, read="mysql")
        tree.set(
            "expressions",
            [exp.alias_(exp.column("raw_record_id", table="tr"), "raw_record_id")],
        )
        tree.set("group", None)
        tree.set("order", exp.Order(expressions=[exp.Ordered(this=exp.column("raw_record_id", table="tr"))]))
        tree.set("limit", exp.Limit(expression=exp.Literal.number(limit)))
        return tree.sql(dialect="mysql", pretty=False)

    @classmethod
    def _hash_rows(cls, rows: tuple[dict[str, Any], ...]) -> str:
        return sha256(
            json.dumps(
                rows,
                default=cls._json_value,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _json_value(value: object) -> str:
        if isinstance(value, (Decimal, date, datetime)):
            return value.isoformat() if hasattr(value, "isoformat") else str(value)
        raise TypeError(f"unsupported SQL result value: {type(value).__name__}")

    @staticmethod
    def _mysql_error_code(error: DBAPIError) -> int | None:
        original = error.orig
        if getattr(original, "args", ()) and type(original.args[0]) is int:
            return original.args[0]
        return None
