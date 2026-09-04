"""Render reviewed SQL plans without interpolating any caller-controlled value."""
from __future__ import annotations

from datetime import date
from decimal import Decimal
import re
from types import MappingProxyType
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trade_agent.db.sql_planner import SqlQueryPlan


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_QUALIFIED = re.compile(r"^([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)$")
SqlParameter = str | int | bool | date | Decimal


class SqlRenderRejected(ValueError):
    """A typed plan contains an identifier or shape the renderer cannot emit."""


class SqlDataScope(BaseModel):
    """Reviewed single-dataset boundary applied to every trade query."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    dataset_id: str = Field(min_length=1, max_length=128)
    synthetic: bool
    start_date: date
    end_date: date
    tenant_mapping: dict[str, str] | None = None

    @model_validator(mode="after")
    def ordered_dates(self) -> "SqlDataScope":
        if self.start_date > self.end_date:
            raise ValueError("data scope start_date must not exceed end_date")
        return self


class RenderedSql(BaseModel):
    """SQL text plus separately bound values and renderer provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    sql: str
    params: Mapping[str, SqlParameter] = Field(default_factory=dict)
    schema_fingerprint: str = ""
    dataset_id: str = ""
    effective_start_date: date | None = None
    effective_end_date: date | None = None
    aggregation_grain: tuple[str, ...] = ()
    time_grain: str = "total"
    metric_names: tuple[str, ...] = ()
    planned_tables: tuple[tuple[str, str], ...] = ()
    planned_joins: tuple[tuple[str, str, str], ...] = ()
    tenant_scope: str | None = None

    @field_validator("sql")
    @classmethod
    def nonblank_sql(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("sql must not be blank")
        return value

    @field_validator("params")
    @classmethod
    def immutable_params(cls, value: Mapping[str, SqlParameter]) -> Mapping[str, SqlParameter]:
        return MappingProxyType(dict(value))


class SqlRenderer:
    """Compile only ``SqlQueryPlan`` fields and fixed reviewed policy fragments."""

    def __init__(self, *, scope: SqlDataScope) -> None:
        if type(scope) is not SqlDataScope:
            raise TypeError("scope must be an exact SqlDataScope")
        self.scope = scope

    def render(self, plan: SqlQueryPlan) -> RenderedSql:
        if type(plan) is not SqlQueryPlan:
            raise TypeError("renderer requires an exact SqlQueryPlan")
        aliases = self._table_aliases(plan)
        if aliases.get("tr") != "trade_records":
            raise SqlRenderRejected("trade_records must be the tr base table")
        if "data_scope" in aliases:
            raise SqlRenderRejected("data_scope alias is reserved for policy enforcement")

        aggregate_aliases = {aggregation.metric for aggregation in plan.aggregations}
        selections: list[str] = []
        for column in plan.columns:
            if column.alias in aggregate_aliases:
                continue
            expression = self._column(column.expression, aliases)
            if plan.time_grain == "month" and column.alias == "trade_date":
                expression = f"DATE_FORMAT({expression}, '%Y-%m')"
            selections.append(f"{expression} AS {self._identifier(column.alias)}")
        for aggregation in plan.aggregations:
            column = self._column(aggregation.column, aliases)
            function = {"sum": "SUM", "count": "COUNT", "max": "MAX"}.get(aggregation.operation)
            if function is None:
                raise SqlRenderRejected("aggregation operation is not reviewed")
            selections.append(f"{function}({column}) AS {self._identifier(aggregation.metric)}")
        if not selections:
            raise SqlRenderRejected("query must select at least one reviewed expression")

        sql = [f"SELECT {', '.join(selections)} FROM trade_records AS tr"]
        joined_aliases = {"tr"}
        table_by_alias = {table.alias: table.table for table in plan.tables}
        joins_by_target: dict[str, tuple[str, str]] = {}
        for join in plan.joins:
            left_alias = self._qualified(join.left)[0]
            right_alias = self._qualified(join.right)[0]
            if left_alias == "tr" and right_alias != "tr":
                target, left, right = right_alias, join.left, join.right
            elif right_alias == "tr" and left_alias != "tr":
                target, left, right = left_alias, join.left, join.right
            else:
                raise SqlRenderRejected("joins must connect directly to the trade-record base")
            if target in joins_by_target:
                raise SqlRenderRejected("duplicate join target")
            joins_by_target[target] = (left, right)
        for table in plan.tables:
            if table.alias == "tr":
                continue
            if table.alias not in joins_by_target:
                raise SqlRenderRejected("every planned table must have a reviewed join")
            left, right = joins_by_target[table.alias]
            sql.append(
                f"JOIN {self._identifier(table.table)} AS {self._identifier(table.alias)} "
                f"ON {self._column(left, aliases)} = {self._column(right, aliases)}"
            )
            joined_aliases.add(table.alias)
        if joined_aliases != set(table_by_alias):
            raise SqlRenderRejected("join graph does not cover every planned alias")

        sql.append("JOIN data_sources AS data_scope ON tr.source_id = data_scope.id")
        params: dict[str, SqlParameter] = {
            "policy_is_synthetic": self.scope.synthetic,
            "policy_start_date": self.scope.start_date,
            "policy_end_date": self.scope.end_date,
        }
        predicates = [
            "data_scope.is_synthetic = :policy_is_synthetic",
            "tr.trade_date BETWEEN :policy_start_date AND :policy_end_date",
        ]
        effective_start = self.scope.start_date
        effective_end = self.scope.end_date
        for index, predicate in enumerate(plan.predicates):
            column = self._column(predicate.column, aliases)
            if predicate.operator == "equals" and len(predicate.values) == 1:
                name = f"filter_{index}_0"
                predicates.append(f"{column} = :{name}")
                params[name] = predicate.values[0]
            elif predicate.operator == "in" and predicate.values:
                names = []
                for value_index, value in enumerate(predicate.values):
                    name = f"filter_{index}_{value_index}"
                    names.append(f":{name}")
                    params[name] = value
                predicates.append(f"{column} IN ({', '.join(names)})")
            elif predicate.operator == "between" and len(predicate.values) == 2:
                start_name, end_name = f"filter_{index}_start", f"filter_{index}_end"
                predicates.append(f"{column} BETWEEN :{start_name} AND :{end_name}")
                params[start_name], params[end_name] = predicate.values
                if predicate.column == "tr.trade_date":
                    try:
                        requested_start = date.fromisoformat(predicate.values[0])
                        requested_end = date.fromisoformat(predicate.values[1])
                    except ValueError as error:
                        raise SqlRenderRejected("trade-date values must be ISO dates") from error
                    effective_start = max(effective_start, requested_start)
                    effective_end = min(effective_end, requested_end)
            else:
                raise SqlRenderRejected("predicate shape is not reviewed")
        sql.append(f"WHERE {' AND '.join(predicates)}")

        if plan.group_by:
            groups = []
            for expression in plan.group_by:
                group = self._column(expression, aliases)
                if plan.time_grain == "month" and expression == "tr.trade_date":
                    group = f"DATE_FORMAT({group}, '%Y-%m')"
                groups.append(group)
            sql.append(f"GROUP BY {', '.join(groups)}")
        if plan.order_by:
            order_parts = []
            selectable_aliases = {column.alias for column in plan.columns} | aggregate_aliases
            for order in plan.order_by:
                if order.expression not in selectable_aliases:
                    raise SqlRenderRejected("order expression is not a selected alias")
                order_parts.append(f"{self._identifier(order.expression)} {order.direction.upper()}")
            sql.append(f"ORDER BY {', '.join(order_parts)}")
        sql.append(f"LIMIT {plan.limit}")

        return RenderedSql(
            sql=" ".join(sql),
            params=params,
            schema_fingerprint=plan.schema_fingerprint,
            dataset_id=self.scope.dataset_id,
            effective_start_date=effective_start,
            effective_end_date=effective_end,
            aggregation_grain=plan.aggregations[0].grain,
            time_grain=plan.time_grain,
            metric_names=tuple(aggregation.metric for aggregation in plan.aggregations),
            planned_tables=tuple((table.alias, table.table) for table in plan.tables)
            + (("data_scope", "data_sources"),),
            planned_joins=tuple((join.name, join.left, join.right) for join in plan.joins)
            + (("fk_trade_records_source", "tr.source_id", "data_scope.id"),),
        )

    @classmethod
    def _table_aliases(cls, plan: SqlQueryPlan) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for table in plan.tables:
            name, alias = cls._identifier(table.table), cls._identifier(table.alias)
            if alias in aliases:
                raise SqlRenderRejected("table aliases must be unique")
            aliases[alias] = name
        return aliases

    @staticmethod
    def _identifier(value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise SqlRenderRejected("identifier is outside the reviewed grammar")
        return value

    @staticmethod
    def _qualified(value: str) -> tuple[str, str]:
        match = _QUALIFIED.fullmatch(value)
        if match is None:
            raise SqlRenderRejected("column must be a qualified reviewed identifier")
        return match.group(1), match.group(2)

    @classmethod
    def _column(cls, value: str, aliases: dict[str, str]) -> str:
        alias, column = cls._qualified(value)
        if alias not in aliases:
            raise SqlRenderRejected("column uses an unknown table alias")
        return f"{alias}.{column}"
