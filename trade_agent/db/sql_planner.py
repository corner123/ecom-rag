"""Compile validated business intent into a restricted, non-executable SQL plan."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from trade_agent.agents.intent import QueryIntent
from trade_agent.db.registry import RegistryJoin, RegistrySnapshot


StructuredPlanProvider = Callable[[QueryIntent], Mapping[str, object]]


class SchemaNotRegistered(ValueError):
    """A request refers to a business field absent from the refreshed registry."""


class OutOfScopeRequest(ValueError):
    """The SQL planner is not permitted to plan this request."""


class StructuredPlanRejected(ValueError):
    """A structured provider response does not satisfy the local plan schema."""


class TableRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    table: str
    alias: str


class SelectColumn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    expression: str
    alias: str


class PlannedJoin(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    left: str
    right: str


class Predicate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str
    operator: Literal["in", "equals", "between"]
    values: tuple[str, ...]


class Aggregation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str
    operation: Literal["sum", "count", "max"]
    column: str
    currency_column: str | None = None
    unit_column: str | None = None
    grain: tuple[str, ...] = ()


class OrderBy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    expression: str
    direction: Literal["asc", "desc"]


class SqlQueryPlan(BaseModel):
    """An allowlisted data plan.  It intentionally has no SQL string field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_fingerprint: str
    tables: tuple[TableRef, ...]
    columns: tuple[SelectColumn, ...]
    joins: tuple[PlannedJoin, ...]
    predicates: tuple[Predicate, ...]
    group_by: tuple[str, ...]
    time_grain: Literal["total", "day", "month"]
    aggregations: tuple[Aggregation, ...]
    order_by: tuple[OrderBy, ...]
    limit: int = Field(ge=1)


class SqlPlanner:
    """Plan only schema-linked aggregate queries over the refreshed snapshot."""

    def __init__(self, *, structured_provider: StructuredPlanProvider | None = None) -> None:
        self.structured_provider = structured_provider

    def plan(self, intent: QueryIntent, registry: RegistrySnapshot) -> SqlQueryPlan:
        if type(intent) is not QueryIntent:
            raise TypeError("intent must be an exact business QueryIntent instance")
        if type(registry) is not RegistrySnapshot:
            raise TypeError("registry must be an exact RegistrySnapshot instance")
        if not intent.need_trade_data or intent.kind == "out_of_scope":
            raise OutOfScopeRequest("request does not permit a trade SQL plan")
        _require_registered(intent, registry)
        deterministic = _compile(intent, registry)
        if self.structured_provider is None:
            return deterministic
        try:
            provided = self.structured_provider(intent)
            if not isinstance(provided, Mapping):
                raise TypeError("provider output must be a mapping")
            unknown = set(provided) - set(SqlQueryPlan.model_fields)
            if unknown:
                raise StructuredPlanRejected("structured SQL plan contains unknown schema elements")
            candidate = SqlQueryPlan.model_validate(provided)
        except StructuredPlanRejected:
            raise
        except (TypeError, ValidationError, ValueError) as error:
            raise StructuredPlanRejected("structured SQL plan does not satisfy the required schema") from error
        if candidate != deterministic:
            raise StructuredPlanRejected("structured SQL plan diverges from the constrained plan")
        return candidate


def _require_registered(intent: QueryIntent, registry: RegistrySnapshot) -> None:
    allowed = {
        "metrics": set(registry.aggregations),
        "dimensions": set(registry.dimensions) | set(registry.aliases),
        "filters": set(registry.filters) | set(registry.aliases),
    }
    missing = tuple(
        field
        for category, requested in (
            ("metrics", intent.constraints.metrics),
            ("dimensions", intent.constraints.dimensions),
            ("filters", intent.constraints.filters),
        )
        for field in requested
        if field not in allowed[category]
    )
    if missing:
        raise SchemaNotRegistered(", ".join(missing))


def _compile(intent: QueryIntent, registry: RegistrySnapshot) -> SqlQueryPlan:
    metric = intent.constraints.metrics[0] if intent.constraints.metrics else "trade_count"
    if metric not in registry.aggregations:
        raise SchemaNotRegistered(metric)
    role = intent.filters.company_role
    company_alias = "importer" if role == "importer_company" else "exporter"
    country_role = "import_country" if role == "importer_company" else "export_country"
    country_alias = country_role
    country_identifier = registry.identifiers.get(country_role)
    if country_identifier != "countries.country_code":
        raise SchemaNotRegistered(f"identifier {country_role}")
    role_join = _join_for_role(registry, role)
    country_join = _join_for_role(registry, country_role)
    hs_join = _join_for_role(registry, "hs_code")

    tables: list[TableRef] = [TableRef(table="trade_records", alias="tr"), TableRef(table="companies", alias=company_alias)]
    joins: list[PlannedJoin] = [_aliased_join(role_join, company_alias)]
    if intent.filters.country_codes:
        tables.append(TableRef(table="countries", alias=country_alias))
        joins.append(_aliased_join(country_join, country_alias))
    if intent.filters.hs_codes:
        tables.append(TableRef(table="hs_codes", alias="hs"))
        joins.append(_aliased_join(hs_join, "hs"))

    group_by: list[str] = []
    columns: list[SelectColumn] = []
    if intent.kind in {"top_importers", "company_trend", "lead_assessment"}:
        group_by.append(f"{company_alias}.company_name")
        columns.append(SelectColumn(expression=f"{company_alias}.company_name", alias=role))
    if intent.time_range.grain in {"day", "month"}:
        group_by.append("tr.trade_date")
        columns.append(SelectColumn(expression="tr.trade_date", alias="trade_date"))

    definition = registry.aggregations[metric]
    operation = str(tuple(definition["operations"])[0])
    source_column = str(definition["column"])
    column = _with_trade_alias(source_column)
    currency_column = _with_trade_alias(str(definition["currency_column"])) if "currency_column" in definition else None
    unit_column = _with_trade_alias(str(definition["unit_column"])) if "unit_column" in definition else None
    if currency_column is not None:
        group_by.append(currency_column)
        columns.append(SelectColumn(expression=currency_column, alias="currency"))
    if unit_column is not None:
        group_by.append(unit_column)
        columns.append(SelectColumn(expression=unit_column, alias="unit"))
    columns.append(SelectColumn(expression=column, alias=metric))

    predicates: list[Predicate] = []
    if intent.filters.country_codes:
        predicates.append(
            Predicate(
                column=f"{country_alias}.{country_identifier.rsplit('.', maxsplit=1)[1]}",
                operator="in",
                values=intent.filters.country_codes,
            )
        )
    if intent.filters.hs_codes:
        predicates.append(Predicate(column="hs.hs_code", operator="in", values=intent.filters.hs_codes))
    if intent.filters.company_names:
        predicates.append(Predicate(column=f"{company_alias}.company_name", operator="in", values=intent.filters.company_names))
    if intent.time_range.start is not None:
        predicates.append(
            Predicate(
                column="tr.trade_date",
                operator="between",
                values=(intent.time_range.start.isoformat(), intent.time_range.end.isoformat()),
            )
        )
    grain = tuple(group_by)
    aggregation = Aggregation(
        metric=metric,
        operation=operation,
        column=column,
        currency_column=currency_column,
        unit_column=unit_column,
        grain=grain,
    )
    return SqlQueryPlan(
        schema_fingerprint=registry.fingerprint,
        tables=tuple(tables),
        columns=tuple(columns),
        joins=tuple(joins),
        predicates=tuple(predicates),
        group_by=tuple(group_by),
        time_grain=intent.time_range.grain,
        aggregations=(aggregation,),
        order_by=(OrderBy(expression=metric, direction="desc"),),
        limit=min(intent.limit, registry.max_result_rows),
    )


def _join_for_role(registry: RegistrySnapshot, role: str) -> RegistryJoin:
    matches = [join for join in registry.joins if role in join.roles]
    if len(matches) != 1:
        raise SchemaNotRegistered(f"join role {role}")
    return matches[0]


def _aliased_join(join: RegistryJoin, dimension_alias: str) -> PlannedJoin:
    left_table, left_column = join.left.split(".", maxsplit=1)
    right_table, right_column = join.right.split(".", maxsplit=1)
    left_alias = "tr" if left_table == "trade_records" else dimension_alias if left_table in {"companies", "countries", "hs_codes"} else left_table
    right_alias = "tr" if right_table == "trade_records" else dimension_alias if right_table in {"companies", "countries", "hs_codes"} else right_table
    return PlannedJoin(name=join.name, left=f"{left_alias}.{left_column}", right=f"{right_alias}.{right_column}")


def _with_trade_alias(column: str) -> str:
    table, name = column.split(".", maxsplit=1)
    return f"tr.{name}" if table == "trade_records" else column
