"""Compile validated business intent into a restricted, non-executable SQL plan."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from trade_agent.agents.intent import QueryIntent
from trade_agent.db.registry import RegistryJoin, RegistrySnapshot
from trade_agent.retrieval.filters import RetrievalFilter


StructuredPlanProvider = Callable[[QueryIntent], Mapping[str, object]]


class SchemaNotRegistered(ValueError):
    pass


class OutOfScopeRequest(ValueError):
    pass


class StructuredPlanRejected(ValueError):
    pass


class UnresolvedSqlConstraint(ValueError):
    pass


class UnsupportedQueryShape(ValueError):
    pass


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
    rag_filter: RetrievalFilter
    limit: int = Field(ge=1)


class SqlPlanner:
    def __init__(self, *, structured_provider: StructuredPlanProvider | None = None, entity_bindings: Mapping[str, int] | None = None) -> None:
        self.structured_provider = structured_provider
        self.entity_bindings = dict(entity_bindings or {})

    def plan(self, intent: QueryIntent, registry: RegistrySnapshot) -> SqlQueryPlan:
        if type(intent) is not QueryIntent or type(registry) is not RegistrySnapshot:
            raise TypeError("planner requires exact QueryIntent and RegistrySnapshot instances")
        if not intent.need_trade_data or intent.kind == "out_of_scope":
            raise OutOfScopeRequest("request does not permit a trade SQL plan")
        _require_registered(intent, registry)
        _require_supported_shape(intent)
        plan = _compile(intent, registry, self.entity_bindings)
        if self.structured_provider is None:
            return plan
        try:
            supplied = self.structured_provider(intent)
            if not isinstance(supplied, Mapping):
                raise TypeError("provider output must be a mapping")
            if set(supplied) - set(SqlQueryPlan.model_fields):
                raise StructuredPlanRejected("structured SQL plan contains unknown schema elements")
            candidate = SqlQueryPlan.model_validate(supplied)
        except StructuredPlanRejected:
            raise
        except (TypeError, ValidationError, ValueError) as error:
            raise StructuredPlanRejected("structured SQL plan does not satisfy the required schema") from error
        if candidate != plan:
            raise StructuredPlanRejected("structured SQL plan diverges from the constrained plan")
        return candidate


def _require_registered(intent: QueryIntent, registry: RegistrySnapshot) -> None:
    allowed = {"metrics": set(registry.aggregations), "dimensions": set(registry.dimensions) | set(registry.aliases), "filters": set(registry.filters) | set(registry.aliases)}
    missing = tuple(field for category, fields in (("metrics", intent.constraints.metrics), ("dimensions", intent.constraints.dimensions), ("filters", intent.constraints.filters)) for field in fields if field not in allowed[category])
    if missing:
        raise SchemaNotRegistered(", ".join(missing))


def _require_supported_shape(intent: QueryIntent) -> None:
    if len(intent.constraints.metrics) != 1:
        raise UnsupportedQueryShape("multiple metrics cannot be planned without changing aggregate grain")
    expected = {
        "top_importers": {intent.filters.company_role},
        "top_exporters": {intent.filters.company_role},
        "company_trend": {intent.filters.company_role, "time"},
        "country_hs_activity": {"time"} if intent.time_range.grain == "month" else set(),
        "lead_assessment": {intent.filters.company_role},
    }.get(intent.kind, set())
    if set(intent.constraints.dimensions) != expected:
        raise UnsupportedQueryShape("dimensions cannot be planned without discarding fields")


def _compile(intent: QueryIntent, registry: RegistrySnapshot, entity_bindings: Mapping[str, int]) -> SqlQueryPlan:
    metric = intent.constraints.metrics[0]
    role = intent.filters.company_role
    company_alias = "importer" if role == "importer_company" else "exporter"
    country_role = "import_country" if role == "importer_company" else "export_country"
    include_company = role in intent.constraints.dimensions
    requires_company = include_company or bool(intent.filters.company_names) or bool(intent.filters.entity_ids)
    company = _semantic_column(registry.dimensions, role, registry) if requires_company else None
    country_code = _physical_column(_identifier(registry, country_role), registry) if intent.filters.country_codes else None
    country_region = _semantic_column(registry.filters, "region", registry) if intent.retrieval_filter.region is not None else None
    hs_code = _physical_column(_identifier(registry, "hs_code"), registry) if intent.filters.hs_codes else None
    trade_date = _semantic_column(registry.dimensions, "time", registry)
    role_join = _validated_join(registry, role) if requires_company else None
    country_join = _validated_join(registry, country_role) if intent.filters.country_codes or intent.retrieval_filter.region is not None else None
    hs_join = _validated_join(registry, "hs_code") if intent.filters.hs_codes else None
    _require_table(registry, "trade_records")
    tables = [TableRef(table="trade_records", alias="tr")]
    joins: list[PlannedJoin] = []
    if requires_company:
        _require_table(registry, "companies")
        tables.append(TableRef(table="companies", alias=company_alias))
        joins.append(_aliased_join(role_join, company_alias))
    if intent.filters.country_codes or intent.retrieval_filter.region is not None:
        _require_table(registry, "countries")
        tables.append(TableRef(table="countries", alias=country_role))
        joins.append(_aliased_join(country_join, country_role))
    if intent.filters.hs_codes:
        _require_table(registry, "hs_codes")
        tables.append(TableRef(table="hs_codes", alias="hs"))
        joins.append(_aliased_join(hs_join, "hs"))

    group_by: list[str] = []
    columns: list[SelectColumn] = []
    if include_company:
        field = _alias(company, company_alias)
        group_by.append(field)
        columns.append(SelectColumn(expression=field, alias=role))
    if intent.time_range.grain in {"day", "month"}:
        field = _alias(trade_date, "tr")
        group_by.append(field)
        columns.append(SelectColumn(expression=field, alias="trade_date"))

    definition = registry.aggregations[metric]
    value = _alias(_physical_column(str(definition["column"]), registry), "tr")
    currency = _alias(_physical_column(str(definition["currency_column"]), registry), "tr") if "currency_column" in definition else None
    unit = _alias(_physical_column(str(definition["unit_column"]), registry), "tr") if "unit_column" in definition else None
    for field, name in ((currency, "currency"), (unit, "unit")):
        if field:
            group_by.append(field)
            columns.append(SelectColumn(expression=field, alias=name))
    columns.append(SelectColumn(expression=value, alias=metric))

    predicates: list[Predicate] = []
    if intent.filters.country_codes:
        predicates.append(Predicate(column=_alias(country_code, country_role), operator="in", values=intent.filters.country_codes))
    if intent.retrieval_filter.region is not None:
        predicates.append(Predicate(column=_alias(country_region, country_role), operator="equals", values=(intent.retrieval_filter.region,)))
    if intent.filters.hs_codes:
        predicates.append(Predicate(column=_alias(hs_code, "hs"), operator="in", values=intent.filters.hs_codes))
    if intent.filters.company_names:
        predicates.append(Predicate(column=_alias(company, company_alias), operator="in", values=intent.filters.company_names))
    if intent.filters.entity_ids:
        _physical_column("companies.id", registry)
        predicates.append(Predicate(column=f"{company_alias}.id", operator="in", values=_entity_ids(intent.filters.entity_ids, entity_bindings)))
    if intent.time_range.start is not None:
        predicates.append(Predicate(column=_alias(trade_date, "tr"), operator="between", values=(intent.time_range.start.isoformat(), intent.time_range.end.isoformat())))

    return SqlQueryPlan(
        schema_fingerprint=registry.fingerprint, tables=tuple(tables), columns=tuple(columns), joins=tuple(joins), predicates=tuple(predicates),
        group_by=tuple(group_by), time_grain=intent.time_range.grain,
        aggregations=(Aggregation(metric=metric, operation=str(tuple(definition["operations"])[0]), column=value, currency_column=currency, unit_column=unit, grain=tuple(group_by)),),
        order_by=(OrderBy(expression=metric, direction="desc"),), rag_filter=intent.retrieval_filter, limit=min(intent.limit, registry.max_result_rows),
    )


def _entity_ids(entity_ids: tuple[str, ...], bindings: Mapping[str, int]) -> tuple[str, ...]:
    if any(item not in bindings for item in entity_ids):
        raise UnresolvedSqlConstraint("entity_ids have no reviewed company binding")
    values = tuple(bindings[item] for item in entity_ids)
    if any(type(value) is not int or value <= 0 for value in values):
        raise UnresolvedSqlConstraint("entity binding must be a positive reviewed company primary key")
    return tuple(str(value) for value in values)


def _identifier(registry: RegistrySnapshot, role: str) -> str:
    if role not in registry.identifiers:
        raise SchemaNotRegistered(f"identifier {role}")
    return registry.identifiers[role]


def _require_table(registry: RegistrySnapshot, table: str) -> None:
    if table not in registry.tables:
        raise SchemaNotRegistered(table)


def _physical_column(column: str, registry: RegistrySnapshot) -> str:
    if column not in registry.columns:
        raise SchemaNotRegistered(column)
    _require_table(registry, column.split(".", maxsplit=1)[0])
    return column


def _semantic_column(mapping: Mapping[str, str], field: str, registry: RegistrySnapshot) -> str:
    if field not in mapping:
        raise SchemaNotRegistered(field)
    return _physical_column(mapping[field], registry)


def _validated_join(registry: RegistrySnapshot, role: str) -> RegistryJoin:
    matches = [join for join in registry.joins if role in join.roles]
    if len(matches) != 1:
        raise SchemaNotRegistered(f"join role {role}")
    join = matches[0]
    _physical_column(join.left, registry)
    _physical_column(join.right, registry)
    return join


def _alias(column: str, alias: str) -> str:
    return f"{alias}.{column.rsplit('.', maxsplit=1)[1]}"


def _aliased_join(join: RegistryJoin, target_alias: str) -> PlannedJoin:
    left_table, left_column = join.left.split(".", maxsplit=1)
    right_table, right_column = join.right.split(".", maxsplit=1)
    return PlannedJoin(name=join.name, left=f"{'tr' if left_table == 'trade_records' else target_alias}.{left_column}", right=f"{'tr' if right_table == 'trade_records' else target_alias}.{right_column}")
