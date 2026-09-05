"""Fail-closed SQLGlot AST and policy validation for rendered trade SQL."""
from __future__ import annotations

from datetime import date
from decimal import Decimal
import re
from types import MappingProxyType
from collections.abc import Mapping
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from trade_agent.db.registry import RegistrySnapshot
from trade_agent.db.sql_planner import SqlQueryPlan
from trade_agent.db.sql_renderer import (
    ProjectionExpression,
    ProjectionItem,
    ProjectionOrder,
    RenderedSql,
    SemanticProjectionManifest,
    SqlDataScope,
    SqlParameter,
    SqlRenderRejected,
    build_projection_manifest,
)


_COMMENT = re.compile(r"(?:/\*|\*/|--|#)")


class SqlAstRejected(ValueError):
    """SQL is outside the reviewed SELECT grammar."""

    error_code = "sql_ast_rejected"


class SqlPolicyDenied(SqlAstRejected):
    """SQL does not preserve the mandatory dataset or request scope."""

    error_code = "policy_denied"


class ValidatedSql(BaseModel):
    """Normalized executable SQL plus non-secret validation metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    sql: str
    params: Mapping[str, SqlParameter]
    schema_fingerprint: str
    dataset_id: str
    is_synthetic: bool
    effective_start_date: date
    effective_end_date: date
    aggregation_grain: tuple[str, ...]
    time_grain: str
    metric_names: tuple[str, ...]
    projection_aliases: tuple[str, ...]
    bound_filter_names: tuple[str, ...]
    limit: int = Field(ge=1)

    @field_validator("params")
    @classmethod
    def immutable_params(cls, value: Mapping[str, SqlParameter]) -> Mapping[str, SqlParameter]:
        return MappingProxyType(dict(value))


class SqlValidator:
    """Treat SQLGlot as a parser, then enforce every accepted AST shape."""

    def __init__(self, *, scope: SqlDataScope) -> None:
        if type(scope) is not SqlDataScope:
            raise TypeError("scope must be an exact SqlDataScope")
        self.scope = scope

    def validate(self, rendered: RenderedSql, registry: RegistrySnapshot) -> ValidatedSql:
        if type(rendered) is not RenderedSql or type(registry) is not RegistrySnapshot:
            raise TypeError("validator requires exact RenderedSql and RegistrySnapshot instances")
        if rendered.tenant_scope is not None:
            if not self.scope.tenant_mapping or rendered.tenant_scope not in self.scope.tenant_mapping:
                raise SqlPolicyDenied("tenant scope has no reviewed mapping")
            raise SqlPolicyDenied("tenant-scoped SQL is unavailable for the single-dataset schema")
        if rendered.schema_fingerprint != registry.fingerprint:
            raise SqlPolicyDenied("schema fingerprint does not match the reviewed registry")
        if rendered.dataset_id != self.scope.dataset_id:
            raise SqlPolicyDenied("dataset identity does not match the configured scope")
        if rendered.plan is None or rendered.projection_manifest is None:
            raise SqlAstRejected("SQL lacks a plan-derived projection manifest")
        if rendered.plan.schema_fingerprint != registry.fingerprint:
            raise SqlPolicyDenied("plan schema fingerprint does not match the reviewed registry")
        try:
            expected_manifest = build_projection_manifest(rendered.plan)
        except SqlRenderRejected as error:
            raise SqlAstRejected("projection aggregate manifest is not reviewed") from error
        if rendered.projection_manifest != expected_manifest:
            raise SqlAstRejected("projection manifest diverges from the typed SQL plan")
        self._validate_manifest_semantics(expected_manifest, rendered.plan, registry)
        if (
            rendered.effective_start_date is None
            or rendered.effective_end_date is None
            or rendered.effective_start_date < self.scope.start_date
            or rendered.effective_end_date > self.scope.end_date
            or rendered.effective_start_date > rendered.effective_end_date
        ):
            raise SqlPolicyDenied("query time range does not intersect the configured data scope")
        if _COMMENT.search(rendered.sql):
            raise SqlAstRejected("SQL comments and optimizer hints are forbidden")
        try:
            statements = sqlglot.parse(rendered.sql, read="mysql", error_level="RAISE")
        except ParseError as error:
            raise SqlAstRejected("SQL could not be parsed completely") from error
        if len(statements) != 1 or statements[0] is None:
            raise SqlAstRejected("exactly one SQL statement is required")
        tree = statements[0]
        if type(tree) is not exp.Select:
            raise SqlAstRejected("only a top-level SELECT is allowed")
        if any(isinstance(node, (exp.Subquery, exp.SetOperation, exp.With, exp.CTE)) for node in tree.walk()):
            raise SqlAstRejected("subqueries, CTEs, and set operations are forbidden")
        if tree.args.get("hint") is not None or any(node.comments for node in tree.walk()):
            raise SqlAstRejected("SQL comments and optimizer hints are forbidden")
        allowed_root_args = {"expressions", "from_", "joins", "where", "group", "order", "limit"}
        if any(value is not None and key not in allowed_root_args for key, value in tree.args.items()):
            raise SqlAstRejected("SELECT contains an unreviewed clause")

        alias_to_table = self._validate_tables(tree, rendered, registry)
        self._validate_joins(tree, rendered, registry, alias_to_table)
        actual_projections = self._validate_select(tree, registry, alias_to_table)
        if actual_projections != expected_manifest.projections:
            raise SqlAstRejected("SQL projection diverges from the typed plan manifest")
        self._validate_where(tree, registry, alias_to_table, rendered.params)
        actual_groups, actual_orders = self._validate_group_order(
            tree, alias_to_table, expected_manifest
        )
        actual_time_grain = self._derive_time_grain(
            actual_projections, actual_groups, registry
        )
        if actual_time_grain != expected_manifest.time_grain:
            raise SqlAstRejected("SQL time grain diverges from the accepted time expression")
        accepted_manifest = SemanticProjectionManifest(
            projections=actual_projections,
            group_by=actual_groups,
            order_by=actual_orders,
            time_grain=actual_time_grain,
        )
        limit = self._validate_limit(tree, registry)
        self._validate_parameters(tree, rendered.params)
        self._validate_policy(tree, rendered.params, alias_to_table)
        self._validate_effective_time(tree, rendered, alias_to_table)
        aggregation_grain = self._aggregation_grain(accepted_manifest)
        metric_names = tuple(
            item.alias for item in accepted_manifest.projections if item.operation is not None
        )

        return ValidatedSql(
            sql=tree.sql(dialect="mysql", pretty=False),
            params=dict(rendered.params),
            schema_fingerprint=registry.fingerprint,
            dataset_id=rendered.dataset_id,
            is_synthetic=self.scope.synthetic,
            effective_start_date=rendered.effective_start_date,
            effective_end_date=rendered.effective_end_date,
            aggregation_grain=aggregation_grain,
            time_grain=accepted_manifest.time_grain,
            metric_names=metric_names,
            projection_aliases=tuple(item.alias for item in accepted_manifest.projections),
            bound_filter_names=tuple(sorted(rendered.params)),
            limit=limit,
        )

    @staticmethod
    def _validate_tables(
        tree: exp.Select, rendered: RenderedSql, registry: RegistrySnapshot
    ) -> dict[str, str]:
        tables = tuple(tree.find_all(exp.Table))
        bindings: list[tuple[str, str]] = []
        for table in tables:
            if table.db or table.catalog or not table.alias:
                raise SqlAstRejected("tables must be unqualified and explicitly aliased")
            if table.name not in registry.tables:
                raise SqlAstRejected("table is not registered")
            bindings.append((table.alias, table.name))
        if len({alias for alias, _ in bindings}) != len(bindings):
            raise SqlAstRejected("table aliases must be unique")
        if tuple(bindings) != rendered.planned_tables:
            raise SqlAstRejected("SQL table bindings diverge from the reviewed plan")
        if not bindings or bindings[0] != ("tr", "trade_records"):
            raise SqlAstRejected("trade_records must be the base table")
        return dict(bindings)

    @classmethod
    def _validate_joins(
        cls,
        tree: exp.Select,
        rendered: RenderedSql,
        registry: RegistrySnapshot,
        alias_to_table: dict[str, str],
    ) -> None:
        actual: list[tuple[str, str]] = []
        for join in tree.args.get("joins") or ():
            if type(join) is not exp.Join or join.args.get("on") is None:
                raise SqlAstRejected("cartesian and implicit joins are forbidden")
            if join.args.get("side") not in (None, "") or join.args.get("kind") not in (None, "", "INNER"):
                raise SqlAstRejected("only inner joins are reviewed")
            on = join.args["on"]
            if type(on) is not exp.EQ or type(on.this) is not exp.Column or type(on.expression) is not exp.Column:
                raise SqlAstRejected("join condition must be one column equality")
            left = cls._physical(on.this, alias_to_table)
            right = cls._physical(on.expression, alias_to_table)
            if left not in registry.columns or right not in registry.columns:
                raise SqlAstRejected("join column is not registered")
            actual.append((left, right))
        expected_physical = []
        registry_by_name = {join.name: join for join in registry.joins}
        for name, left, right in rendered.planned_joins:
            registered = registry_by_name.get(name)
            if registered is None:
                raise SqlAstRejected("planned join is not registered")
            expected_pair = frozenset((registered.left, registered.right))
            planned_pair = frozenset(
                (cls._physical_sql_name(left, alias_to_table), cls._physical_sql_name(right, alias_to_table))
            )
            if planned_pair != expected_pair:
                raise SqlAstRejected("planned join endpoints do not match the registry")
            expected_physical.append((registered.left, registered.right))
        if len(actual) != len(expected_physical):
            raise SqlAstRejected("SQL join count diverges from the reviewed plan")
        for pair, expected in zip(actual, expected_physical, strict=True):
            if frozenset(pair) != frozenset(expected):
                raise SqlAstRejected("SQL join endpoints diverge from the reviewed plan")

    @classmethod
    def _validate_select(
        cls, tree: exp.Select, registry: RegistrySnapshot, alias_to_table: dict[str, str]
    ) -> tuple[ProjectionItem, ...]:
        if not tree.expressions:
            raise SqlAstRejected("SELECT list must not be empty")
        aliases: set[str] = set()
        projections: list[ProjectionItem] = []
        allowed_dimensions = set(registry.dimensions.values())
        support_dimensions = {
            str(definition[key])
            for definition in registry.aggregations.values()
            for key in ("currency_column", "unit_column")
            if key in definition
        }
        for selection in tree.expressions:
            if type(selection) is not exp.Alias or not selection.alias:
                raise SqlAstRejected("every selected expression must have an explicit alias")
            if selection.alias in aliases:
                raise SqlAstRejected("selected aliases must be unique")
            aliases.add(selection.alias)
            expression = selection.this
            if type(expression) is exp.Column:
                physical = cls._registered_column(expression, alias_to_table, registry)
                if physical not in allowed_dimensions | support_dimensions:
                    raise SqlAstRejected("projection column is not a reviewed dimension or unit")
                base = cls._projection_expression(expression, alias_to_table)
                projections.append(ProjectionItem(**base.model_dump(), alias=selection.alias))
            elif isinstance(expression, (exp.Sum, exp.Count, exp.Max)):
                if type(expression.this) is not exp.Column:
                    raise SqlAstRejected("aggregate argument must be one registered column")
                physical = cls._registered_column(expression.this, alias_to_table, registry)
                operation = {exp.Sum: "sum", exp.Count: "count", exp.Max: "max"}[type(expression)]
                definition = registry.aggregations.get(selection.alias)
                if (
                    definition is None
                    or operation not in tuple(str(item) for item in definition["operations"])
                    or physical != str(definition["column"])
                ):
                    raise SqlAstRejected("projection aggregate does not match its registered metric")
                base = cls._projection_expression(expression.this, alias_to_table)
                projections.append(
                    ProjectionItem(**base.model_dump(), alias=selection.alias, operation=operation)
                )
            elif cls._is_month_expression(expression, alias_to_table):
                column = next(expression.find_all(exp.Column))
                physical = cls._registered_column(column, alias_to_table, registry)
                if physical != "trade_records.trade_date":
                    raise SqlAstRejected("month bucketing is allowed only for trade_date")
                base = cls._projection_expression(column, alias_to_table, transform="month")
                projections.append(ProjectionItem(**base.model_dump(), alias=selection.alias))
            else:
                raise SqlAstRejected("selected expression is outside the reviewed grammar")
        return tuple(projections)

    @classmethod
    def _validate_where(
        cls,
        tree: exp.Select,
        registry: RegistrySnapshot,
        alias_to_table: dict[str, str],
        params: Mapping[str, SqlParameter],
    ) -> None:
        where = tree.args.get("where")
        if type(where) is not exp.Where:
            raise SqlPolicyDenied("every query requires scoped predicates")
        for predicate in cls._flatten_and(where.this):
            if type(predicate) is exp.EQ:
                if type(predicate.this) is not exp.Column or type(predicate.expression) is not exp.Placeholder:
                    raise SqlAstRejected("equality predicates require a column and bound parameter")
                cls._registered_column(predicate.this, alias_to_table, registry)
            elif type(predicate) is exp.In:
                if type(predicate.this) is not exp.Column or predicate.args.get("query") is not None:
                    raise SqlAstRejected("IN predicates require one column and bound values")
                cls._registered_column(predicate.this, alias_to_table, registry)
                if not predicate.expressions or any(type(item) is not exp.Placeholder for item in predicate.expressions):
                    raise SqlAstRejected("IN values must all be bound parameters")
            elif type(predicate) is exp.Between:
                if type(predicate.this) is not exp.Column:
                    raise SqlAstRejected("BETWEEN requires a registered column")
                cls._registered_column(predicate.this, alias_to_table, registry)
                if type(predicate.args.get("low")) is not exp.Placeholder or type(predicate.args.get("high")) is not exp.Placeholder:
                    raise SqlAstRejected("BETWEEN values must be bound parameters")
            else:
                raise SqlAstRejected("WHERE contains an unreviewed predicate")

    @classmethod
    def _validate_group_order(
        cls,
        tree: exp.Select,
        alias_to_table: dict[str, str],
        manifest: SemanticProjectionManifest,
    ) -> tuple[tuple[ProjectionExpression, ...], tuple[ProjectionOrder, ...]]:
        group = tree.args.get("group")
        actual_groups: list[ProjectionExpression] = []
        for expression in group.expressions if group is not None else ():
            if type(expression) is exp.Column:
                actual_groups.append(cls._projection_expression(expression, alias_to_table))
            elif cls._is_month_expression(expression, alias_to_table):
                actual_groups.append(
                    cls._projection_expression(
                        next(expression.find_all(exp.Column)), alias_to_table, transform="month"
                    )
                )
            else:
                raise SqlAstRejected("GROUP BY expression is not reviewed")
        if tuple(actual_groups) != manifest.group_by:
            raise SqlAstRejected("GROUP BY diverges from the typed plan projection manifest")
        order = tree.args.get("order")
        actual_orders: list[ProjectionOrder] = []
        for ordered in order.expressions if order is not None else ():
            if type(ordered) is not exp.Ordered or type(ordered.this) is not exp.Column or ordered.this.table:
                raise SqlAstRejected("ORDER BY expression is not reviewed")
            actual_orders.append(
                ProjectionOrder(alias=ordered.this.name, direction="desc" if ordered.args.get("desc") else "asc")
            )
        if tuple(actual_orders) != manifest.order_by:
            raise SqlAstRejected("ORDER BY diverges from the typed plan projection manifest")
        return tuple(actual_groups), tuple(actual_orders)

    @staticmethod
    def _validate_limit(tree: exp.Select, registry: RegistrySnapshot) -> int:
        limit = tree.args.get("limit")
        if type(limit) is not exp.Limit or type(limit.expression) is not exp.Literal or limit.expression.is_string:
            raise SqlAstRejected("a literal bounded LIMIT is required")
        try:
            value = int(limit.expression.this)
        except (TypeError, ValueError) as error:
            raise SqlAstRejected("LIMIT must be an integer") from error
        if value < 1 or value > registry.max_result_rows:
            raise SqlAstRejected("LIMIT exceeds the registry maximum")
        return value

    @staticmethod
    def _validate_parameters(tree: exp.Select, params: Mapping[str, SqlParameter]) -> None:
        names = [placeholder.name for placeholder in tree.find_all(exp.Placeholder)]
        if len(names) != len(set(names)):
            raise SqlAstRejected("bound parameter names must be unique")
        if set(names) != set(params):
            if any(name.startswith("policy_") for name in set(names) ^ set(params)):
                raise SqlPolicyDenied("policy parameters must match placeholders exactly")
            raise SqlAstRejected("bound parameters must match placeholders exactly")
        if any(type(value) not in (str, int, bool, date, Decimal) for value in params.values()):
            raise SqlAstRejected("bound parameter type is not allowed")
        for literal in tree.find_all(exp.Literal):
            if literal.parent and isinstance(literal.parent, exp.Limit):
                continue
            if literal.is_string and literal.this == "%Y-%m" and isinstance(literal.parent, exp.TimeToStr):
                continue
            raise SqlAstRejected("caller values must not appear as SQL literals")

    def _validate_policy(
        self, tree: exp.Select, params: Mapping[str, SqlParameter], alias_to_table: dict[str, str]
    ) -> None:
        expected = {
            "policy_is_synthetic": self.scope.synthetic,
            "policy_start_date": self.scope.start_date,
            "policy_end_date": self.scope.end_date,
        }
        if any(params.get(name) != value for name, value in expected.items()):
            raise SqlPolicyDenied("policy parameter does not match configured scope")
        predicates = tuple(self._flatten_and(tree.args["where"].this))
        synthetic_ok = any(
            type(item) is exp.EQ
            and type(item.this) is exp.Column
            and self._physical(item.this, alias_to_table) == "data_sources.is_synthetic"
            and type(item.expression) is exp.Placeholder
            and item.expression.name == "policy_is_synthetic"
            for item in predicates
        )
        time_ok = any(
            type(item) is exp.Between
            and type(item.this) is exp.Column
            and self._physical(item.this, alias_to_table) == "trade_records.trade_date"
            and type(item.args.get("low")) is exp.Placeholder
            and item.args["low"].name == "policy_start_date"
            and type(item.args.get("high")) is exp.Placeholder
            and item.args["high"].name == "policy_end_date"
            for item in predicates
        )
        if not synthetic_ok or not time_ok:
            raise SqlPolicyDenied("mandatory synthetic and trade-date scope is missing")

    def _validate_effective_time(
        self, tree: exp.Select, rendered: RenderedSql, alias_to_table: dict[str, str]
    ) -> None:
        start, end = self.scope.start_date, self.scope.end_date
        for predicate in self._flatten_and(tree.args["where"].this):
            if (
                type(predicate) is exp.Between
                and type(predicate.this) is exp.Column
                and self._physical(predicate.this, alias_to_table) == "trade_records.trade_date"
                and type(predicate.args.get("low")) is exp.Placeholder
                and type(predicate.args.get("high")) is exp.Placeholder
            ):
                low = self._bound_date(rendered.params[predicate.args["low"].name])
                high = self._bound_date(rendered.params[predicate.args["high"].name])
                if low > high:
                    raise SqlPolicyDenied("trade-date filter is not ordered")
                start, end = max(start, low), min(end, high)
        if start > end:
            raise SqlPolicyDenied("query time range does not intersect the configured data scope")
        if rendered.effective_start_date != start or rendered.effective_end_date != end:
            raise SqlPolicyDenied("effective time metadata diverges from bound SQL predicates")

    @staticmethod
    def _bound_date(value: SqlParameter) -> date:
        if type(value) is date:
            return value
        if type(value) is str:
            try:
                return date.fromisoformat(value)
            except ValueError as error:
                raise SqlPolicyDenied("trade-date parameter must be an ISO date") from error
        raise SqlPolicyDenied("trade-date parameter must be a date")

    @staticmethod
    def _flatten_and(expression: exp.Expression) -> Iterable[exp.Expression]:
        if type(expression) is exp.And:
            yield from SqlValidator._flatten_and(expression.this)
            yield from SqlValidator._flatten_and(expression.expression)
        else:
            yield expression

    @staticmethod
    def _physical(column: exp.Column, alias_to_table: dict[str, str]) -> str:
        if not column.table or column.table not in alias_to_table:
            raise SqlAstRejected("column must use a known table alias")
        return f"{alias_to_table[column.table]}.{column.name}"

    @staticmethod
    def _physical_sql_name(value: str, alias_to_table: dict[str, str]) -> str:
        pieces = value.split(".")
        if len(pieces) != 2 or pieces[0] not in alias_to_table:
            raise SqlAstRejected("planned join uses an unknown alias")
        return f"{alias_to_table[pieces[0]]}.{pieces[1]}"

    @classmethod
    def _registered_column(
        cls, column: exp.Column, alias_to_table: dict[str, str], registry: RegistrySnapshot
    ) -> str:
        physical = cls._physical(column, alias_to_table)
        if physical not in registry.columns or physical in registry.sensitive_fields:
            raise SqlAstRejected("column is not registered for SQL output")
        return physical

    @staticmethod
    def _projection_expression(
        column: exp.Column,
        alias_to_table: dict[str, str],
        *,
        transform: Literal["identity", "month"] = "identity",
    ) -> ProjectionExpression:
        if not column.table or column.table not in alias_to_table:
            raise SqlAstRejected("projection column must use a known alias")
        return ProjectionExpression(
            table_alias=column.table,
            table=alias_to_table[column.table],
            column=column.name,
            transform=transform,
        )

    @classmethod
    def _validate_manifest_semantics(
        cls,
        manifest: SemanticProjectionManifest,
        plan: SqlQueryPlan,
        registry: RegistrySnapshot,
    ) -> None:
        metrics = tuple(item for item in manifest.projections if item.operation is not None)
        if len(metrics) != 1 or len(plan.aggregations) != 1:
            raise SqlAstRejected("projection requires exactly one reviewed aggregate")
        dimensions = tuple(
            ProjectionExpression(**item.model_dump(exclude={"alias", "operation"}))
            for item in manifest.projections
            if item.operation is None
        )
        if dimensions != manifest.group_by:
            raise SqlAstRejected("all and only projected dimensions must define aggregation grain")
        definition = registry.aggregations.get(metrics[0].alias)
        if definition is None:
            raise SqlAstRejected("projection metric is not registered")
        for item in (projection for projection in manifest.projections if projection.operation is None):
            expected_alias = cls._reviewed_dimension_alias(item, plan, registry, definition)
            if item.alias != expected_alias:
                raise SqlAstRejected("dimension semantic name and output alias do not match")
        grouped_physical = {f"{item.table}.{item.column}" for item in manifest.group_by}
        for key in ("currency_column", "unit_column"):
            if key in definition and str(definition[key]) not in grouped_physical:
                raise SqlAstRejected("projection omits the registered metric unit or currency grain")
        time_items = [
            item for item in manifest.group_by if f"{item.table}.{item.column}" == "trade_records.trade_date"
        ]
        if manifest.time_grain == "month":
            if len(time_items) != 1 or time_items[0].transform != "month":
                raise SqlAstRejected("monthly time grain requires one month-bucketed trade date")
        elif manifest.time_grain == "day":
            if len(time_items) != 1 or time_items[0].transform != "identity":
                raise SqlAstRejected("daily time grain requires one trade-date dimension")
        elif any(item.transform == "month" for item in manifest.group_by):
            raise SqlAstRejected("total time grain cannot contain a month transform")

    @staticmethod
    def _reviewed_dimension_alias(
        item: ProjectionItem,
        plan: SqlQueryPlan,
        registry: RegistrySnapshot,
        aggregate_definition: Mapping[str, object],
    ) -> str:
        physical = f"{item.table}.{item.column}"
        if physical == aggregate_definition.get("currency_column"):
            return "currency"
        if physical == aggregate_definition.get("unit_column"):
            return "unit"
        if physical == registry.dimensions.get("time"):
            return "trade_date"

        candidates = {
            name
            for name, registered in registry.dimensions.items()
            if registered == physical and name != "time"
        }
        roles: set[str] = set()
        joins_by_name = {join.name: join for join in registry.joins}
        for planned_join in plan.joins:
            endpoint_aliases = {
                planned_join.left.split(".", maxsplit=1)[0],
                planned_join.right.split(".", maxsplit=1)[0],
            }
            if item.table_alias in endpoint_aliases:
                registered_join = joins_by_name.get(planned_join.name)
                if registered_join is not None:
                    roles.update(registered_join.roles)
        role_matches = candidates & roles
        if len(role_matches) == 1:
            return next(iter(role_matches))
        if len(candidates) == 1:
            return next(iter(candidates))
        raise SqlAstRejected("dimension has no unique reviewed semantic alias")

    @staticmethod
    def _derive_time_grain(
        projections: tuple[ProjectionItem, ...],
        groups: tuple[ProjectionExpression, ...],
        registry: RegistrySnapshot,
    ) -> Literal["total", "day", "month"]:
        time_column = registry.dimensions.get("time")
        if time_column is None:
            raise SqlAstRejected("registry has no reviewed time dimension")
        time_projections = tuple(
            item
            for item in projections
            if item.operation is None and f"{item.table}.{item.column}" == time_column
        )
        time_groups = tuple(item for item in groups if f"{item.table}.{item.column}" == time_column)
        if not time_projections and not time_groups:
            return "total"
        if (
            len(time_projections) != 1
            or len(time_groups) != 1
            or time_projections[0].alias != "trade_date"
            or ProjectionExpression(
                **time_projections[0].model_dump(exclude={"alias", "operation"})
            )
            != time_groups[0]
        ):
            raise SqlAstRejected("time projection and group require one reviewed trade_date alias")
        return "month" if time_projections[0].transform == "month" else "day"

    @staticmethod
    def _aggregation_grain(manifest: SemanticProjectionManifest) -> tuple[str, ...]:
        aliases: list[str] = []
        for group in manifest.group_by:
            matches = [
                item.alias
                for item in manifest.projections
                if item.operation is None
                and item.table_alias == group.table_alias
                and item.table == group.table
                and item.column == group.column
                and item.transform == group.transform
            ]
            if len(matches) != 1:
                raise SqlAstRejected("aggregation grain does not map to one projected dimension")
            aliases.append(matches[0])
        return tuple(aliases)

    @staticmethod
    def _is_month_expression(expression: exp.Expression, alias_to_table: dict[str, str]) -> bool:
        if type(expression) is not exp.TimeToStr:
            return False
        literals = tuple(expression.find_all(exp.Literal))
        columns = tuple(expression.find_all(exp.Column))
        return (
            len(literals) == 1
            and literals[0].is_string
            and literals[0].this == "%Y-%m"
            and len(columns) == 1
            and columns[0].table in alias_to_table
        )
