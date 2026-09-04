from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from types import MappingProxyType

import pytest

from trade_agent.db.registry import RegistryJoin, RegistrySnapshot


AS_OF = date(2026, 9, 4)


@pytest.fixture
def registry() -> RegistrySnapshot:
    return RegistrySnapshot(
        fingerprint="reviewed-fingerprint",
        tables=("companies", "countries", "hs_codes", "trade_records"),
        columns=(
            "companies.id",
            "companies.company_name",
            "countries.id",
            "countries.country_code",
            "countries.region",
            "hs_codes.id",
            "hs_codes.hs_code",
            "trade_records.id",
            "trade_records.importer_id",
            "trade_records.exporter_id",
            "trade_records.import_country_id",
            "trade_records.export_country_id",
            "trade_records.hs_code_id",
            "trade_records.trade_date",
            "trade_records.trade_amount",
            "trade_records.currency",
            "trade_records.quantity",
            "trade_records.unit",
        ),
        joins=(
            RegistryJoin("fk_trade_records_importer", "trade_records.importer_id", "companies.id", ("importer_company",)),
            RegistryJoin("fk_trade_records_exporter", "trade_records.exporter_id", "companies.id", ("exporter_company",)),
            RegistryJoin("fk_trade_records_import_country", "trade_records.import_country_id", "countries.id", ("import_country",)),
            RegistryJoin("fk_trade_records_export_country", "trade_records.export_country_id", "countries.id", ("export_country",)),
            RegistryJoin("fk_trade_records_hs_code", "trade_records.hs_code_id", "hs_codes.id", ("hs_code",)),
        ),
        aliases=MappingProxyType({}),
        identifiers=MappingProxyType(
            {
                "import_country": "countries.country_code",
                "export_country": "countries.country_code",
                "hs_code": "hs_codes.hs_code",
            }
        ),
        aggregations=MappingProxyType(
            {
                "trade_amount": MappingProxyType(
                    {"column": "trade_records.trade_amount", "operations": ("sum",), "currency_column": "trade_records.currency"}
                ),
                "quantity": MappingProxyType(
                    {"column": "trade_records.quantity", "operations": ("sum",), "unit_column": "trade_records.unit"}
                ),
            }
        ),
        dimensions=MappingProxyType(
            {
                "importer_company": "companies.company_name",
                "exporter_company": "companies.company_name",
                "import_country": "countries.country_name",
                "export_country": "countries.country_name",
                "hs_code": "hs_codes.hs_code",
                "region": "countries.region",
                "time": "trade_records.trade_date",
            }
        ),
        filters=MappingProxyType(
            {
                "importer_company": "companies.company_name",
                "exporter_company": "companies.company_name",
                "import_country": "countries.country_name",
                "export_country": "countries.country_name",
                "hs_code": "hs_codes.hs_code",
                "region": "countries.region",
                "time": "trade_records.trade_date",
            }
        ),
        sensitive_fields=(),
        max_result_rows=50,
    )


def test_top_importers_plan_uses_role_specific_joins_iso_country_and_currency_grain(
    registry: RegistrySnapshot,
) -> None:
    from trade_agent.agents.intent import IntentParser
    from trade_agent.db.sql_planner import SqlPlanner

    intent = IntentParser(as_of=AS_OF).parse("最近半年美国采购 HS850440 金额最高的 10 家公司")
    plan = SqlPlanner().plan(intent, registry)

    assert plan.schema_fingerprint == "reviewed-fingerprint"
    assert [(table.table, table.alias) for table in plan.tables] == [
        ("trade_records", "tr"),
        ("companies", "importer"),
        ("countries", "import_country"),
        ("hs_codes", "hs"),
    ]
    assert [(join.name, join.left, join.right) for join in plan.joins] == [
        ("fk_trade_records_importer", "tr.importer_id", "importer.id"),
        ("fk_trade_records_import_country", "tr.import_country_id", "import_country.id"),
        ("fk_trade_records_hs_code", "tr.hs_code_id", "hs.id"),
    ]
    assert ("import_country.country_code", "in", ("US",)) in {
        (predicate.column, predicate.operator, predicate.values) for predicate in plan.predicates
    }
    assert ("tr.trade_date", "between", ("2026-03-04", "2026-09-04")) in {
        (predicate.column, predicate.operator, predicate.values) for predicate in plan.predicates
    }
    assert plan.group_by == ("importer.company_name", "tr.currency")
    assert plan.aggregations[0].metric == "trade_amount"
    assert plan.aggregations[0].column == "tr.trade_amount"
    assert plan.aggregations[0].currency_column == "tr.currency"
    assert plan.aggregations[0].grain == ("importer.company_name", "tr.currency")
    assert plan.order_by[0].expression == "trade_amount"
    assert plan.order_by[0].direction == "desc"
    assert plan.limit == 10


def test_monthly_export_quantity_plan_keeps_export_country_alias_unit_and_time_grain(
    registry: RegistrySnapshot,
) -> None:
    from trade_agent.agents.intent import IntentParser
    from trade_agent.db.sql_planner import SqlPlanner

    plan = SqlPlanner().plan(IntentParser(as_of=AS_OF).parse("中国出口 HS850440 的月度数量"), registry)

    assert [(table.table, table.alias) for table in plan.tables] == [
        ("trade_records", "tr"),
        ("companies", "exporter"),
        ("countries", "export_country"),
        ("hs_codes", "hs"),
    ]
    assert ("fk_trade_records_exporter", "tr.exporter_id", "exporter.id") in {
        (join.name, join.left, join.right) for join in plan.joins
    }
    assert ("fk_trade_records_export_country", "tr.export_country_id", "export_country.id") in {
        (join.name, join.left, join.right) for join in plan.joins
    }
    assert ("export_country.country_code", "in", ("CN",)) in {
        (predicate.column, predicate.operator, predicate.values) for predicate in plan.predicates
    }
    assert plan.group_by == ("tr.trade_date", "tr.unit")
    assert plan.aggregations[0].metric == "quantity"
    assert plan.aggregations[0].unit_column == "tr.unit"
    assert plan.aggregations[0].grain == ("tr.trade_date", "tr.unit")


def test_planner_rejects_out_of_scope_and_unregistered_business_fields(registry: RegistrySnapshot) -> None:
    from trade_agent.agents.intent import IntentParser, QueryIntent
    from trade_agent.db.contracts import QueryConstraints
    from trade_agent.db.sql_planner import OutOfScopeRequest, SchemaNotRegistered, SqlPlanner

    planner = SqlPlanner()
    with pytest.raises(OutOfScopeRequest):
        planner.plan(IntentParser(as_of=AS_OF).parse("删除所有贸易记录"), registry)
    with pytest.raises(SchemaNotRegistered, match="profit_margin"):
        planner.plan(
            QueryIntent(
                question="显示利润率",
                kind="company_trend",
                constraints=QueryConstraints(metrics=["profit_margin"], dimensions=["importer_company"], filters=[]),
                need_trade_data=True,
            ),
            registry,
        )


def test_structured_sql_plan_rejects_unknown_schema_elements(registry: RegistrySnapshot) -> None:
    from trade_agent.agents.intent import IntentParser
    from trade_agent.db.sql_planner import StructuredPlanRejected, SqlPlanner

    intent = IntentParser(as_of=AS_OF).parse("最近半年美国采购 HS850440 金额最高的 10 家公司")
    planner = SqlPlanner(structured_provider=lambda _: {"tables": [], "unreviewed": True})

    with pytest.raises(StructuredPlanRejected, match="unknown schema"):
        planner.plan(intent, registry)


def test_planner_compiles_reviewed_region_and_entity_binding_or_rejects_unresolved_entity(
    registry: RegistrySnapshot,
) -> None:
    from trade_agent.agents.intent import IntentParser
    from trade_agent.db.sql_planner import SqlPlanner, UnresolvedSqlConstraint
    from trade_agent.retrieval.filters import RetrievalFilter

    intent = IntentParser(as_of=AS_OF).parse(
        "最近半年美国采购 HS850440 金额最高的 10 家公司",
        explicit_filters=RetrievalFilter(region="Europe", entity_ids=["company:buyer-7"]),
    )

    with pytest.raises(UnresolvedSqlConstraint, match="entity_ids"):
        SqlPlanner().plan(intent, registry)

    plan = SqlPlanner(entity_bindings={"company:buyer-7": 7}).plan(intent, registry)
    predicates = {(item.column, item.operator, item.values) for item in plan.predicates}
    assert ("import_country.region", "equals", ("Europe",)) in predicates
    assert ("importer.id", "in", ("7",)) in predicates


def test_entity_binding_uses_company_primary_key_even_when_names_are_not_unique(registry: RegistrySnapshot) -> None:
    from trade_agent.agents.intent import IntentParser
    from trade_agent.db.sql_planner import SqlPlanner
    from trade_agent.retrieval.filters import RetrievalFilter

    intent = IntentParser(as_of=AS_OF).parse(
        "最近半年美国采购 HS850440 金额最高的 10 家公司",
        explicit_filters=RetrievalFilter(entity_ids=["company:left", "company:right"]),
    )

    plan = SqlPlanner(entity_bindings={"company:left": 17, "company:right": 29}).plan(intent, registry)

    assert ("importer.id", "in", ("17", "29")) in {
        (item.column, item.operator, item.values) for item in plan.predicates
    }


def test_planner_leaves_publication_window_for_rag_and_does_not_convert_it_to_trade_date(
    registry: RegistrySnapshot,
) -> None:
    from trade_agent.agents.intent import IntentParser
    from trade_agent.db.sql_planner import SqlPlanner
    from trade_agent.retrieval.filters import RetrievalFilter

    published_after = datetime(2026, 1, 1, tzinfo=timezone.utc)
    intent = IntentParser(as_of=AS_OF).parse(
        "美国采购 HS850440 金额最高的 10 家公司",
        explicit_filters=RetrievalFilter(published_after=published_after),
    )
    plan = SqlPlanner().plan(intent, registry)

    assert plan.rag_filter.published_after == published_after
    assert all(predicate.column != "tr.trade_date" for predicate in plan.predicates)


def test_planner_rejects_multi_metric_or_dimension_shape_instead_of_discarding_fields(
    registry: RegistrySnapshot,
) -> None:
    from trade_agent.agents.intent import QueryIntent
    from trade_agent.db.contracts import QueryConstraints
    from trade_agent.db.sql_planner import UnsupportedQueryShape, SqlPlanner

    intent = QueryIntent(
        question="two metrics",
        kind="top_importers",
        constraints=QueryConstraints(
            metrics=["trade_amount", "quantity"],
            dimensions=["importer_company", "hs_code"],
            filters=[],
        ),
        need_trade_data=True,
    )

    with pytest.raises(UnsupportedQueryShape, match="multiple metrics"):
        SqlPlanner().plan(intent, registry)


def test_planner_rejects_a_snapshot_missing_a_referenced_physical_column(registry: RegistrySnapshot) -> None:
    from trade_agent.agents.intent import IntentParser
    from trade_agent.db.sql_planner import SchemaNotRegistered, SqlPlanner

    missing_company_name = replace(
        registry,
        columns=tuple(column for column in registry.columns if column != "companies.company_name"),
    )

    with pytest.raises(SchemaNotRegistered, match="companies.company_name"):
        SqlPlanner().plan(
            IntentParser(as_of=AS_OF).parse("最近半年美国采购 HS850440 金额最高的 10 家公司"),
            missing_company_name,
        )
