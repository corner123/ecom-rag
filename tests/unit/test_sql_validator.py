from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import MappingProxyType

import pytest

from trade_agent.agents.intent import IntentParser
from trade_agent.db.registry import RegistryJoin, RegistrySnapshot
from trade_agent.db.sql_planner import SelectColumn, SqlPlanner
from trade_agent.db.sql_renderer import RenderedSql, SqlDataScope, SqlRenderer
from trade_agent.db.sql_validator import SqlAstRejected, SqlPolicyDenied, SqlValidator


AS_OF = date(2026, 9, 4)


@pytest.fixture
def registry() -> RegistrySnapshot:
    return RegistrySnapshot(
        fingerprint="reviewed-fingerprint",
        tables=("companies", "countries", "data_sources", "hs_codes", "trade_records"),
        columns=(
            "companies.id",
            "companies.company_name",
            "companies.is_synthetic",
            "countries.id",
            "countries.country_code",
            "countries.region",
            "data_sources.id",
            "data_sources.is_synthetic",
            "hs_codes.id",
            "hs_codes.hs_code",
            "trade_records.id",
            "trade_records.raw_record_id",
            "trade_records.source_id",
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
            RegistryJoin("fk_trade_records_source", "trade_records.source_id", "data_sources.id", ()),
        ),
        aliases=MappingProxyType({}),
        identifiers=MappingProxyType(
            {"import_country": "countries.country_code", "export_country": "countries.country_code", "hs_code": "hs_codes.hs_code"}
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
                "import_country": "countries.country_code",
                "hs_code": "hs_codes.hs_code",
                "region": "countries.region",
                "time": "trade_records.trade_date",
            }
        ),
        filters=MappingProxyType(
            {
                "importer_company": "companies.company_name",
                "exporter_company": "companies.company_name",
                "import_country": "countries.country_code",
                "hs_code": "hs_codes.hs_code",
                "region": "countries.region",
                "time": "trade_records.trade_date",
            }
        ),
        sensitive_fields=(),
        max_result_rows=50,
    )


@pytest.fixture
def scope() -> SqlDataScope:
    return SqlDataScope(
        dataset_id="synthetic-demo-v1",
        synthetic=True,
        start_date=date(2025, 3, 1),
        end_date=date(2026, 8, 31),
    )


@pytest.fixture
def rendered(registry: RegistrySnapshot, scope: SqlDataScope) -> RenderedSql:
    intent = IntentParser(as_of=AS_OF).parse("最近半年美国采购 HS850440 金额最高的 10 家公司")
    plan = SqlPlanner().plan(intent, registry)
    return SqlRenderer(scope=scope).render(plan)


def test_renderer_emits_bound_values_and_required_data_policy(rendered: RenderedSql) -> None:
    assert "US" not in rendered.sql
    assert "850440" not in rendered.sql
    assert "2026-03-04" not in rendered.sql
    assert set(rendered.params) >= {
        "filter_0_0",
        "filter_1_0",
        "filter_2_start",
        "filter_2_end",
        "policy_is_synthetic",
        "policy_start_date",
        "policy_end_date",
    }
    assert "JOIN data_sources AS data_scope" in rendered.sql
    assert "data_scope.is_synthetic = :policy_is_synthetic" in rendered.sql
    assert "tr.trade_date BETWEEN :policy_start_date AND :policy_end_date" in rendered.sql


def test_validator_accepts_renderer_output_and_preserves_only_filter_names(
    registry: RegistrySnapshot, rendered: RenderedSql, scope: SqlDataScope
) -> None:
    validated = SqlValidator(scope=scope).validate(rendered, registry)
    assert validated.schema_fingerprint == registry.fingerprint
    assert validated.limit == 10
    assert "US" not in validated.sql
    assert validated.bound_filter_names == tuple(sorted(rendered.params))


def test_rendered_parameters_are_immutable(rendered: RenderedSql) -> None:
    with pytest.raises(TypeError):
        rendered.params["filter_0_0"] = "CN"  # type: ignore[index]


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE companies",
        "SELECT 1; DELETE FROM trade_records",
        "SELECT * FROM mysql.user",
        "SELECT * FROM companies",
        "SELECT tr.trade_amount FROM trade_records AS tr LIMIT 10",
        "SELECT SUM(tr.trade_amount) FROM trade_records AS tr JOIN companies AS c ON 1 = 1 LIMIT 10",
        "SELECT SUM(tr.trade_amount) FROM trade_records AS tr JOIN companies AS c ON tr.exporter_id = c.id LIMIT 10",
        "SELECT SUM(unknown.trade_amount) FROM trade_records AS tr LIMIT 10",
        "SELECT SUM(tr.not_registered) FROM trade_records AS tr LIMIT 10",
        "SELECT AVG(tr.trade_amount) FROM trade_records AS tr LIMIT 10",
        "SELECT SUM((SELECT trade_amount FROM trade_records LIMIT 1)) FROM trade_records AS tr LIMIT 10",
        "SELECT SUM(tr.trade_amount) FROM trade_records AS tr UNION SELECT SUM(tr.trade_amount) FROM trade_records AS tr LIMIT 10",
        "WITH x AS (SELECT * FROM trade_records) SELECT * FROM x LIMIT 10",
        "SELECT /*+ MAX_EXECUTION_TIME(999999) */ SUM(tr.trade_amount) FROM trade_records AS tr LIMIT 10",
        "SELECT SUM(tr.trade_amount) FROM trade_records AS tr -- bypass\nLIMIT 10",
    ],
)
def test_unsafe_sql_is_rejected(
    registry: RegistrySnapshot, rendered: RenderedSql, scope: SqlDataScope, sql: str
) -> None:
    candidate = rendered.model_copy(update={"sql": sql, "params": {}})
    with pytest.raises(SqlAstRejected):
        SqlValidator(scope=scope).validate(candidate, registry)


def test_validator_rejects_missing_or_oversized_limit(
    registry: RegistrySnapshot, rendered: RenderedSql, scope: SqlDataScope
) -> None:
    for sql in (rendered.sql.replace(" LIMIT 10", ""), rendered.sql.replace(" LIMIT 10", " LIMIT 51")):
        with pytest.raises(SqlAstRejected):
            SqlValidator(scope=scope).validate(rendered.model_copy(update={"sql": sql}), registry)


def test_validator_denies_tenant_scope_without_reviewed_mapping(
    registry: RegistrySnapshot, rendered: RenderedSql, scope: SqlDataScope
) -> None:
    tenant_rendered = rendered.model_copy(update={"tenant_scope": "tenant-acme"})
    with pytest.raises(SqlPolicyDenied, match="tenant"):
        SqlValidator(scope=scope).validate(tenant_rendered, registry)


def test_validator_denies_policy_weakening_and_schema_drift(
    registry: RegistrySnapshot, rendered: RenderedSql, scope: SqlDataScope
) -> None:
    weakened = rendered.sql.replace("data_scope.is_synthetic = :policy_is_synthetic AND ", "")
    with pytest.raises(SqlPolicyDenied):
        SqlValidator(scope=scope).validate(rendered.model_copy(update={"sql": weakened}), registry)

    with pytest.raises(SqlPolicyDenied):
        SqlValidator(scope=scope).validate(rendered, replace(registry, fingerprint="changed"))


def test_validator_recomputes_effective_time_instead_of_trusting_metadata(
    registry: RegistrySnapshot, rendered: RenderedSql, scope: SqlDataScope
) -> None:
    forged = rendered.model_copy(update={"effective_start_date": scope.start_date})
    with pytest.raises(SqlPolicyDenied, match="effective time"):
        SqlValidator(scope=scope).validate(forged, registry)


def test_validator_rejects_literal_or_unmatched_parameters(
    registry: RegistrySnapshot, rendered: RenderedSql, scope: SqlDataScope
) -> None:
    literal = rendered.sql.replace(":filter_0_0", "'US'")
    with pytest.raises(SqlAstRejected):
        SqlValidator(scope=scope).validate(rendered.model_copy(update={"sql": literal}), registry)

    with pytest.raises(SqlAstRejected):
        SqlValidator(scope=scope).validate(rendered.model_copy(update={"params": {**rendered.params, "unused": "secret"}}), registry)


@pytest.mark.parametrize(
    "sql",
    [
        (
            "SELECT tr.raw_record_id AS importer_company, tr.currency AS currency, "
            "SUM(tr.trade_amount) AS trade_amount"
        ),
        (
            "SELECT importer.company_name AS importer_company, tr.currency AS currency, "
            "SUM(tr.quantity) AS trade_amount"
        ),
        (
            "SELECT importer.company_name AS importer_company, "
            "SUM(tr.trade_amount) AS trade_amount"
        ),
        (
            "SELECT importer.company_name AS importer_company, tr.currency AS currency, "
            "SUM(tr.trade_amount) AS trade_amount, tr.raw_record_id AS raw_record_id"
        ),
    ],
)
def test_validator_rejects_projection_that_differs_from_the_plan_manifest(
    registry: RegistrySnapshot, rendered: RenderedSql, scope: SqlDataScope, sql: str
) -> None:
    suffix = rendered.sql.split(" FROM trade_records", maxsplit=1)[1]
    candidate_sql = f"{sql} FROM trade_records{suffix}"
    if "raw_record_id" in candidate_sql:
        candidate_sql = candidate_sql.replace(
            "GROUP BY importer.company_name, tr.currency", "GROUP BY tr.raw_record_id, tr.currency"
        )
    if "tr.currency AS currency" not in candidate_sql:
        candidate_sql = candidate_sql.replace(" GROUP BY importer.company_name, tr.currency", " GROUP BY importer.company_name")
    with pytest.raises(SqlAstRejected, match="projection"):
        SqlValidator(scope=scope).validate(rendered.model_copy(update={"sql": candidate_sql}), registry)


def test_validated_semantics_are_derived_instead_of_trusting_rendered_metadata(
    registry: RegistrySnapshot, rendered: RenderedSql, scope: SqlDataScope
) -> None:
    forged = rendered.model_copy(
        update={
            "metric_names": ("quantity",),
            "aggregation_grain": ("tr.raw_record_id",),
            "time_grain": "month",
        }
    )
    validated = SqlValidator(scope=scope).validate(forged, registry)
    assert validated.metric_names == ("trade_amount",)
    assert validated.aggregation_grain == ("importer_company", "currency")
    assert validated.time_grain == "total"


def test_missing_date_parameter_is_a_typed_rejection_not_key_error(
    registry: RegistrySnapshot, rendered: RenderedSql, scope: SqlDataScope
) -> None:
    params = dict(rendered.params)
    params.pop("filter_2_start")
    with pytest.raises(SqlAstRejected, match="parameters"):
        SqlValidator(scope=scope).validate(rendered.model_copy(update={"params": params}), registry)


def test_validator_rejects_plan_that_swaps_reviewed_dimension_aliases(
    registry: RegistrySnapshot, rendered: RenderedSql, scope: SqlDataScope
) -> None:
    assert rendered.plan is not None
    columns = tuple(
        column.model_copy(
            update={
                "alias": {
                    "importer_company": "currency",
                    "currency": "importer_company",
                }.get(column.alias, column.alias)
            }
        )
        for column in rendered.plan.columns
    )
    candidate = SqlRenderer(scope=scope).render(rendered.plan.model_copy(update={"columns": columns}))

    with pytest.raises(SqlAstRejected, match="semantic|alias|dimension"):
        SqlValidator(scope=scope).validate(candidate, registry)


def test_validator_rejects_total_grain_that_projects_raw_trade_date(
    registry: RegistrySnapshot, rendered: RenderedSql, scope: SqlDataScope
) -> None:
    assert rendered.plan is not None
    plan = rendered.plan.model_copy(
        update={
            "columns": (
                SelectColumn(expression="tr.trade_date", alias="trade_date"),
                *rendered.plan.columns,
            ),
            "group_by": ("tr.trade_date", *rendered.plan.group_by),
            "time_grain": "total",
        }
    )
    candidate = SqlRenderer(scope=scope).render(plan)

    with pytest.raises(SqlAstRejected, match="time|total"):
        SqlValidator(scope=scope).validate(candidate, registry)


def test_validator_rejects_month_grain_with_unreviewed_time_alias(
    registry: RegistrySnapshot, rendered: RenderedSql, scope: SqlDataScope
) -> None:
    assert rendered.plan is not None
    plan = rendered.plan.model_copy(
        update={
            "columns": (
                SelectColumn(expression="tr.trade_date", alias="calendar_month"),
                *rendered.plan.columns,
            ),
            "group_by": ("tr.trade_date", *rendered.plan.group_by),
            "time_grain": "month",
        }
    )
    candidate = SqlRenderer(scope=scope).render(plan)

    with pytest.raises(SqlAstRejected, match="time|alias|dimension"):
        SqlValidator(scope=scope).validate(candidate, registry)


def test_renderer_rejects_plan_without_the_one_required_aggregate(
    rendered: RenderedSql, scope: SqlDataScope
) -> None:
    assert rendered.plan is not None
    plan = rendered.plan.model_copy(
        update={
            "columns": tuple(
                column for column in rendered.plan.columns if column.alias != "trade_amount"
            ),
            "aggregations": (),
            "order_by": (),
        }
    )

    with pytest.raises(ValueError, match="aggregate"):
        SqlRenderer(scope=scope).render(plan)


def test_validator_returns_typed_ast_rejection_for_forged_zero_aggregate_plan(
    registry: RegistrySnapshot, rendered: RenderedSql, scope: SqlDataScope
) -> None:
    assert rendered.plan is not None
    forged_plan = rendered.plan.model_copy(update={"aggregations": ()})

    with pytest.raises(SqlAstRejected, match="aggregate"):
        SqlValidator(scope=scope).validate(
            rendered.model_copy(update={"plan": forged_plan}), registry
        )
