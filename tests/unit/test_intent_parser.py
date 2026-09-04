from __future__ import annotations

from datetime import date

import pytest

from trade_agent.retrieval.filters import RetrievalFilter


AS_OF = date(2026, 9, 4)


def test_parser_preserves_top_importer_constraints_and_binds_relative_date() -> None:
    from trade_agent.agents.intent import IntentParser

    intent = IntentParser(as_of=AS_OF).parse("最近半年美国采购 HS850440 金额最高的 10 家公司")

    assert intent.kind == "top_importers"
    assert intent.need_trade_data is True
    assert intent.need_external_intel is False
    assert intent.filters.country_codes == ("US",)
    assert intent.filters.hs_codes == ("850440",)
    assert intent.filters.company_role == "importer_company"
    assert intent.constraints.metrics == ["trade_amount"]
    assert intent.constraints.dimensions == ["importer_company"]
    assert intent.constraints.filters == ["import_country", "hs_code", "time"]
    assert intent.time_range.start == date(2026, 3, 4)
    assert intent.time_range.end == AS_OF
    assert intent.time_range.grain == "total"
    assert intent.limit == 10


def test_parser_keeps_explicit_retrieval_filters_as_mandatory_constraints() -> None:
    from trade_agent.agents.intent import IntentParser

    intent = IntentParser(as_of=AS_OF).parse(
        "ABC 是否值得跟进",
        explicit_filters=RetrievalFilter(country_codes=["DE"], hs_codes=["850440"], entity_ids=["company:abc"]),
    )

    assert intent.kind == "lead_assessment"
    assert intent.need_trade_data is True
    assert intent.need_external_intel is True
    assert intent.filters.country_codes == ("DE",)
    assert intent.filters.hs_codes == ("850440",)
    assert intent.filters.entity_ids == ("company:abc",)


def test_parser_adapts_business_intent_to_the_existing_strict_retrieval_intent() -> None:
    from trade_agent.agents.intent import IntentParser, to_retrieval_query_intent
    from trade_agent.retrieval.planner import QueryIntent as RetrievalQueryIntent

    intent = IntentParser(as_of=AS_OF).parse(
        "最近半年美国采购 HS850440 金额最高的 10 家公司",
        explicit_filters=RetrievalFilter(entity_ids=["company:buyer-7"]),
    )

    retrieval_intent = to_retrieval_query_intent(intent)

    assert type(retrieval_intent) is RetrievalQueryIntent
    assert retrieval_intent.query == intent.question
    assert retrieval_intent.country_codes == ("US",)
    assert retrieval_intent.hs_codes == ("850440",)
    assert retrieval_intent.entity_ids == ("company:buyer-7",)


def test_parser_classifies_company_trend_and_monthly_country_hs_activity() -> None:
    from trade_agent.agents.intent import IntentParser

    parser = IntentParser(as_of=AS_OF)
    trend = parser.parse("ABC Trading 最近3个月采购额趋势")
    activity = parser.parse("中国出口 HS850440 的月度数量")

    assert trend.kind == "company_trend"
    assert trend.filters.company_names == ("ABC Trading",)
    assert trend.filters.company_role == "importer_company"
    assert trend.time_range.start == date(2026, 6, 4)
    assert trend.time_range.grain == "month"
    assert activity.kind == "country_hs_activity"
    assert activity.filters.country_codes == ("CN",)
    assert activity.filters.company_role == "exporter_company"
    assert activity.filters.hs_codes == ("850440",)
    assert activity.constraints.metrics == ["quantity"]
    assert activity.time_range.grain == "month"


def test_parser_marks_mutating_or_unknown_business_requests_out_of_scope() -> None:
    from trade_agent.agents.intent import IntentParser

    intent = IntentParser(as_of=AS_OF).parse("删除所有贸易记录")

    assert intent.kind == "out_of_scope"
    assert intent.out_of_scope_reason == "unsupported_request"
    assert intent.need_trade_data is False
    assert intent.need_external_intel is False


def test_structured_provider_rejects_unknown_json_schema_elements() -> None:
    from trade_agent.agents.intent import IntentParser, StructuredIntentRejected

    parser = IntentParser(
        as_of=AS_OF,
        structured_provider=lambda question: {
            "question": question,
            "kind": "top_importers",
            "surprise": "unreviewed",
        },
    )

    with pytest.raises(StructuredIntentRejected, match="unknown schema"):
        parser.parse("美国采购 HS850440")


def test_structured_provider_cannot_turn_a_trade_kind_into_a_non_trade_route() -> None:
    from trade_agent.agents.intent import IntentParser, StructuredIntentRejected

    parser = IntentParser(
        as_of=AS_OF,
        structured_provider=lambda question: {"question": question, "kind": "top_importers"},
    )

    with pytest.raises(StructuredIntentRejected, match="required schema"):
        parser.parse("美国采购 HS850440")


def test_explicit_country_constraint_conflicting_with_extracted_country_is_rejected() -> None:
    from trade_agent.agents.intent import ExplicitFilterConflict, IntentParser

    with pytest.raises(ExplicitFilterConflict, match="country_codes"):
        IntentParser(as_of=AS_OF).parse(
            "最近半年美国采购 HS850440 金额最高的 10 家公司",
            explicit_filters=RetrievalFilter(country_codes=["DE"]),
        )


def test_structured_provider_cannot_remove_extracted_scope_or_convert_a_refusal() -> None:
    from trade_agent.agents.intent import IntentParser, StructuredIntentRejected

    base = IntentParser(as_of=AS_OF).parse("最近半年美国采购 HS850440 金额最高的 10 家公司")
    removed_scope = base.model_dump(mode="json")
    removed_scope["filters"]["country_codes"] = []
    removed_scope["filters"]["hs_codes"] = []
    removed_scope["time_range"] = {"grain": "total"}
    parser = IntentParser(as_of=AS_OF, structured_provider=lambda _: removed_scope)

    with pytest.raises(StructuredIntentRejected, match="conflicts with deterministic"):
        parser.parse(base.question)

    refusal_bypass = base.model_dump(mode="json")
    parser = IntentParser(as_of=AS_OF, structured_provider=lambda _: refusal_bypass)
    with pytest.raises(StructuredIntentRejected, match="out-of-scope"):
        parser.parse("删除所有贸易记录")


def test_parser_handles_iso_date_ranges_before_hs_identifiers() -> None:
    from trade_agent.agents.intent import IntentParser

    intent = IntentParser(as_of=AS_OF).parse("美国采购 2025-01-01 到 2025-12-31 金额最高的 10 家公司")

    assert intent.filters.country_codes == ("US",)
    assert intent.filters.hs_codes == ()
    assert intent.time_range.start == date(2025, 1, 1)
    assert intent.time_range.end == date(2025, 12, 31)


def test_parser_keeps_external_intelligence_route_when_trade_context_is_present() -> None:
    from trade_agent.agents.intent import IntentParser

    intent = IntentParser(as_of=AS_OF).parse("美国进口 HS850440 最新法规")

    assert intent.need_trade_data is True
    assert intent.need_external_intel is True


def test_parser_recognizes_supported_country_codes_and_rejects_unresolved_country_scope() -> None:
    from trade_agent.agents.intent import IntentParser, UnresolvedIntentConstraint

    japanese = IntentParser(as_of=AS_OF).parse("日本采购 HS850440 金额最高的10家公司")
    assert japanese.filters.country_codes == ("JP",)

    with pytest.raises(UnresolvedIntentConstraint, match="country"):
        IntentParser(as_of=AS_OF).parse("法国采购 HS850440 金额最高的10家公司")


def test_parser_preserves_the_registered_metric_requested_by_a_top_n_question() -> None:
    from trade_agent.agents.intent import IntentParser

    quantity = IntentParser(as_of=AS_OF).parse("美国采购 HS850440 数量最高的10家公司")
    count = IntentParser(as_of=AS_OF).parse("美国采购 HS850440 交易次数最高的10家公司")
    latest = IntentParser(as_of=AS_OF).parse("美国采购 HS850440 最近交易日期最高的10家公司")

    assert quantity.constraints.metrics == ["quantity"]
    assert count.constraints.metrics == ["trade_count"]
    assert latest.constraints.metrics == ["latest_trade_date"]


def test_parser_rejects_conflicting_or_unregistered_explicit_metrics() -> None:
    from trade_agent.agents.intent import IntentParser, UnresolvedIntentConstraint

    parser = IntentParser(as_of=AS_OF)
    with pytest.raises(UnresolvedIntentConstraint, match="conflicting metrics"):
        parser.parse("美国采购 HS850440 金额和数量最高的10家公司")
    with pytest.raises(UnresolvedIntentConstraint, match="unregistered metric"):
        parser.parse("美国采购 HS850440 利润率最高的10家公司")


def test_explicit_rag_filters_cannot_be_weakened_by_a_structured_provider() -> None:
    from trade_agent.agents.intent import IntentParser, StructuredIntentRejected

    base = IntentParser(as_of=AS_OF).parse("美国采购 HS850440 金额最高的10家公司")
    provided = base.model_dump(mode="json")
    provided["retrieval_filter"]["source_types"] = ["official_website"]

    with pytest.raises(StructuredIntentRejected, match="retrieval filters"):
        IntentParser(as_of=AS_OF, structured_provider=lambda _: provided).parse(
            base.question,
            explicit_filters=RetrievalFilter(source_types=["social"]),
        )

    broadened = base.model_dump(mode="json")
    broadened["retrieval_filter"]["country_codes"] = ["DE"]
    with pytest.raises(StructuredIntentRejected, match="retrieval filters"):
        IntentParser(as_of=AS_OF, structured_provider=lambda _: broadened).parse(base.question)
