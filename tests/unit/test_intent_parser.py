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
