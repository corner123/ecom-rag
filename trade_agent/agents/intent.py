"""Deterministic business-intent parsing for the trade workflow.

This module deliberately does not reuse ``retrieval.planner.QueryIntent``.
That model is the strict retrieval boundary; a business request has SQL route,
time-grain, and aggregate semantics that retrieval does not understand.
"""
from __future__ import annotations

from calendar import monthrange
from collections.abc import Callable, Mapping
from datetime import date
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from trade_agent.db.contracts import QueryConstraints
from trade_agent.retrieval.filters import RetrievalFilter
from trade_agent.retrieval.planner import QueryIntent as RetrievalQueryIntent


IntentKind = Literal[
    "top_importers",
    "company_trend",
    "country_hs_activity",
    "lead_assessment",
    "external_intelligence",
    "out_of_scope",
]
CompanyRole = Literal["importer_company", "exporter_company"]
TimeGrain = Literal["total", "day", "month"]
StructuredIntentProvider = Callable[[str], Mapping[str, object]]


class StructuredIntentRejected(ValueError):
    """A structured provider response does not satisfy the local contract."""


class TimeRange(BaseModel):
    """A trade-date range whose relative bounds are resolved before planning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: date | None = None
    end: date | None = None
    months: int | None = Field(default=None, ge=1, le=120)
    grain: TimeGrain = "total"

    @model_validator(mode="after")
    def ordered(self) -> "TimeRange":
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("time range start must not exceed end")
        if (self.start is None) != (self.end is None):
            raise ValueError("time range requires both start and end")
        return self


class IntentFilters(BaseModel):
    """Identifiers and dimensions preserved from a business request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    country_codes: tuple[str, ...] = ()
    hs_codes: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    company_names: tuple[str, ...] = ()
    company_role: CompanyRole = "importer_company"


class QueryIntent(BaseModel):
    """Typed business intent consumed by SQL, RAG, and evidence policies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str
    kind: IntentKind
    constraints: QueryConstraints = Field(default_factory=QueryConstraints)
    filters: IntentFilters = Field(default_factory=IntentFilters)
    retrieval_filter: RetrievalFilter = Field(default_factory=RetrievalFilter)
    time_range: TimeRange = Field(default_factory=TimeRange)
    need_trade_data: bool = False
    need_external_intel: bool = False
    limit: int = Field(default=50, ge=1, le=50)
    out_of_scope_reason: Literal["unsupported_request", "schema_unknown"] | None = None

    @model_validator(mode="after")
    def route_is_consistent(self) -> "QueryIntent":
        if not self.question.strip():
            raise ValueError("question must not be blank")
        if self.kind == "out_of_scope":
            if self.out_of_scope_reason is None or self.need_trade_data or self.need_external_intel:
                raise ValueError("out-of-scope intent must have only an explicit refusal reason")
        elif self.out_of_scope_reason is not None:
            raise ValueError("supported intents cannot carry an out-of-scope reason")
        if self.kind in {"top_importers", "company_trend", "country_hs_activity", "lead_assessment"} and not self.need_trade_data:
            raise ValueError("trade intent kinds require the trade-data route")
        if self.kind == "lead_assessment" and not self.need_external_intel:
            raise ValueError("lead assessment requires the external-intelligence route")
        if self.kind == "external_intelligence" and not self.need_external_intel:
            raise ValueError("external intelligence intent requires its retrieval route")
        return self


_COUNTRY_NAMES = {
    "US": ("美国", "usa", "united states"),
    "CN": ("中国", "china"),
    "DE": ("德国", "germany"),
    "VN": ("越南", "vietnam"),
    "BR": ("巴西", "brazil"),
    "AE": ("阿联酋", "uae", "united arab emirates"),
    "ZA": ("南非", "south africa"),
    "AU": ("澳大利亚", "australia"),
}
_HS = re.compile(r"(?i)(?:\bhs\s*)?(\d{4,10})\b")
_MONTHS = re.compile(r"(?:最近|近|过去|past|last)\s*(\d+|半)\s*(?:个)?(?:月|months?)", re.IGNORECASE)
_LIMIT = re.compile(r"(?:top\s*|前\s*|最高的?\s*)(\d+)\s*(?:家|个|companies?)?", re.IGNORECASE)
_COMPANY_BEFORE_RANGE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 .&'_-]{1,127}?)\s*(?:最近|近|过去|past|last)", re.IGNORECASE)
_COMPANY_BEFORE_DECISION = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 .&'_-]{1,127}?)\s*(?:是否|is\s+it|官网|website)", re.IGNORECASE)


class IntentParser:
    """Extract only reviewed, deterministic business constraints.

    ``as_of`` is required so relative dates never depend on machine time.  An
    optional provider may supply the same JSON schema, but it cannot relax
    local validation or replace caller-supplied retrieval constraints.
    """

    def __init__(self, *, as_of: date, structured_provider: StructuredIntentProvider | None = None) -> None:
        if not isinstance(as_of, date):
            raise TypeError("as_of must be a date")
        self.as_of = as_of
        self.structured_provider = structured_provider

    def parse(self, question: str, explicit_filters: RetrievalFilter | None = None) -> QueryIntent:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonblank string")
        if explicit_filters is not None and type(explicit_filters) is not RetrievalFilter:
            raise TypeError("explicit_filters must be an exact RetrievalFilter instance")
        deterministic = self._parse_deterministic(question)
        if self.structured_provider is None:
            return _apply_explicit_filters(deterministic, explicit_filters)
        try:
            provided = self.structured_provider(question)
            if not isinstance(provided, Mapping):
                raise TypeError("provider output must be a mapping")
            unknown = set(provided) - set(QueryIntent.model_fields)
            if unknown:
                raise StructuredIntentRejected("structured intent contains unknown schema elements")
            candidate = QueryIntent.model_validate(provided)
        except StructuredIntentRejected:
            raise
        except (TypeError, ValidationError, ValueError) as error:
            raise StructuredIntentRejected("structured intent does not satisfy the required schema") from error
        if candidate.question != question:
            raise StructuredIntentRejected("structured intent question does not match the request")
        return _apply_explicit_filters(candidate, explicit_filters)

    def _parse_deterministic(self, question: str) -> QueryIntent:
        normalized = " ".join(question.split())
        lowered = normalized.casefold()
        if any(token in lowered for token in ("删除", "drop table", "delete ", "update ", "插入", "修改数据库")):
            return QueryIntent(question=question, kind="out_of_scope", out_of_scope_reason="unsupported_request")

        role: CompanyRole = "exporter_company" if any(token in lowered for token in ("出口", "export")) else "importer_company"
        countries = _country_codes(normalized)
        hs_codes = tuple(sorted(set(_HS.findall(normalized))))
        time_range = _time_range(normalized, self.as_of)
        company_names = _company_names(normalized)
        if "月度" in normalized or "monthly" in lowered or "趋势" in normalized or "trend" in lowered:
            time_range = time_range.model_copy(update={"grain": "month"})
        limit = _limit(normalized)
        filters = IntentFilters(
            country_codes=countries,
            hs_codes=hs_codes,
            company_names=company_names,
            company_role=role,
        )

        lead = any(token in lowered for token in ("值得跟进", "是否跟进", "lead", "follow up", "跟进"))
        external = any(token in lowered for token in ("官网", "website", "扩产", "状态", "新闻", "regulation", "法规"))
        quantity = any(token in lowered for token in ("数量", "重量", "quantity", "volume"))
        top = limit is not None and any(token in lowered for token in ("最高", "top", "排名", "前"))
        trade_language = any(
            token in lowered
            for token in ("采购", "进口", "出口", "贸易", "交易", "hs", "金额", "数量", "趋势", "活跃", "增长", "下降")
        ) or bool(hs_codes)

        if lead:
            return QueryIntent(
                question=question,
                kind="lead_assessment",
                constraints=_constraints("trade_amount", role, countries, hs_codes, time_range),
                filters=filters,
                time_range=time_range,
                need_trade_data=True,
                need_external_intel=True,
                limit=50,
            )
        if top:
            return QueryIntent(
                question=question,
                kind="top_importers" if role == "importer_company" else "country_hs_activity",
                constraints=_constraints("trade_amount", role, countries, hs_codes, time_range),
                filters=filters,
                time_range=time_range,
                need_trade_data=True,
                limit=limit,
            )
        if company_names and ("趋势" in normalized or "trend" in lowered or trade_language):
            return QueryIntent(
                question=question,
                kind="company_trend",
                constraints=_constraints("quantity" if quantity else "trade_amount", role, countries, hs_codes, time_range),
                filters=filters,
                time_range=time_range,
                need_trade_data=True,
            )
        if trade_language or countries or hs_codes:
            return QueryIntent(
                question=question,
                kind="country_hs_activity",
                constraints=_constraints("quantity" if quantity else "trade_amount", role, countries, hs_codes, time_range),
                filters=filters,
                time_range=time_range,
                need_trade_data=True,
            )
        if external:
            return QueryIntent(question=question, kind="external_intelligence", need_external_intel=True)
        return QueryIntent(question=question, kind="out_of_scope", out_of_scope_reason="unsupported_request")


def to_retrieval_query_intent(intent: QueryIntent) -> RetrievalQueryIntent:
    """Adapt business constraints to the existing strict retrieval contract."""

    if type(intent) is not QueryIntent:
        raise TypeError("intent must be an exact business QueryIntent instance")
    filter_ = intent.retrieval_filter.model_copy(
        update={
            "country_codes": tuple(sorted(set(intent.retrieval_filter.country_codes) | set(intent.filters.country_codes))),
            "hs_codes": tuple(sorted(set(intent.retrieval_filter.hs_codes) | set(intent.filters.hs_codes))),
            "entity_ids": tuple(sorted(set(intent.retrieval_filter.entity_ids) | set(intent.filters.entity_ids))),
        }
    )
    return RetrievalQueryIntent(
        query=intent.question,
        region=filter_.region,
        country_codes=filter_.country_codes,
        hs_codes=filter_.hs_codes,
        entity_ids=filter_.entity_ids,
        source_types=filter_.source_types,
        fact_types=filter_.fact_types,
        published_after=filter_.published_after,
        published_before=filter_.published_before,
        is_synthetic=filter_.is_synthetic,
    )


def _apply_explicit_filters(intent: QueryIntent, explicit: RetrievalFilter | None) -> QueryIntent:
    if explicit is None:
        return intent
    filters = intent.filters.model_copy(
        update={
            "country_codes": tuple(sorted(set(intent.filters.country_codes) | set(explicit.country_codes))),
            "hs_codes": tuple(sorted(set(intent.filters.hs_codes) | set(explicit.hs_codes))),
            "entity_ids": tuple(sorted(set(intent.filters.entity_ids) | set(explicit.entity_ids))),
        }
    )
    retrieval = intent.retrieval_filter.model_copy(
        update={
            "region": explicit.region if explicit.region is not None else intent.retrieval_filter.region,
            "country_codes": tuple(sorted(set(intent.retrieval_filter.country_codes) | set(explicit.country_codes))),
            "hs_codes": tuple(sorted(set(intent.retrieval_filter.hs_codes) | set(explicit.hs_codes))),
            "entity_ids": tuple(sorted(set(intent.retrieval_filter.entity_ids) | set(explicit.entity_ids))),
            "source_types": tuple(sorted(set(intent.retrieval_filter.source_types) | set(explicit.source_types), key=lambda value: value.value)),
            "fact_types": tuple(sorted(set(intent.retrieval_filter.fact_types) | set(explicit.fact_types), key=lambda value: value.value)),
            "published_after": explicit.published_after or intent.retrieval_filter.published_after,
            "published_before": explicit.published_before or intent.retrieval_filter.published_before,
            "is_synthetic": explicit.is_synthetic if explicit.is_synthetic is not None else intent.retrieval_filter.is_synthetic,
        }
    )
    return intent.model_copy(update={"filters": filters, "retrieval_filter": retrieval})


def _constraints(metric: str, role: CompanyRole, countries: tuple[str, ...], hs_codes: tuple[str, ...], time_range: TimeRange) -> QueryConstraints:
    country_field = "import_country" if role == "importer_company" else "export_country"
    fields = [country_field] if countries else []
    if hs_codes:
        fields.append("hs_code")
    if time_range.start is not None:
        fields.append("time")
    dimensions = [role]
    if time_range.grain == "month":
        dimensions.append("time")
    return QueryConstraints(metrics=[metric], dimensions=dimensions, filters=fields)


def _country_codes(question: str) -> tuple[str, ...]:
    lowered = question.casefold()
    codes = {
        code
        for code, names in _COUNTRY_NAMES.items()
        if any(name in question if any(ord(char) > 127 for char in name) else name in lowered for name in names)
    }
    for candidate in re.findall(r"\b[A-Z]{2}\b", question):
        if candidate in _COUNTRY_NAMES:
            codes.add(candidate)
    return tuple(sorted(codes))


def _time_range(question: str, as_of: date) -> TimeRange:
    if "半年" in question or "half year" in question.casefold():
        return TimeRange(start=_subtract_months(as_of, 6), end=as_of, months=6)
    match = _MONTHS.search(question)
    if match:
        months = 6 if match.group(1) == "半" else int(match.group(1))
        return TimeRange(start=_subtract_months(as_of, months), end=as_of, months=months)
    year = re.search(r"\b(20\d{2})年", question)
    if year:
        value = int(year.group(1))
        return TimeRange(start=date(value, 1, 1), end=date(value, 12, 31))
    return TimeRange()


def _subtract_months(value: date, months: int) -> date:
    absolute = value.year * 12 + value.month - 1 - months
    year, month_zero = divmod(absolute, 12)
    month = month_zero + 1
    return date(year, month, min(value.day, monthrange(year, month)[1]))


def _limit(question: str) -> int | None:
    match = _LIMIT.search(question)
    return int(match.group(1)) if match else None


def _company_names(question: str) -> tuple[str, ...]:
    match = _COMPANY_BEFORE_RANGE.search(question) or _COMPANY_BEFORE_DECISION.search(question)
    if match is None:
        return ()
    name = match.group(1).strip()
    return (name,) if name else ()
