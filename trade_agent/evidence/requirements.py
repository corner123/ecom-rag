"""Versioned evidence contracts derived only from a typed business intent."""
from __future__ import annotations

from copy import deepcopy
from datetime import date
from typing import Literal, Mapping, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, field_validator, model_validator

from trade_agent.agents.intent import QueryIntent


Branch = Literal["sql", "rag"]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def model_copy(self, *, update: Mapping[str, object] | None = None, deep: bool = False) -> Self:
        if update is None:
            return super().model_copy(deep=deep)
        values = self.model_dump(mode="python")
        if deep:
            values = deepcopy(values)
        values.update(update)
        return type(self).model_validate(values)


class FactAgePolicy(_Contract):
    fact_type: StrictStr
    maximum_age_days: StrictInt = Field(gt=0)

    @field_validator("fact_type")
    @classmethod
    def nonblank_fact(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("fact type must not be blank")
        return value


class FactSourcePolicy(_Contract):
    fact_type: StrictStr
    allowed_source_types: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def stable_sources(self) -> Self:
        if not self.fact_type.strip() or any(not item.strip() for item in self.allowed_source_types):
            raise ValueError("fact and source types must not be blank")
        if self.allowed_source_types != tuple(sorted(set(self.allowed_source_types))):
            raise ValueError("allowed source types must be sorted and unique")
        return self


class SufficiencyPolicy(_Contract):
    """All thresholds used by the v1 validator; no score is a trust signal."""

    policy_version: Literal["trade-evidence-sufficiency-v1"] = "trade-evidence-sufficiency-v1"
    fact_age_days: tuple[FactAgePolicy, ...] = (
        FactAgePolicy(fact_type="company_status", maximum_age_days=180),
        FactAgePolicy(fact_type="contact", maximum_age_days=365),
        FactAgePolicy(fact_type="market_signal", maximum_age_days=180),
        FactAgePolicy(fact_type="product_offering", maximum_age_days=365),
        FactAgePolicy(fact_type="regulation", maximum_age_days=365),
        FactAgePolicy(fact_type="risk", maximum_age_days=180),
        FactAgePolicy(fact_type="trade_activity", maximum_age_days=365),
    )
    fact_sources: tuple[FactSourcePolicy, ...] = (
        FactSourcePolicy(
            fact_type="company_status",
            allowed_source_types=("industry_news", "official_website", "regulator"),
        ),
        FactSourcePolicy(
            fact_type="contact",
            allowed_source_types=("b2b", "official_website"),
        ),
        FactSourcePolicy(
            fact_type="market_signal",
            allowed_source_types=("industry_news", "official_website", "regulator"),
        ),
        FactSourcePolicy(
            fact_type="product_offering",
            allowed_source_types=("b2b", "official_website"),
        ),
        FactSourcePolicy(fact_type="regulation", allowed_source_types=("regulator",)),
        FactSourcePolicy(
            fact_type="risk",
            allowed_source_types=("industry_news", "official_website", "regulator"),
        ),
        FactSourcePolicy(
            fact_type="trade_activity",
            allowed_source_types=("customs_profile", "regulator"),
        ),
    )
    lead_status_independent_sources: StrictInt = Field(default=2, ge=1)
    external_independent_sources: StrictInt = Field(default=1, ge=1)
    lead_sql_maximum_age_days: StrictInt = Field(default=365, gt=0)
    fatal_degraded_components: tuple[StrictStr, ...] = (
        "all_retrievers_unavailable",
        "branch_timeout",
        "required_branch_timeout",
        "retrieval_timeout",
        "retrieval_unavailable",
    )

    @model_validator(mode="after")
    def stable_policy(self) -> Self:
        ages = tuple(item.fact_type for item in self.fact_age_days)
        sources = tuple(item.fact_type for item in self.fact_sources)
        if ages != tuple(sorted(set(ages))) or sources != tuple(sorted(set(sources))):
            raise ValueError("fact policy entries must be sorted and unique")
        if ages != sources:
            raise ValueError("fact age and source policies must cover the same facts")
        if self.fatal_degraded_components != tuple(sorted(set(self.fatal_degraded_components))):
            raise ValueError("fatal degraded components must be sorted and unique")
        return self

    @property
    def age_by_fact(self) -> Mapping[str, int]:
        return {item.fact_type: item.maximum_age_days for item in self.fact_age_days}

    @property
    def sources_by_fact(self) -> Mapping[str, tuple[str, ...]]:
        return {item.fact_type: item.allowed_source_types for item in self.fact_sources}


DEFAULT_SUFFICIENCY_POLICY = SufficiencyPolicy()


class EvidenceRequirement(_Contract):
    requirement_id: StrictStr
    branch: Branch
    fact_types: tuple[StrictStr, ...]
    allowed_source_types: tuple[StrictStr, ...]
    requested_source_types: tuple[StrictStr, ...] = ()
    minimum_independent_sources: StrictInt = Field(ge=1)
    maximum_age_days: StrictInt | None = Field(default=None, gt=0)
    require_resolved_entity: StrictBool = False
    required_entity_ids: tuple[StrictStr, ...] = ()
    required_company_names: tuple[StrictStr, ...] = ()
    required_country_codes: tuple[StrictStr, ...] = ()
    required_hs_codes: tuple[StrictStr, ...] = ()
    required_metrics: tuple[StrictStr, ...] = ()
    required_dimensions: tuple[StrictStr, ...] = ()
    required_time_grain: StrictStr | None = None
    require_currency: StrictBool = False
    require_unit: StrictBool = False
    temporal_start: date | None = None
    temporal_end: date | None = None
    require_synthetic: StrictBool | None = None
    core: Literal[True] = True

    @field_validator(
        "fact_types",
        "allowed_source_types",
        "requested_source_types",
        "required_entity_ids",
        "required_company_names",
        "required_country_codes",
        "required_hs_codes",
        "required_metrics",
        "required_dimensions",
    )
    @classmethod
    def sorted_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if values != tuple(sorted(set(values))):
            raise ValueError("requirement values must be sorted and unique")
        return values

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if not self.requirement_id.strip() or not self.fact_types:
            raise ValueError("requirement identity and fact types are required")
        if self.temporal_start is not None and self.temporal_end is not None and self.temporal_start > self.temporal_end:
            raise ValueError("requirement temporal range is reversed")
        if (self.temporal_start is None) != (self.temporal_end is None):
            raise ValueError("requirement temporal range requires both bounds")
        if self.branch == "sql" and self.allowed_source_types != ("sql",):
            raise ValueError("SQL requirements accept only SQL Evidence")
        if self.branch == "rag" and "sql" in self.allowed_source_types:
            raise ValueError("RAG requirements cannot accept SQL Evidence")
        return self


class EvidenceRequirements(_Contract):
    policy_version: StrictStr
    intent_kind: StrictStr
    required_branches: tuple[Branch, ...]
    items: tuple[EvidenceRequirement, ...]

    @model_validator(mode="after")
    def stable_contract(self) -> Self:
        if self.required_branches != tuple(sorted(set(self.required_branches))):
            raise ValueError("required branches must be sorted and unique")
        ids = tuple(item.requirement_id for item in self.items)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("requirements must be sorted by unique ID")
        if tuple(sorted({item.branch for item in self.items})) != self.required_branches:
            raise ValueError("required branches must equal requirement branches")
        return self

    @classmethod
    def for_intent(
        cls,
        intent: QueryIntent,
        *,
        policy: SufficiencyPolicy = DEFAULT_SUFFICIENCY_POLICY,
    ) -> "EvidenceRequirements":
        if type(intent) is not QueryIntent:
            raise TypeError("intent must be an exact business QueryIntent")
        if type(policy) is not SufficiencyPolicy:
            raise TypeError("policy must be an exact SufficiencyPolicy")
        items: list[EvidenceRequirement] = []
        entity_ids = _union(intent.filters.entity_ids, intent.retrieval_filter.entity_ids)
        countries = _union(intent.filters.country_codes, intent.retrieval_filter.country_codes)
        hs_codes = _union(intent.filters.hs_codes, intent.retrieval_filter.hs_codes)
        company_names = tuple(sorted(set(intent.filters.company_names)))
        synthetic = intent.retrieval_filter.is_synthetic

        if intent.need_trade_data:
            metrics = tuple(sorted(set(intent.constraints.metrics)))
            metric = metrics[0] if len(metrics) == 1 else "trade_aggregate"
            dimensions = tuple(
                sorted(
                    "trade_date" if item == "time" else item
                    for item in set(intent.constraints.dimensions)
                )
            )
            items.append(
                EvidenceRequirement(
                    requirement_id="sql.trade_aggregate",
                    branch="sql",
                    fact_types=metrics or ("trade_aggregate",),
                    allowed_source_types=("sql",),
                    minimum_independent_sources=1,
                    maximum_age_days=(
                        policy.lead_sql_maximum_age_days
                        if intent.kind == "lead_assessment" and intent.time_range.end is None
                        else None
                    ),
                    required_entity_ids=entity_ids,
                    required_company_names=company_names,
                    required_country_codes=countries,
                    required_hs_codes=hs_codes,
                    required_metrics=metrics,
                    required_dimensions=dimensions,
                    required_time_grain=intent.time_range.grain,
                    require_currency=metric == "trade_amount",
                    require_unit=metric == "quantity",
                    temporal_start=intent.time_range.start,
                    temporal_end=intent.time_range.end,
                    require_synthetic=synthetic,
                )
            )

        if intent.need_external_intel:
            fact_groups = _required_rag_fact_groups(intent, policy)
            for facts in fact_groups:
                policy_sources = tuple(
                    sorted(
                        {
                            source
                            for fact in facts
                            for source in policy.sources_by_fact.get(fact, ())
                        }
                    )
                )
                explicit_sources = tuple(sorted(item.value for item in intent.retrieval_filter.source_types))
                published_start = (
                    intent.retrieval_filter.published_after.date()
                    if intent.retrieval_filter.published_after is not None
                    else None
                )
                published_end = (
                    intent.retrieval_filter.published_before.date()
                    if intent.retrieval_filter.published_before is not None
                    else None
                )
                if published_start is None and published_end is not None:
                    published_start = date.min
                if published_end is None and published_start is not None:
                    published_end = date.max
                items.append(
                    EvidenceRequirement(
                        requirement_id=(
                            f"rag.{facts[0]}" if len(facts) == 1 else "rag.external_fact"
                        ),
                        branch="rag",
                        fact_types=facts,
                        allowed_source_types=policy_sources,
                        requested_source_types=explicit_sources,
                        minimum_independent_sources=(
                            policy.lead_status_independent_sources
                            if intent.kind == "lead_assessment"
                            else policy.external_independent_sources
                        ),
                        maximum_age_days=(
                            None
                            if published_start is not None
                            else min(policy.age_by_fact[fact] for fact in facts)
                        ),
                        require_resolved_entity=bool(entity_ids or company_names),
                        required_entity_ids=entity_ids,
                        required_company_names=company_names,
                        required_country_codes=countries,
                        required_hs_codes=hs_codes,
                        temporal_start=published_start,
                        temporal_end=published_end,
                        require_synthetic=synthetic,
                    )
                )

        ordered = tuple(sorted(items, key=lambda item: item.requirement_id))
        return cls(
            policy_version=policy.policy_version,
            intent_kind=intent.kind,
            required_branches=tuple(sorted({item.branch for item in ordered})),
            items=ordered,
        )


def _union(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(set(left) | set(right)))


def _required_rag_fact_groups(
    intent: QueryIntent,
    policy: SufficiencyPolicy,
) -> tuple[tuple[str, ...], ...]:
    explicit = tuple(sorted(item.value for item in intent.retrieval_filter.fact_types))
    if explicit:
        return (explicit,)
    if intent.kind == "lead_assessment":
        return (("company_status",),)
    question = intent.question.casefold()
    rules = (
        ("regulation", ("regulation", "regulatory", "compliance", "法规", "合规")),
        ("risk", ("risk", "recall", "sanction", "风险", "召回", "制裁")),
        ("contact", ("contact", "email", "phone", "联系", "邮箱", "电话")),
        ("product_offering", ("product", "catalog", "产品", "供应")),
        ("company_status", ("website", "status", "operat", "expan", "官网", "状态", "扩产", "开业", "停业")),
        ("market_signal", ("market", "demand", "price", "市场", "需求", "价格", "新闻")),
    )
    matched = tuple(sorted(fact for fact, tokens in rules if any(token in question for token in tokens)))
    # Generic external intelligence still needs one known factual category.  The
    # OR contract does not accept fact_type=unknown.
    if matched:
        return tuple((fact,) for fact in matched)
    return (tuple(item.fact_type for item in policy.fact_age_days),)
