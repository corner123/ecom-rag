"""Versioned evidence contracts derived only from a validated business intent."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
from hashlib import sha256
from itertools import product
import json
from typing import Literal, Mapping, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from trade_agent.agents.intent import QueryIntent

Branch = Literal["sql", "rag"]
BranchErrorCode = Literal[
    "invalid_contract", "policy_denied", "schema_drift", "timeout", "transport",
    "unavailable", "unknown",
]
RetryableBranchErrorCode = Literal["timeout", "transport", "unavailable"]
RETRYABLE_BRANCH_ERROR_CODES: tuple[RetryableBranchErrorCode, ...] = (
    "timeout", "transport", "unavailable",
)
HARD_BRANCH_ERROR_CODES: tuple[BranchErrorCode, ...] = (
    "invalid_contract", "policy_denied", "schema_drift", "unknown",
)
PartialDegradationCode = Literal[
    "dense_unavailable", "reranker_unavailable", "sparse_unavailable",
    "structured_output_unavailable",
]
FatalDegradationCode = Literal[
    "all_retrievers_unavailable", "branch_timeout", "required_branch_timeout",
    "retrieval_timeout", "retrieval_unavailable", "sql_timeout", "sql_unavailable",
]
IndependenceDimension = Literal[
    "content_sha256", "dedupe_cluster_id", "publisher_identity",
]
SqlScopeColumn = Literal[
    "export_country.country_code", "exporter.company_name", "exporter.id",
    "hs.hs_code", "import_country.country_code", "importer.company_name",
    "importer.id",
]


class InvalidIntentContract(ValueError):
    """The supplied intent cannot safely define evidence requirements."""


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

    @model_validator(mode="after")
    def nonblank(self) -> Self:
        if not self.fact_type.strip():
            raise ValueError("fact type must not be blank")
        return self


class FactSourcePolicy(_Contract):
    fact_type: StrictStr
    allowed_source_types: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def stable(self) -> Self:
        if not self.fact_type.strip() or any(not item.strip() for item in self.allowed_source_types) or self.allowed_source_types != tuple(sorted(set(self.allowed_source_types))):
            raise ValueError("fact source policy must be nonblank, sorted and unique")
        return self


class SourceAuthorityRule(_Contract):
    rule_id: StrictStr
    publisher_id: StrictStr
    source_type: StrictStr
    exact_source_identities: tuple[StrictStr, ...] = ()
    domain_suffixes: tuple[StrictStr, ...] = ()
    directness: Literal["direct", "authoritative", "secondary", "derived"]

    @model_validator(mode="after")
    def stable(self) -> Self:
        if not self.rule_id.strip() or not self.publisher_id.strip() or not self.source_type.strip():
            raise ValueError("authority rule, publisher, and source type are required")
        for values in (self.exact_source_identities, self.domain_suffixes):
            if values != tuple(sorted(set(values))) or any(not item.strip() for item in values):
                raise ValueError("authority identities must be nonblank, sorted and unique")
        if not self.exact_source_identities and not self.domain_suffixes:
            raise ValueError("authority rule needs an exact identity or reviewed domain suffix")
        return self


_DEFAULT_AUTHORITY_RULES = tuple(sorted((
    SourceAuthorityRule(rule_id="b2b.reviewed", publisher_id="synthetic-marketplace", source_type="b2b", domain_suffixes=("marketplace.example",), directness="secondary"),
    SourceAuthorityRule(rule_id="customs.reviewed", publisher_id="synthetic-customs", source_type="customs_profile", domain_suffixes=("profiles.example",), directness="authoritative"),
    SourceAuthorityRule(rule_id="news.reviewed", publisher_id="synthetic-newswire", source_type="industry_news", domain_suffixes=("news.example", "newsroom.example", "syndication.example"), directness="secondary"),
    SourceAuthorityRule(rule_id="official.acme", publisher_id="acme", source_type="official_website", domain_suffixes=("acme.example",), directness="direct"),
    *(SourceAuthorityRule(rule_id=f"official.company-{index:02d}", publisher_id=f"company-{index:02d}", source_type="official_website", domain_suffixes=(f"company-{index:02d}.example",), directness="direct") for index in range(1, 13)),
    SourceAuthorityRule(rule_id="official.reviewed", publisher_id="official-demo", source_type="official_website", domain_suffixes=("official.example",), directness="direct"),
    SourceAuthorityRule(rule_id="regulator.reviewed", publisher_id="synthetic-regulator", source_type="regulator", domain_suffixes=("regulator.example",), directness="authoritative"),
), key=lambda item: item.rule_id))


class SufficiencyPolicy(_Contract):
    """Every semantic threshold used by the score-independent validator."""

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
        FactSourcePolicy(fact_type="company_status", allowed_source_types=("industry_news", "official_website", "regulator")),
        FactSourcePolicy(fact_type="contact", allowed_source_types=("b2b", "official_website")),
        FactSourcePolicy(fact_type="market_signal", allowed_source_types=("industry_news", "official_website", "regulator")),
        FactSourcePolicy(fact_type="product_offering", allowed_source_types=("b2b", "official_website")),
        FactSourcePolicy(fact_type="regulation", allowed_source_types=("regulator",)),
        FactSourcePolicy(fact_type="risk", allowed_source_types=("industry_news", "official_website", "regulator")),
        FactSourcePolicy(fact_type="trade_activity", allowed_source_types=("customs_profile", "regulator")),
    )
    authority_rules: tuple[SourceAuthorityRule, ...] = _DEFAULT_AUTHORITY_RULES
    accepted_authority_directness: tuple[Literal["authoritative", "direct", "secondary"], ...] = ("authoritative", "direct", "secondary")
    lead_status_independent_sources: StrictInt = Field(default=2, ge=1)
    external_independent_sources: StrictInt = Field(default=1, ge=1)
    lead_sql_maximum_age_days: StrictInt = Field(default=365, gt=0)
    independence_dimensions: tuple[IndependenceDimension, ...] = ("content_sha256", "dedupe_cluster_id", "publisher_identity")
    fatal_error_codes: tuple[RetryableBranchErrorCode, ...] = RETRYABLE_BRANCH_ERROR_CODES
    fatal_evidence_degraded_components: tuple[FatalDegradationCode, ...] = (
        "all_retrievers_unavailable", "branch_timeout", "required_branch_timeout",
        "retrieval_timeout", "retrieval_unavailable", "sql_timeout", "sql_unavailable",
    )
    partial_degraded_components: tuple[PartialDegradationCode, ...] = (
        "dense_unavailable", "reranker_unavailable", "sparse_unavailable",
        "structured_output_unavailable",
    )
    partial_degradation_mode: Literal["audit_only", "block"] = "audit_only"
    policy_fingerprint: StrictStr = ""

    @model_validator(mode="after")
    def stable_policy(self) -> Self:
        ages = tuple(item.fact_type for item in self.fact_age_days)
        sources = tuple(item.fact_type for item in self.fact_sources)
        if ages != tuple(sorted(set(ages))) or sources != tuple(sorted(set(sources))) or ages != sources:
            raise ValueError("fact policies must cover identical sorted unique facts")
        rule_ids = tuple(item.rule_id for item in self.authority_rules)
        if rule_ids != tuple(sorted(set(rule_ids))):
            raise ValueError("authority rules must be sorted by unique rule ID")
        for values in (
            self.accepted_authority_directness, self.independence_dimensions,
            self.fatal_error_codes,
            self.fatal_evidence_degraded_components, self.partial_degraded_components,
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError("policy tuple values must be sorted and unique")
        if not self.independence_dimensions:
            raise ValueError("at least one independence dimension is required")
        expected = _policy_fingerprint(self)
        if "policy_fingerprint" in self.model_fields_set and self.policy_fingerprint != expected:
            raise ValueError("policy_fingerprint does not match policy semantics")
        object.__setattr__(self, "policy_fingerprint", expected)
        return self

    def model_copy(self, *, update: Mapping[str, object] | None = None, deep: bool = False) -> Self:
        values = self.model_dump(mode="python")
        if deep:
            values = deepcopy(values)
        values.pop("policy_fingerprint", None)
        if update:
            values.update(update)
        return type(self).model_validate(values)

    @property
    def age_by_fact(self) -> Mapping[str, int]:
        return {item.fact_type: item.maximum_age_days for item in self.fact_age_days}

    @property
    def sources_by_fact(self) -> Mapping[str, tuple[str, ...]]:
        return {item.fact_type: item.allowed_source_types for item in self.fact_sources}

    @property
    def hard_error_codes(self) -> tuple[BranchErrorCode, ...]:
        """Deterministic contract and policy failures can never become retryable."""
        return HARD_BRANCH_ERROR_CODES


def _policy_fingerprint(policy: SufficiencyPolicy) -> str:
    payload = policy.model_dump(mode="json", exclude={"policy_fingerprint"})
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(canonical.encode()).hexdigest()


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
    required_entity_column: SqlScopeColumn | None = None
    required_company_column: SqlScopeColumn | None = None
    required_country_column: SqlScopeColumn | None = None
    required_hs_column: SqlScopeColumn | None = None
    required_time_grain: StrictStr | None = None
    require_currency: StrictBool = False
    require_unit: StrictBool = False
    temporal_start: date | datetime | None = None
    temporal_end: date | datetime | None = None
    require_synthetic: StrictBool | None = None
    core: Literal[True] = True

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if not self.requirement_id.strip() or not self.fact_types:
            raise ValueError("requirement identity and fact types are required")
        for values in (self.fact_types, self.allowed_source_types, self.requested_source_types, self.required_entity_ids, self.required_company_names, self.required_country_codes, self.required_hs_codes, self.required_metrics, self.required_dimensions):
            if values != tuple(sorted(set(values))):
                raise ValueError("requirement values must be sorted and unique")
        if (self.temporal_start is None) != (self.temporal_end is None):
            raise ValueError("requirement temporal range requires both bounds")
        if self.temporal_start is not None and self.temporal_end is not None and self.temporal_start > self.temporal_end:
            raise ValueError("requirement temporal range is reversed")
        if self.branch == "sql" and self.allowed_source_types != ("sql",):
            raise ValueError("SQL requirements accept only SQL Evidence")
        if self.branch == "rag" and "sql" in self.allowed_source_types:
            raise ValueError("RAG requirements cannot accept SQL Evidence")
        return self


class EvidenceRequirements(_Contract):
    policy_version: StrictStr
    policy_fingerprint: StrictStr
    intent_kind: StrictStr
    intent_valid: StrictBool = True
    required_branches: tuple[Branch, ...]
    items: tuple[EvidenceRequirement, ...]
    requirements_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def stable_contract(self) -> Self:
        ids = tuple(item.requirement_id for item in self.items)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("requirements must be sorted by unique ID")
        if self.required_branches != tuple(sorted(set(self.required_branches))):
            raise ValueError("required branches must be sorted and unique")
        if tuple(sorted({item.branch for item in self.items})) != self.required_branches:
            raise ValueError("required branches must equal requirement branches")
        if self.intent_valid and self.intent_kind != "out_of_scope" and not self.items:
            raise ValueError("a supported intent must produce requirements")
        expected = _requirements_fingerprint(self)
        if self.requirements_fingerprint != expected:
            raise ValueError("requirements_fingerprint does not match requirement semantics")
        return self

    def model_copy(self, *, update: Mapping[str, object] | None = None, deep: bool = False) -> Self:
        values = self.model_dump(mode="python")
        if deep:
            values = deepcopy(values)
        values.pop("requirements_fingerprint", None)
        if update:
            values.update(update)
        values.pop("requirements_fingerprint", None)
        return type(self)._from_semantics(values)

    @classmethod
    def _from_semantics(cls, values: Mapping[str, object]) -> "EvidenceRequirements":
        payload = dict(values)
        payload.pop("requirements_fingerprint", None)
        payload.setdefault("intent_valid", True)
        payload["requirements_fingerprint"] = _requirements_payload_fingerprint(payload)
        return cls.model_validate(payload)

    @classmethod
    def invalid(cls, kind: str, policy: SufficiencyPolicy) -> "EvidenceRequirements":
        return cls._from_semantics({
            "policy_version": policy.policy_version,
            "policy_fingerprint": policy.policy_fingerprint,
            "intent_kind": kind,
            "intent_valid": False,
            "required_branches": (),
            "items": (),
        })

    @classmethod
    def for_intent(cls, intent: QueryIntent, *, policy: SufficiencyPolicy = DEFAULT_SUFFICIENCY_POLICY) -> "EvidenceRequirements":
        intent = validated_intent(intent)
        if type(policy) is not SufficiencyPolicy:
            raise TypeError("policy must be an exact SufficiencyPolicy")
        items: list[EvidenceRequirement] = []
        entity_ids = _union(intent.filters.entity_ids, intent.retrieval_filter.entity_ids)
        countries = _union(intent.filters.country_codes, intent.retrieval_filter.country_codes)
        hs_codes = _union(intent.filters.hs_codes, intent.retrieval_filter.hs_codes)
        company_names = tuple(sorted(set(intent.filters.company_names)))
        company_alias = "importer" if intent.filters.company_role == "importer_company" else "exporter"
        country_alias = "import_country" if intent.filters.company_role == "importer_company" else "export_country"
        synthetic = intent.retrieval_filter.is_synthetic

        if intent.need_trade_data:
            metrics = tuple(sorted(set(intent.constraints.metrics)))
            if len(metrics) != 1 or metrics[0] not in {"quantity", "trade_amount"}:
                raise InvalidIntentContract("SQL intent requires exactly one supported metric")
            metric = metrics[0]
            dimensions = tuple(sorted("trade_date" if item == "time" else item for item in set(intent.constraints.dimensions)))
            items.append(EvidenceRequirement(requirement_id="sql.trade_aggregate", branch="sql", fact_types=(metric,), allowed_source_types=("sql",), minimum_independent_sources=1,
                maximum_age_days=policy.lead_sql_maximum_age_days if intent.kind == "lead_assessment" and intent.time_range.end is None else None,
                required_entity_ids=entity_ids, required_company_names=company_names, required_country_codes=countries, required_hs_codes=hs_codes,
                required_metrics=(metric,), required_dimensions=dimensions, required_time_grain=intent.time_range.grain,
                required_entity_column=f"{company_alias}.id",
                required_company_column=f"{company_alias}.company_name",
                required_country_column=f"{country_alias}.country_code",
                required_hs_column="hs.hs_code",
                require_currency=metric == "trade_amount", require_unit=metric == "quantity", temporal_start=intent.time_range.start,
                temporal_end=intent.time_range.end, require_synthetic=synthetic))

        if intent.need_external_intel:
            explicit_sources = tuple(sorted(item.value for item in intent.retrieval_filter.source_types))
            start = intent.retrieval_filter.published_after
            end = intent.retrieval_filter.published_before
            if start is None and end is not None:
                start = datetime.min.replace(tzinfo=end.tzinfo)
            if end is None and start is not None:
                end = datetime.max.replace(tzinfo=start.tzinfo)
            facts = _required_rag_facts(intent, policy)
            atomic = tuple(product(facts, entity_ids or (None,), company_names or (None,), countries or (None,), hs_codes or (None,)))
            fact_counts = {fact: sum(item[0] == fact for item in atomic) for fact in facts}
            fact_indexes = {fact: 0 for fact in facts}
            for fact, entity, company, country, hs_code in atomic:
                sources = policy.sources_by_fact.get(fact)
                if not sources:
                    raise InvalidIntentContract(f"unknown fact type {fact!r} has no policy")
                index = fact_indexes[fact]
                fact_indexes[fact] += 1
                requirement_id = f"rag.{fact}" if fact_counts[fact] == 1 else f"rag.{fact}.{index:04d}"
                items.append(EvidenceRequirement(requirement_id=requirement_id, branch="rag", fact_types=(fact,), allowed_source_types=sources,
                    requested_source_types=explicit_sources,
                    minimum_independent_sources=policy.lead_status_independent_sources if intent.kind == "lead_assessment" and fact == "company_status" else policy.external_independent_sources,
                    maximum_age_days=None if start is not None else policy.age_by_fact[fact], require_resolved_entity=entity is not None or company is not None,
                    required_entity_ids=(entity,) if entity else (), required_company_names=(company,) if company else (), required_country_codes=(country,) if country else (), required_hs_codes=(hs_code,) if hs_code else (),
                    temporal_start=start, temporal_end=end, require_synthetic=synthetic))

        ordered = tuple(sorted(items, key=lambda item: item.requirement_id))
        return cls._from_semantics({
            "policy_version": policy.policy_version,
            "policy_fingerprint": policy.policy_fingerprint,
            "intent_kind": intent.kind,
            "required_branches": tuple(sorted({item.branch for item in ordered})),
            "items": ordered,
        })


def _requirements_fingerprint(requirements: EvidenceRequirements) -> str:
    payload = requirements.model_dump(mode="python", exclude={"requirements_fingerprint"})
    return _requirements_payload_fingerprint(payload)


def _requirements_payload_fingerprint(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        _fingerprint_value(payload), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256(canonical.encode()).hexdigest()


def _fingerprint_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return _fingerprint_value(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {str(key): _fingerprint_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_fingerprint_value(item) for item in value]
    if isinstance(value, datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, date):
        return {"$date": value.isoformat()}
    return value


def validated_intent(value: QueryIntent) -> QueryIntent:
    if type(value) is not QueryIntent:
        raise InvalidIntentContract("intent must be an exact QueryIntent")
    try:
        intent = QueryIntent.model_validate(value.model_dump(mode="python", round_trip=True))
    except Exception as error:
        raise InvalidIntentContract("intent failed strict revalidation") from error
    external_tokens = ("官网", "website", "扩产", "状态", "新闻", "regulation", "法规")
    trade_kinds = {"top_importers", "top_exporters", "company_trend", "country_hs_activity", "lead_assessment"}
    expected_trade = intent.kind in trade_kinds
    expected_external = intent.kind in {"lead_assessment", "external_intelligence"} or (expected_trade and any(token in intent.question.casefold() for token in external_tokens))
    if intent.kind == "out_of_scope":
        expected_trade = expected_external = False
    if intent.need_trade_data is not expected_trade or intent.need_external_intel is not expected_external:
        raise InvalidIntentContract("intent route flags differ from canonical kind/question routes")
    return intent


def _union(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(set(left) | set(right)))


def _required_rag_facts(intent: QueryIntent, policy: SufficiencyPolicy) -> tuple[str, ...]:
    explicit = {item.value for item in intent.retrieval_filter.fact_types}
    if intent.kind == "lead_assessment":
        explicit.add("company_status")
    if explicit:
        return tuple(sorted(explicit))
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
    if matched:
        return matched
    raise InvalidIntentContract("external intent has no supported fact requirement")
