"""Strict immutable contracts shared by SQL, retrieval and generation."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
from hashlib import sha256
import json
import re
from typing import Annotated, Any, Literal, Mapping, Self
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator, model_validator


_EVIDENCE_ID = re.compile(r"(?:sql|rag)_[0-9a-f]{64}")
_CLAIM_ID = re.compile(r"claim_[0-9a-f]{64}")
_CONFLICT_ID = re.compile(r"conflict_[0-9a-f]{64}")
_BUILD_ID = re.compile(r"build_[0-9a-f]{32}")
def _identity_value(value: Any) -> Any:
    """Return a JSON-safe, typed value for stable content addressing."""
    if isinstance(value, BaseModel):
        return _identity_value(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {str(key): _identity_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_identity_value(item) for item in value]
    if isinstance(value, datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, date):
        return {"$date": value.isoformat()}
    if type(value) is float:
        return {"$float_hex": value.hex()}
    return value


def _identity_hash(branch: Literal["sql", "rag"], identity: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {"branch": branch, "identity_version": "evidence-v2", "payload": _identity_value(identity)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return branch + "_" + sha256(canonical.encode("utf-8")).hexdigest()


def sql_evidence_id(*, identity: Mapping[str, Any]) -> str:
    """Hash a length-safe canonical SQL provenance and semantic payload."""
    return _identity_hash("sql", identity)


def retrieval_evidence_id(*, identity: Mapping[str, Any]) -> str:
    """Hash a published RAG locator and its public semantic payload."""
    return _identity_hash("rag", identity)


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Retain validation when callers derive a changed immutable value."""
        if update is None:
            return super().model_copy(deep=deep)
        values = self.model_dump(mode="python")
        if deep:
            values = deepcopy(values)
        values.update(update)
        return type(self).model_validate(values)


def _nonblank(value: str | None) -> str | None:
    if value is not None and not value.strip():
        raise ValueError("value must not be blank")
    return value


def canonical_public_url(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("public source URL must be credential-free HTTP(S)")
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("public source URL has an invalid port") from exc
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", "", ""))


def _public_url(value: str | None) -> str | None:
    canonical = canonical_public_url(value)
    if canonical != value:
        raise ValueError("public source URL must already be canonical and omit query or fragment")
    return value


def _evidence_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    if len(set(values)) != len(values):
        raise ValueError("evidence_ids must be unique")
    if any(not _EVIDENCE_ID.fullmatch(value) for value in values):
        raise ValueError("evidence_ids contain an invalid Evidence identity")
    return values


class RawRecordLocator(_Contract):
    """The reviewed composite unique key for one source trade record."""
    source_id: StrictInt = Field(gt=0)
    raw_record_id: StrictStr = Field(min_length=1, max_length=100)


class EvidenceLocator(_Contract):
    """Replay locator for an aggregate over a bounded SQL predicate population."""
    branch: Literal["sql"] = "sql"
    query_id: StrictStr
    table: Literal["trade_records"] = "trade_records"
    scope: Literal["bounded_predicate_population"] = "bounded_predicate_population"
    raw_record_locators: tuple[RawRecordLocator, ...]
    raw_record_locators_truncated: StrictBool = False

    @property
    def raw_record_ids(self) -> tuple[str, ...]:
        return tuple(item.raw_record_id for item in self.raw_record_locators)

    @model_validator(mode="after")
    def unique_composite_keys(self) -> Self:
        keys = tuple((item.source_id, item.raw_record_id) for item in self.raw_record_locators)
        if len(set(keys)) != len(keys):
            raise ValueError("raw record locators must be unique")
        return self


class RetrievalEvidenceLocator(_Contract):
    """Published-build locator sufficient to replay one selected chunk."""
    branch: Literal["rag"] = "rag"
    manifest_id: StrictStr
    manifest_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    build_id: StrictStr
    collection_name: StrictStr
    source_identity: StrictStr
    document_id: StrictStr
    chunk_id: StrictStr
    chunk_index: StrictInt = Field(ge=0)
    content_hash: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    source_record_id: StrictStr | None = None
    page: StrictInt | None = Field(default=None, ge=1)
    page_end: StrictInt | None = Field(default=None, ge=1)
    block: StrictInt | None = Field(default=None, ge=0)
    table: StrictStr | None = None
    section: StrictStr | None = None
    post_id: StrictStr | None = None
    row: StrictInt | None = Field(default=None, ge=0)
    profile: StrictStr | None = None

    @field_validator("manifest_id", "build_id", "collection_name", "source_identity", "document_id", "chunk_id", "source_record_id", "table", "section", "post_id", "profile")
    @classmethod
    def nonblank_values(cls, value: str | None) -> str | None:
        return _nonblank(value)

    @model_validator(mode="after")
    def valid_replay_identity(self) -> Self:
        if not _BUILD_ID.fullmatch(self.build_id):
            raise ValueError("build_id is not a published build identity")
        if self.manifest_id != self.build_id:
            raise ValueError("manifest and build identities must match")
        expected_collection = "trade_intel_chunks_" + self.build_id.removeprefix("build_")[:20]
        if self.collection_name != expected_collection:
            raise ValueError("collection name does not match build identity")
        if self.source_identity.startswith("urn:trade-agent:document:"):
            if self.source_identity != f"urn:trade-agent:document:{self.document_id}":
                raise ValueError("source_identity document URN does not match document_id")
        else:
            try:
                _public_url(self.source_identity)
            except ValueError as exc:
                raise ValueError("source_identity must be a public URL or matching document URN") from exc
        if self.page_end is not None and (self.page is None or self.page_end < self.page):
            raise ValueError("page_end requires an ordered page range")
        return self


class RetrievalComponentProvenance(_Contract):
    retriever: StrictStr
    rank: StrictInt = Field(ge=1)
    raw_score: StrictFloat | None
    retriever_weight: StrictFloat = Field(gt=0, le=1)
    relevance_contribution: StrictFloat = Field(ge=0)


class RetrievalProvenance(_Contract):
    """Ranking trace. Scores select context; they do not measure factual truth."""
    branch: Literal["rag"] = "rag"
    build_id: StrictStr
    manifest_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    collection_name: StrictStr
    profile_id: StrictStr
    profile_version: StrictStr
    planner_version: StrictStr
    filter_expression_version: StrictStr
    rank: StrictInt = Field(ge=1)
    pre_rerank_rank: StrictInt = Field(ge=1)
    components: tuple[RetrievalComponentProvenance, ...]
    fusion_score: StrictFloat
    source_prior: StrictFloat = Field(ge=0, le=1)
    rerank_score: StrictFloat | None
    score_meaning: Literal["ranking_only"] = "ranking_only"
    entity_resolution_status: StrictStr
    entity_resolution_id: StrictStr | None
    entity_resolution_reason: StrictStr | None
    entity_resolution_confidence: StrictFloat = Field(ge=0, le=1)
    dedupe_cluster_id: StrictStr
    duplicate_chunk_ids: tuple[StrictStr, ...]
    dedupe_reasons: tuple[StrictStr, ...]
    degraded_components: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def ordered_audit_values(self) -> Self:
        names = tuple(item.retriever for item in self.components)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("retrieval components must be sorted and unique")
        for values, field in ((self.duplicate_chunk_ids, "duplicate_chunk_ids"), (self.dedupe_reasons, "dedupe_reasons"), (self.degraded_components, "degraded_components")):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field} must be sorted and unique")
        return self


class SqlProvenance(_Contract):
    branch: Literal["sql"] = "sql"
    query_id: StrictStr
    normalized_sql: StrictStr
    bound_filter_names: tuple[StrictStr, ...]
    schema_fingerprint: StrictStr
    dataset_id: StrictStr
    is_synthetic: StrictBool
    effective_start_date: date
    effective_end_date: date
    aggregation_grain: tuple[StrictStr, ...]
    time_grain: StrictStr
    metric_names: tuple[StrictStr, ...]
    content_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    execution_ms: StrictFloat = Field(ge=0)
    row_count: StrictInt = Field(ge=0)
    result_hash: StrictStr
    estimated_scan_rows: StrictInt = Field(ge=0)
    max_execution_time_ms: StrictInt = Field(gt=0)
    client_timeout_ms: StrictInt = Field(gt=0)


Locator = Annotated[EvidenceLocator | RetrievalEvidenceLocator, Field(discriminator="branch")]


def _canonical_sql_content(content: str) -> tuple[str, dict[str, Any]]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("SQL Evidence content has duplicate JSON keys")
            result[key] = value
        return result

    try:
        payload = json.loads(content, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("SQL Evidence content must be canonical JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"metrics", "rows"}:
        raise ValueError("SQL Evidence content must contain only metrics and rows")
    if (
        not isinstance(payload["metrics"], list)
        or not payload["metrics"]
        or any(not isinstance(item, str) or not item.strip() for item in payload["metrics"])
        or not isinstance(payload["rows"], list)
        or any(not isinstance(row, dict) for row in payload["rows"])
    ):
        raise ValueError("SQL Evidence content has invalid metrics or rows")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if canonical != content:
        raise ValueError("SQL Evidence content must use canonical JSON serialization")
    return canonical, payload


def _row_values(rows: list[dict[str, Any]], field: str) -> tuple[str, ...]:
    values = {row[field] for row in rows if row.get(field) is not None}
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"SQL Evidence {field} values must be nonblank strings")
    return tuple(sorted(values))


def _evidence_identity(evidence: "Evidence") -> dict[str, Any]:
    common = {
        "entity_id": evidence.entity_id,
        "company_name": evidence.company_name,
        "country_code": evidence.country_code,
        "hs_code": evidence.hs_code,
        "fact_type": evidence.fact_type,
        "source_type": evidence.source_type,
        "source_id": evidence.source_id,
        "source_weight": evidence.source_weight,
        "content_sha256": sha256(evidence.content.encode("utf-8")).hexdigest(),
        "source_url": evidence.source_url,
        "canonical_url": evidence.canonical_url,
        "locator": evidence.locator,
        "raw_record_id": evidence.raw_record_id,
        "publish_time": evidence.publish_time,
        "valid_from": evidence.valid_from,
        "valid_to": evidence.valid_to,
        "time_grain": evidence.time_grain,
        "currencies": evidence.currencies,
        "units": evidence.units,
        "aggregation_grain": evidence.aggregation_grain,
        "confidence": evidence.confidence,
        "confidence_basis": evidence.confidence_basis,
        "is_synthetic": evidence.is_synthetic,
    }
    if evidence.locator.branch == "sql":
        provenance = evidence.sql_provenance
        if provenance is None:
            raise ValueError("SQL Evidence is missing SQL provenance")
        common["sql_provenance"] = provenance.model_dump(
            mode="python", exclude={"execution_ms", "estimated_scan_rows"}
        )
    return common


class Evidence(_Contract):
    """One bounded fact bundle with stable identity and replay provenance."""
    evidence_id: StrictStr
    entity_id: StrictStr | None = None
    company_name: StrictStr | None = None
    country_code: StrictStr | None = None
    hs_code: StrictStr | None = None
    fact_type: StrictStr
    source_type: StrictStr
    source_id: StrictStr
    source_weight: StrictFloat = Field(ge=0, le=1)
    content: StrictStr
    source_url: StrictStr | None = None
    canonical_url: StrictStr | None = None
    locator: Locator
    raw_record_id: StrictStr | None = None
    publish_time: datetime | None = None
    valid_from: date | datetime | None = None
    valid_to: date | datetime | None = None
    time_grain: StrictStr
    currencies: tuple[StrictStr, ...] = ()
    units: tuple[StrictStr, ...] = ()
    aggregation_grain: tuple[StrictStr, ...] = ()
    confidence: StrictFloat = Field(ge=0, le=1)
    confidence_basis: Literal["validated_query_execution", "content_hash_verified", "ocr_extraction"]
    is_synthetic: StrictBool
    retrieval_provenance: RetrievalProvenance | None = None
    sql_provenance: SqlProvenance | None = None
    conflict_group_id: StrictStr | None = None

    @field_validator("entity_id", "company_name", "country_code", "hs_code", "fact_type", "source_type", "source_id", "content", "source_url", "canonical_url", "raw_record_id", "time_grain", "conflict_group_id")
    @classmethod
    def nonblank_strings(cls, value: str | None) -> str | None:
        return _nonblank(value)

    @field_validator("source_url", "canonical_url")
    @classmethod
    def public_source_urls(cls, value: str | None) -> str | None:
        return _public_url(value)

    @model_validator(mode="after")
    def coherent_contract(self) -> Self:
        if not _EVIDENCE_ID.fullmatch(self.evidence_id):
            raise ValueError("evidence_id must be a stable SQL or RAG identity")
        if self.valid_from is not None and self.valid_to is not None:
            start = self.valid_from.date() if isinstance(self.valid_from, datetime) else self.valid_from
            end = self.valid_to.date() if isinstance(self.valid_to, datetime) else self.valid_to
            if start > end:
                raise ValueError("valid_from must not exceed valid_to")
        for values, field in ((self.currencies, "currencies"), (self.units, "units")):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field} must be sorted and unique")
        if self.locator.branch == "sql":
            if self.sql_provenance is None or self.retrieval_provenance is not None:
                raise ValueError("SQL Evidence requires only SQL provenance")
            if self.sql_provenance.query_id != self.locator.query_id:
                raise ValueError("SQL locator and provenance query IDs differ")
            canonical_content, content_payload = _canonical_sql_content(self.content)
            content_digest = sha256(canonical_content.encode("utf-8")).hexdigest()
            row_result_hash = sha256(
                json.dumps(
                    content_payload["rows"], sort_keys=True,
                    separators=(",", ":"), allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            expected_fact_type = (
                self.sql_provenance.metric_names[0]
                if len(self.sql_provenance.metric_names) == 1
                else "trade_aggregate"
            )
            if (
                self.source_id != self.sql_provenance.dataset_id
                or self.valid_from != self.sql_provenance.effective_start_date
                or self.valid_to != self.sql_provenance.effective_end_date
                or self.time_grain != self.sql_provenance.time_grain
                or self.aggregation_grain != self.sql_provenance.aggregation_grain
                or self.is_synthetic != self.sql_provenance.is_synthetic
            ):
                raise ValueError("SQL Evidence fields do not match SQL provenance")
            if (
                self.source_type != "sql"
                or self.fact_type != expected_fact_type
                or self.source_weight != 1.0
                or self.confidence != 1.0
                or self.confidence_basis != "validated_query_execution"
            ):
                raise ValueError("SQL Evidence fixed semantics do not match validated execution")
            if tuple(content_payload["metrics"]) != self.sql_provenance.metric_names:
                raise ValueError("SQL Evidence metrics do not match SQL provenance")
            if len(content_payload["rows"]) != self.sql_provenance.row_count:
                raise ValueError("SQL Evidence row count does not match SQL provenance")
            if content_digest != self.sql_provenance.content_sha256:
                raise ValueError("SQL Evidence content digest does not match SQL provenance")
            if row_result_hash != self.sql_provenance.result_hash:
                raise ValueError("SQL Evidence result hash does not match content rows")
            if self.currencies != _row_values(content_payload["rows"], "currency"):
                raise ValueError("SQL Evidence currencies do not match content rows")
            if self.units != _row_values(content_payload["rows"], "unit"):
                raise ValueError("SQL Evidence units do not match content rows")
            expected_id = sql_evidence_id(identity=_evidence_identity(self))
        elif self.retrieval_provenance is None or self.sql_provenance is not None:
            raise ValueError("RAG Evidence requires only retrieval provenance")
        elif (
            self.retrieval_provenance.build_id != self.locator.build_id
            or self.retrieval_provenance.collection_name != self.locator.collection_name
            or self.retrieval_provenance.manifest_fingerprint != self.locator.manifest_fingerprint
        ):
            raise ValueError("RAG locator and provenance build identities differ")
        else:
            if sha256(self.content.encode("utf-8")).hexdigest() != self.locator.content_hash:
                raise ValueError("RAG Evidence content hash does not match its locator")
            if self.source_id != self.locator.source_identity:
                raise ValueError("RAG Evidence source identity does not match its locator")
            if self.raw_record_id != self.locator.source_record_id:
                raise ValueError("RAG Evidence raw record identity does not match its locator")
            expected_id = retrieval_evidence_id(identity=_evidence_identity(self))
        if self.evidence_id != expected_id:
            raise ValueError("evidence_id does not match its content-addressed scope")
        return self


class Claim(_Contract):
    claim_id: StrictStr
    text: StrictStr
    status: Literal["supported", "conflicted", "stale", "insufficient", "analysis"]
    evidence_ids: tuple[StrictStr, ...] = ()
    entity_id: StrictStr | None = None
    fact_type: StrictStr | None = None
    value: StrictStr | None = None
    unit: StrictStr | None = None
    currency: StrictStr | None = None
    period_start: date | datetime | None = None
    period_end: date | datetime | None = None
    confidence: StrictFloat = Field(ge=0, le=1)

    @model_validator(mode="after")
    def valid_claim(self) -> Self:
        if not _CLAIM_ID.fullmatch(self.claim_id):
            raise ValueError("claim_id must be a stable Claim identity")
        if not self.text.strip():
            raise ValueError("claim text must not be blank")
        _evidence_ids(self.evidence_ids)
        if self.status == "supported" and not self.evidence_ids:
            raise ValueError("supported claim requires evidence")
        if self.period_start is not None and self.period_end is not None:
            start = self.period_start.date() if isinstance(self.period_start, datetime) else self.period_start
            end = self.period_end.date() if isinstance(self.period_end, datetime) else self.period_end
            if start > end:
                raise ValueError("period_start must not exceed period_end")
        return self


class Conflict(_Contract):
    conflict_id: StrictStr
    entity_id: StrictStr
    fact_type: StrictStr
    evidence_ids: tuple[StrictStr, ...] = Field(min_length=2)
    status: Literal["unresolved", "resolved"]
    valid_from: date | datetime | None = None
    valid_to: date | datetime | None = None
    unit: StrictStr | None = None
    currency: StrictStr | None = None
    aggregation_grain: tuple[StrictStr, ...] = ()
    explanation: StrictStr
    selected_evidence_ids: tuple[StrictStr, ...] = ()

    @model_validator(mode="after")
    def valid_conflict(self) -> Self:
        if not _CONFLICT_ID.fullmatch(self.conflict_id):
            raise ValueError("conflict_id must be a stable Conflict identity")
        _evidence_ids(self.evidence_ids)
        _evidence_ids(self.selected_evidence_ids)
        if not set(self.selected_evidence_ids).issubset(self.evidence_ids):
            raise ValueError("selected conflict evidence must belong to the conflict")
        if self.status == "unresolved" and self.selected_evidence_ids:
            raise ValueError("unresolved conflict cannot select factual support")
        if self.valid_from is not None and self.valid_to is not None:
            start = self.valid_from.date() if isinstance(self.valid_from, datetime) else self.valid_from
            end = self.valid_to.date() if isinstance(self.valid_to, datetime) else self.valid_to
            if start > end:
                raise ValueError("valid_from must not exceed valid_to")
        if not self.explanation.strip():
            raise ValueError("conflict explanation must not be blank")
        return self


class PublicTrace(_Contract):
    branches: tuple[Literal["sql", "rag"], ...]
    evidence_ids: tuple[StrictStr, ...]
    conflict_ids: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def valid_trace(self) -> Self:
        if self.branches != tuple(sorted(set(self.branches))):
            raise ValueError("branches must be sorted and unique")
        _evidence_ids(self.evidence_ids)
        if len(set(self.conflict_ids)) != len(self.conflict_ids) or any(not _CONFLICT_ID.fullmatch(value) for value in self.conflict_ids):
            raise ValueError("conflict_ids contain an invalid or duplicate identity")
        return self


def _as_date(value: date | datetime | None) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    return value


def _claim_matches_evidence(claim: Claim, evidence: Evidence) -> bool:
    if claim.entity_id is not None and claim.entity_id != evidence.entity_id:
        return False
    if claim.fact_type is not None and claim.fact_type != evidence.fact_type:
        return False
    if claim.currency is not None and claim.currency not in evidence.currencies:
        return False
    if claim.unit is not None and claim.unit not in evidence.units:
        return False
    claim_start, claim_end = _as_date(claim.period_start), _as_date(claim.period_end)
    evidence_start, evidence_end = _as_date(evidence.valid_from), _as_date(evidence.valid_to)
    if claim_start is None and claim_end is None:
        return True
    if claim_end is not None and evidence_start is not None and claim_end < evidence_start:
        return False
    if claim_start is not None and evidence_end is not None and claim_start > evidence_end:
        return False
    return True


class IntelligenceAnswer(_Contract):
    answer: StrictStr | None
    claims: tuple[Claim, ...]
    evidence: tuple[Evidence, ...]
    conflicts: tuple[Conflict, ...]
    refusal_reason: StrictStr | None
    degraded_components: tuple[StrictStr, ...]
    public_trace: PublicTrace

    @model_validator(mode="after")
    def references_generation_context_only(self) -> Self:
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("generation evidence IDs must be unique")
        known = set(evidence_ids)
        cited = {item for claim in self.claims for item in claim.evidence_ids}
        conflict_cited = {item for conflict in self.conflicts for item in conflict.evidence_ids}
        if not cited.issubset(known) or not conflict_cited.issubset(known):
            raise ValueError("claim or conflict cites evidence outside generation evidence")
        evidence_by_id = {item.evidence_id: item for item in self.evidence}
        for claim in self.claims:
            if claim.status != "supported":
                continue
            if not any(
                _claim_matches_evidence(claim, evidence_by_id[evidence_id])
                for evidence_id in claim.evidence_ids
            ):
                raise ValueError("supported claim has no semantically compatible Evidence")
        conflict_ids = tuple(item.conflict_id for item in self.conflicts)
        if set(self.public_trace.evidence_ids) != known or set(self.public_trace.conflict_ids) != set(conflict_ids):
            raise ValueError("public trace does not match answer evidence and conflicts")
        actual_branches = tuple(sorted({item.locator.branch for item in self.evidence}))
        if self.public_trace.branches != actual_branches:
            raise ValueError("public trace branches do not match answer evidence")
        if (self.answer is None) == (self.refusal_reason is None):
            raise ValueError("exactly one of answer or refusal_reason is required")
        if self.answer is not None and not self.answer.strip():
            raise ValueError("answer must not be blank")
        if self.refusal_reason is not None and not self.refusal_reason.strip():
            raise ValueError("refusal_reason must not be blank")
        if self.degraded_components != tuple(sorted(set(self.degraded_components))):
            raise ValueError("degraded_components must be sorted and unique")
        return self
