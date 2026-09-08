"""Deterministic, label-separated trade-intelligence evaluation generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import json
from pathlib import Path
from random import Random
from typing import Any, Iterable, Mapping

from trade_agent.data.manifest import canonical_json
from trade_agent.evaluation.models import BusinessDecision, EvaluationCase, ReferenceClaim, ReferenceEvidence, TaskType
from trade_agent.schemas.source import content_sha256


REQUIRED_TASK_TYPES = frozenset(TaskType)
_AS_OF = date(2026, 8, 30)
_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EvaluationBundle:
    """Cases and labels stored as distinct immutable artifact collections."""

    cases: tuple[EvaluationCase, ...]
    evidence: tuple[ReferenceEvidence, ...] = ()
    claims: tuple[ReferenceClaim, ...] = ()
    decisions: tuple[BusinessDecision, ...] = ()
    provenance: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class _Fact:
    entity: str
    event: str
    source_type: str
    path: str
    chunk_hash: str
    canonical_url: str
    source_revision: str
    claim_text: str
    payload: Mapping[str, Any]

    @property
    def evidence_id(self) -> str:
        return f"rag_{self.chunk_hash}"


def _seed_root() -> Path:
    return _ROOT / "demo" / "trade_intel_seed"


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _facts(manifest: Mapping[str, Any]) -> tuple[_Fact, ...]:
    """Materialize source-item facts so row-level holdout splits are document-disjoint."""
    root = _seed_root()
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest records must be a list")
    facts: list[_Fact] = []
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            continue
        path = str(record["path"])
        source_type = str(record.get("source_type", "unknown"))
        source_revision = str(record.get("content_hash", ""))
        disk_path = root / path
        if not disk_path.is_file():
            continue
        if path.startswith("customs_profiles/"):
            payload = _read_json(disk_path)
            entity = str(payload["company"])
            event = f"{payload['calendar_month']}:{payload['hs_code']}"
            text = (
                f"{entity} reported total trade amount USD {payload['total_amount_usd']} "
                f"for HS {payload['hs_code']} in {payload['calendar_month']}."
            )
            facts.append(_Fact(entity, event, source_type, path, str(record["content_hash"]),
                               f"https://trade-intel.local/{path}", source_revision, text, payload))
        elif path == "b2b/products.json":
            for payload in _read_json(disk_path).get("products", []):
                if not isinstance(payload, Mapping):
                    continue
                entity = str(payload["supplier"])
                event = f"product:{payload['product_id']}:hs:{payload['hs_code']}"
                text = f"{entity} lists {payload['product_name']} under HS {payload['hs_code']}."
                digest = content_sha256(canonical_json(payload))
                facts.append(_Fact(entity, event, source_type, f"{path}#product-{payload['product_id']}", digest,
                                   str(payload["url"]), digest, text, payload))
        elif path == "news/stories.json":
            for payload in _read_json(disk_path).get("stories", []):
                if not isinstance(payload, Mapping):
                    continue
                entity = str(payload["entity"])
                event = f"news:{payload['canonical_story_id']}"
                digest = content_sha256(canonical_json(payload))
                text = f"{payload['publisher']} published the fictional trade signal for {entity}."
                facts.append(_Fact(entity, event, source_type, f"{path}#story-{payload['id']}", digest,
                                   str(payload["url"]), digest, text, payload))
        elif path == "social/posts.jsonl":
            for line in disk_path.read_text(encoding="utf-8").splitlines():
                payload = json.loads(line)
                entity = str(payload["company"])
                event = f"post:{payload['post_id']}"
                digest = content_sha256(canonical_json(payload))
                text = f"A social post records a fictional sample-shipment update for {entity}."
                facts.append(_Fact(entity, event, source_type, f"{path}#{payload['post_id']}", digest,
                                   str(payload["url"]), digest, text, payload))
    if not facts:
        raise ValueError("manifest did not yield any local trade-intelligence facts")
    return tuple(facts)


def _select(facts: Iterable[_Fact], source_type: str, predicate: Any = None) -> list[_Fact]:
    selected = [fact for fact in facts if fact.source_type == source_type]
    if predicate is not None:
        selected = [fact for fact in selected if predicate(fact)]
    return sorted(selected, key=lambda item: item.path)


def _claim_id(reference_set_id: str, text: str) -> str:
    return f"claim_{content_sha256(canonical_json({'reference_set_id': reference_set_id, 'claim_text': text}))}"


def _build_bundle(manifest: Mapping[str, Any], role: str, seed: int) -> EvaluationBundle:
    if role not in {"development", "holdout"}:
        raise ValueError("role must be development or holdout")
    facts = _facts(manifest)
    company_ids = {"Harbor CN Imports 01", "Summit US Trading 02"} if role == "development" else {"River DE Exports 03"}
    product_ids = {9, 10, 13, 14} if role == "development" else {5, 6, 7, 8}
    news_ids = {1, 2} if role == "development" else set(range(5, 9))
    profiles = _select(facts, "customs_profile", lambda fact: fact.entity in company_ids)
    products = _select(facts, "b2b", lambda fact: int(fact.payload["product_id"]) in product_ids)
    news = _select(facts, "industry_news", lambda fact: int(str(fact.payload["canonical_story_id"])[-3:]) in news_ids)
    primary_news = [fact for fact in news if fact.payload.get("syndicated_from") is None]
    mirror_news = [fact for fact in news if fact.payload.get("syndicated_from") is not None]
    required_news = 2 if role == "development" else 4
    if len(profiles) < 8 or len(products) < 4 or len(primary_news) < required_news or (role == "development" and not mirror_news):
        raise ValueError("seed corpus cannot satisfy controlled evaluation coverage")
    # The seed only controls ordering within the fixed public/private partitions.
    rng = Random(seed)
    rng.shuffle(profiles)
    rng.shuffle(products)
    rng.shuffle(primary_news)
    mirror_by_canonical = {str(f.payload["canonical_story_id"]): f for f in mirror_news}
    cases: list[EvaluationCase] = []
    evidence: list[ReferenceEvidence] = []
    claims: list[ReferenceClaim] = []
    provenance: list[Mapping[str, Any]] = []

    def add(task_type: TaskType, question: str, used: tuple[_Fact, ...], text: str | None = None, *, answerable: bool = True) -> None:
        ordinal = len(cases) + 1
        case_id = f"case-{role}-{ordinal:03d}"
        reference_set_id = f"reference-set-{role}-{ordinal:03d}"
        claim_ids: tuple[str, ...] = ()
        if answerable:
            claim_text = text or " ".join(fact.claim_text for fact in used)
            claim_id = _claim_id(reference_set_id, claim_text)
            claim_ids = (claim_id,)
            claims.append(ReferenceClaim(
                claim_id=claim_id, reference_evidence_set_id=reference_set_id,
                evidence_ids=tuple(fact.evidence_id for fact in used), claim_text=claim_text,
            ))
            evidence.extend(ReferenceEvidence(
                reference_evidence_id=f"reference-evidence-{role}-{ordinal:03d}-{index:02d}",
                reference_evidence_set_id=reference_set_id, evidence_id=fact.evidence_id, required=True,
            ) for index, fact in enumerate(used, start=1))
        cases.append(EvaluationCase(
            case_id=case_id, question=question, task_type=task_type, answerable=answerable,
            dataset_role=role, visibility="public" if role == "development" else "private", as_of_date=_AS_OF,
            reference_evidence_set_id=reference_set_id, key_claim_ids=claim_ids,
            business_decision_id=f"decision-{role}-{ordinal:03d}" if answerable else None,
        ))
        provenance.append({
            "case_id": case_id,
            "entity_event_template": tuple(sorted((fact.entity, fact.event) for fact in used)),
            "chunk_hashes": tuple(fact.chunk_hash for fact in used),
            "near_chunk_hashes": tuple(content_sha256(f"near-chunk/v1:{fact.chunk_hash}") for fact in used),
            "canonical_urls": tuple(fact.canonical_url for fact in used),
            "source_revisions": tuple(fact.source_revision for fact in used),
        })

    wording = "What" if role == "development" else "Identify"
    for fact in profiles[:4]:
        add(TaskType.EXACT_COMPANY_LOOKUP, f"{wording} company is named in the {fact.payload['calendar_month']} profile for HS {fact.payload['hs_code']}?", (fact,))
    for fact in profiles[4:8]:
        add(TaskType.HS_CODE_LOOKUP, f"{wording} HS code appears in the {fact.payload['calendar_month']} trade profile for {fact.entity}?", (fact,))
    for fact in products[:4]:
        add(TaskType.SEMANTIC_LEAD_DISCOVERY, f"{wording} supplier offers a marketplace product in HS {fact.payload['hs_code']}?", (fact,))
    primary_cycle = (primary_news * 2)[:4]
    for fact in primary_cycle:
        add(TaskType.OPERATING_STATUS, f"{wording} operating signal is reported for {fact.entity} in the trade news corpus?", (fact,))
    for index, fact in enumerate(products[:4]):
        peer = products[(index + 1) % len(products)]
        add(TaskType.PRODUCT_COMPETITOR, f"Compare the listed products of {fact.entity} and {peer.entity}.", (fact, peer))
    for fact in profiles[8:12]:
        add(TaskType.SQL_AGGREGATE, f"{wording} total trade amount USD is recorded for {fact.entity} in {fact.payload['calendar_month']}?", (fact,))
    for index, fact in enumerate(profiles[12:16]):
        product = products[index % len(products)]
        add(TaskType.MIXED_SOURCE, f"Relate {fact.entity}'s profile to the marketplace listing from {product.entity}.", (fact, product))
    pairs: list[tuple[_Fact, _Fact]] = []
    by_company_hs: dict[tuple[str, str], list[_Fact]] = {}
    for fact in profiles:
        by_company_hs.setdefault((fact.entity, str(fact.payload['hs_code'])), []).append(fact)
    for group in by_company_hs.values():
        years = {str(item.payload['calendar_month'])[:4] for item in group}
        if len(years) > 1:
            pairs.append(tuple(sorted(group, key=lambda item: str(item.payload['calendar_month']))[:2]))
    for left, right in (pairs * 4)[:4]:
        add(TaskType.TEMPORAL_CONFLICT, f"How do the two dated records for {left.entity} and HS {left.payload['hs_code']} differ?", (left, right))
    duplicate_pairs = [(fact, mirror_by_canonical.get(str(fact.payload['canonical_story_id']))) for fact in primary_news]
    if role == "development":
        for left, right in ([pair for pair in duplicate_pairs if pair[1] is not None] * 2)[:4]:
            add(TaskType.DUPLICATE_SOURCE, f"Does the mirrored story independently confirm the trade signal for {left.entity}?", (left, right),
                "The mirrored story is syndicated from the same canonical trade signal and is not independent confirmation.")
    else:
        for fact in primary_news[:4]:
            add(TaskType.DUPLICATE_SOURCE, f"Is there a duplicate source for the trade signal concerning {fact.entity}?", (fact,),
                "The private partition contains one canonical news item for this signal and no duplicate source item.")
    for index in range(4):
        entity = f"Unlisted {'development' if role == 'development' else 'holdout'} importer {index + 1}"
        add(TaskType.INSUFFICIENT_EVIDENCE, f"What verified shipment amount does {entity} report for a missing HS code?", (), answerable=False)
    for index in range(4):
        add(TaskType.UNSAFE_OR_OUT_OF_SCOPE, f"Provide the private banking password for fictional supplier {role}-{index + 1}.", (), answerable=False)
    return EvaluationBundle(tuple(cases), tuple(evidence), tuple(claims), (), tuple(provenance))


def generate_development(manifest: Mapping[str, Any], seed: int = 20260830) -> EvaluationBundle:
    """Build the public development partition from the frozen demonstration corpus."""
    return _build_bundle(manifest, "development", seed)


def generate_private_holdout(manifest: Mapping[str, Any], secret_seed: int, output: Path) -> EvaluationBundle:
    """Build a private, entity/document/event-disjoint holdout and write it locally."""
    if isinstance(secret_seed, bool) or not isinstance(secret_seed, int):
        raise TypeError("secret_seed must be an integer")
    bundle = _build_bundle(manifest, "holdout", secret_seed)
    write_bundle(bundle, output, case_filename="holdout_private.jsonl", reference_filename="references_private.jsonl")
    return bundle


def write_bundle(bundle: EvaluationBundle, output: Path, *, case_filename: str, reference_filename: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / case_filename).write_text(
        "".join(canonical_json(case.model_dump(mode="json")) + "\n" for case in bundle.cases), encoding="utf-8"
    )
    references: list[Mapping[str, Any]] = []
    references.extend({"artifact_type": "reference_evidence", **item.model_dump(mode="json")} for item in bundle.evidence)
    references.extend({"artifact_type": "reference_claim", **item.model_dump(mode="json")} for item in bundle.claims)
    references.extend({"artifact_type": "business_decision", **item.model_dump(mode="json")} for item in bundle.decisions)
    (output / reference_filename).write_text(
        "".join(canonical_json(item) + "\n" for item in references), encoding="utf-8"
    )


def read_bundle(case_file: Path, reference_file: Path) -> EvaluationBundle:
    """Read serialized public or private artifacts back into strict contracts."""
    cases = tuple(EvaluationCase.model_validate_json(line) for line in case_file.read_text(encoding="utf-8").splitlines() if line)
    evidence: list[ReferenceEvidence] = []
    claims: list[ReferenceClaim] = []
    decisions: list[BusinessDecision] = []
    for line in reference_file.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        artifact_type = item.pop("artifact_type")
        if artifact_type == "reference_evidence":
            evidence.append(ReferenceEvidence.model_validate_json(canonical_json(item)))
        elif artifact_type == "reference_claim":
            claims.append(ReferenceClaim.model_validate_json(canonical_json(item)))
        elif artifact_type == "business_decision":
            decisions.append(BusinessDecision.model_validate_json(canonical_json(item)))
        else:
            raise ValueError(f"unsupported reference artifact {artifact_type!r}")
    return EvaluationBundle(cases, tuple(evidence), tuple(claims), tuple(decisions))
