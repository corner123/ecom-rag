from __future__ import annotations

from collections.abc import Sequence
from difflib import SequenceMatcher

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictStr

from trade_agent.entities.models import EntityMention, EntityRecord


_EXACT_FUZZY = 0.98
_REVIEW_FUZZY = 0.90


class EntityCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_id: StrictStr
    reason: StrictStr
    confidence: StrictFloat = Field(ge=0, le=1)


class ResolutionOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    entity_id: StrictStr | None
    reason: StrictStr | None
    confidence: StrictFloat = Field(ge=0, le=1)
    candidates: tuple[EntityCandidate, ...] = ()


class EntityResolver:
    """Deterministic-first entity resolution with bounded ambiguity."""

    def __init__(self, registry: Sequence[EntityRecord]) -> None:
        self.registry = tuple(registry)

    def resolve(self, mention: EntityMention) -> ResolutionOutcome:
        if type(mention) is not EntityMention:
            raise TypeError("mention must be an exact EntityMention")
        if mention.registration_id:
            exact = [record for record in self.registry if record.registration_id == mention.registration_id]
            if len(exact) == 1:
                return self._resolved(exact[0], "registration_id", 1.0)
        if mention.website_domain:
            domain = mention.website_domain.casefold().removesuffix("/")
            exact = [record for record in self.registry if record.website_domain == domain]
            if len(exact) == 1:
                return self._resolved(exact[0], "website_domain", 0.99)

        aliases = {mention.name.casefold(), *(alias.casefold() for alias in mention.aliases)}
        alias_matches = [
            record
            for record in self.registry
            if record.normalized_name.casefold() in aliases
            or record.canonical_name.casefold() in aliases
            or any(alias.casefold() in aliases for alias in record.aliases)
        ]
        if len(alias_matches) > 1 and mention.country_code is None:
            return ResolutionOutcome(
                status="ambiguous",
                entity_id=None,
                reason="multiple_alias_candidates",
                confidence=max(0.9, min(0.95, len(alias_matches) / 10)),
                candidates=tuple(
                    EntityCandidate(entity_id=record.entity_id, reason="exact_alias", confidence=0.9)
                    for record in alias_matches
                ),
            )
        if mention.country_code:
            country_matches = [
                record
                for record in self.registry
                if record.country_code == mention.country_code
                and (
                    record.normalized_name.casefold() in aliases
                    or record.canonical_name.casefold() in aliases
                    or any(alias.casefold() in aliases for alias in record.aliases)
                )
            ]
            if len(country_matches) == 1:
                return self._resolved(country_matches[0], "exact_alias_country", 0.95)

        normalized = mention.name.casefold()
        candidates = [
            EntityCandidate(
                entity_id=record.entity_id,
                reason="normalized_name_country" if record.country_code == mention.country_code else "normalized_name",
                confidence=0.90 if record.country_code == mention.country_code else 0.70,
            )
            for record in self.registry
            if SequenceMatcher(None, normalized, record.normalized_name.casefold()).ratio() >= _REVIEW_FUZZY
        ]
        if len(candidates) == 1 and candidates[0].confidence >= _EXACT_FUZZY:
            return self._resolved_by_candidate(candidates[0], "bounded_fuzzy", candidates[0].confidence)
        if len(candidates) == 1:
            return ResolutionOutcome(
                status="review",
                entity_id=None,
                reason="bounded_fuzzy_requires_review",
                confidence=candidates[0].confidence,
                candidates=candidates,
            )
        if len(candidates) > 1:
            return ResolutionOutcome(
                status="ambiguous" if mention.country_code is None else "review",
                entity_id=None,
                reason="multiple_fuzzy_candidates",
                confidence=max(candidate.confidence for candidate in candidates),
                candidates=tuple(sorted(candidates, key=lambda item: item.entity_id)),
            )
        return ResolutionOutcome(status="unresolved", entity_id=None, reason="no_safe_match", confidence=0.0)

    @staticmethod
    def _resolved(record: EntityRecord, reason: str, confidence: float) -> ResolutionOutcome:
        return ResolutionOutcome(
            status="resolved",
            entity_id=record.entity_id,
            reason=reason,
            confidence=confidence,
        )

    @staticmethod
    def _resolved_by_candidate(candidate: EntityCandidate, reason: str, confidence: float) -> ResolutionOutcome:
        return ResolutionOutcome(
            status="resolved",
            entity_id=candidate.entity_id,
            reason=reason,
            confidence=confidence,
        )
