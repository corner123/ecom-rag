from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime

from trade_agent.entities.models import ArbitrationFact, ArbitrationOutcome, EvidenceCluster


class ConflictArbitrator:
    """Explain temporal, source-authority, directness, and independence conflicts."""

    def arbitrate(
        self,
        clusters: Sequence[EvidenceCluster],
        *,
        as_of: datetime,
    ) -> ArbitrationOutcome:
        representatives = [min(cluster.candidates, key=lambda item: (-item.source_weight, item.evidence_id)) for cluster in clusters]
        groups: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
        for index, item in enumerate(representatives):
            key = (item.entity_id, item.fact_type, item.unit or "none", item.aggregation_grain or "none")
            groups[key].append(index)

        facts: list[ArbitrationFact] = []
        conflict_groups: set[str] = set()
        for indexes in groups.values():
            active = [index for index in indexes if self._active(representatives[index], as_of)]
            expired = [index for index in indexes if index not in active]
            if len(active) > 1 and len({representatives[index].fact_value for index in active}) > 1:
                highest_weight = max(representatives[index].source_weight for index in active)
                has_independent_sources = len(
                    {representatives[index].canonical_url or representatives[index].source_url for index in active}
                ) > 1
                high_authority = highest_weight >= 0.8 and has_independent_sources
                for index in active:
                    item = representatives[index]
                    reasons = {"same_grain_temporal_conflict"}
                    if high_authority:
                        reasons.add("high_authority_conflict")
                    facts.append(
                        ArbitrationFact(
                            evidence_id=item.evidence_id,
                            entity_id=item.entity_id,
                            fact_type=item.fact_type,
                            fact_value=item.fact_value,
                            status="conflicted",
                            reasons=frozenset(reasons),
                        )
                    )
                    conflict_groups.add(f"{key_string(groups_key(groups, indexes))}:{as_of.isoformat()}")
                for index in expired:
                    item = representatives[index]
                    facts.append(self._fact(item, "superseded", {"same_grain_expired"}))
                continue
            for index in active:
                item = representatives[index]
                facts.append(self._fact(item, "supported", {"temporal_active"}))
            for index in expired:
                item = representatives[index]
                expired_status = "superseded" if len(indexes) > 1 and len(active) == 1 else "supported"
                facts.append(self._fact(item, expired_status, {"temporal_expired"} if expired_status == "superseded" else {"historical_valid_window"}))

        ordered = sorted(facts, key=lambda item: item.evidence_id)
        return ArbitrationOutcome(
            facts=tuple(ordered),
            groups=tuple(sorted(conflict_groups)),
            lead_status="block" if conflict_groups else "low",
        )

    @staticmethod
    def _active(item, as_of: datetime) -> bool:
        return item.valid_from <= as_of and (item.valid_to is None or as_of <= item.valid_to)

    @staticmethod
    def _fact(item, status: str, reasons: set[str]) -> ArbitrationFact:
        return ArbitrationFact(
            evidence_id=item.evidence_id,
            entity_id=item.entity_id,
            fact_type=item.fact_type,
            fact_value=item.fact_value,
            status=status,
            reasons=frozenset(reasons),
        )


def groups_key(groups: dict, indexes: list[int]):
    for key, values in groups.items():
        if values == indexes:
            return key
    return None


def key_string(key) -> str:
    return ":".join(str(part) for part in key)
