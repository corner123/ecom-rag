from __future__ import annotations

from collections.abc import Sequence
import hashlib
import re

from trade_agent.entities.models import EvidenceCandidate, EvidenceCluster


_TOKEN = re.compile(r"[a-z0-9]+")
_NEAR_DUPLICATE_THRESHOLD = 0.75


class EvidenceDeduplicator:
    """Cluster syndication, exact identity, and bounded lexical near duplicates."""

    def cluster(self, candidates: Sequence[EvidenceCandidate]) -> list[EvidenceCluster]:
        if isinstance(candidates, (str, bytes)):
            raise TypeError("candidates must be a sequence")
        items = list(candidates)
        if len({item.evidence_id for item in items}) != len(items):
            raise ValueError("candidates contain duplicate evidence IDs")
        parent = list(range(len(items)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                return
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

        reasons: dict[tuple[int, int], set[str]] = {}
        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                edge_reasons: set[str] = set()
                left_item, right_item = items[left], items[right]
                if left_item.canonical_url and left_item.canonical_url == right_item.canonical_url:
                    edge_reasons.add("canonical_url")
                if left_item.content_hash == right_item.content_hash:
                    edge_reasons.add("content_hash")
                if (
                    left_item.syndication_group_id
                    and left_item.syndication_group_id == right_item.syndication_group_id
                ):
                    edge_reasons.add("syndication_lineage")
                if not edge_reasons and self._jaccard(left_item.content, right_item.content) >= _NEAR_DUPLICATE_THRESHOLD:
                    edge_reasons.add("near_duplicate")
                if edge_reasons:
                    union(left, right)
                    reasons[(find(left), find(right))] = set()

        groups: dict[int, list[int]] = {}
        for index, item in enumerate(items):
            groups.setdefault(find(index), []).append(index)
        clusters: list[EvidenceCluster] = []
        for root in sorted(groups):
            indexes = sorted(groups[root])
            members = [items[index] for index in indexes]
            cluster_reasons: set[str] = set()
            for first_index, left in enumerate(indexes):
                for right in indexes[first_index + 1 :]:
                    if items[left].canonical_url and items[left].canonical_url == items[right].canonical_url:
                        cluster_reasons.add("canonical_url")
                    if items[left].content_hash == items[right].content_hash:
                        cluster_reasons.add("content_hash")
                    if (
                        items[left].syndication_group_id
                        and items[left].syndication_group_id == items[right].syndication_group_id
                    ):
                        cluster_reasons.add("syndication_lineage")
                    if self._jaccard(items[left].content, items[right].content) >= _NEAR_DUPLICATE_THRESHOLD:
                        cluster_reasons.add("near_duplicate")
            canonical = min(member.evidence_id for member in members)
            cluster_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
            clusters.append(
                EvidenceCluster(
                    cluster_id=f"dedupe_{cluster_id}",
                    candidates=tuple(sorted(members, key=lambda item: item.evidence_id)),
                    reasons=frozenset(cluster_reasons),
                )
            )
        return clusters

    @staticmethod
    def _jaccard(left: str, right: str) -> float:
        left_tokens = set(_TOKEN.findall(left.casefold()))
        right_tokens = set(_TOKEN.findall(right.casefold()))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
