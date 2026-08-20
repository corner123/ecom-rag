"""Explicit, evaluation-only retrieval profiles.

The production index deliberately keeps its constructor defaults.  These
profiles exist only so development experiments can vary hybrid-RRF parameters
without silently changing the runtime service or hiding the tested settings in
ad-hoc factory code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Sequence


_PROFILE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class RetrievalExperimentProfile:
    """One named hybrid-RRF configuration used by offline evaluation only."""

    name: str
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    partition_candidate_multiplier: int = 3
    federated_candidate_multiplier: int = 3

    def __post_init__(self) -> None:
        if not _PROFILE_NAME_RE.fullmatch(self.name):
            raise ValueError(
                "retrieval profile name must use lowercase letters, digits, '.', "
                "'_' or '-'"
            )
        for field_name, value in (
            ("dense_weight", self.dense_weight),
            ("bm25_weight", self.bm25_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be a finite non-negative number")
        if self.dense_weight == 0 and self.bm25_weight == 0:
            raise ValueError("at least one retrieval weight must be positive")
        for field_name, value in (
            (
                "partition_candidate_multiplier",
                self.partition_candidate_multiplier,
            ),
            (
                "federated_candidate_multiplier",
                self.federated_candidate_multiplier,
            ),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")

    def as_metadata(self) -> dict[str, object]:
        """Return the stable fields that must accompany every experiment."""

        return {
            "retrieval_profile_schema": "engineering-retrieval-profile/v1",
            "retrieval_profile": self.name,
            "retrieval_family": "hybrid_rrf",
            "hybrid_dense_weight": self.dense_weight,
            "hybrid_bm25_weight": self.bm25_weight,
            "partition_candidate_multiplier": self.partition_candidate_multiplier,
            "federated_candidate_multiplier": self.federated_candidate_multiplier,
        }


PRODUCTION_DEFAULT_PROFILE = RetrievalExperimentProfile(
    name="hybrid-equal-cm3",
)


def build_retrieval_profile_grid(
    *,
    dense_weights: Sequence[float],
    bm25_weights: Sequence[float],
    candidate_multipliers: Sequence[int],
) -> tuple[RetrievalExperimentProfile, ...]:
    """Build a deterministic development grid without selecting a winner.

    The same multiplier is applied at the partition and federated layers.  A
    caller that needs asymmetric budgets should construct profiles explicitly.
    The returned order follows the caller-provided sequences so experiment
    reports remain stable and easy to diff.
    """

    profiles: list[RetrievalExperimentProfile] = []
    seen_names: set[str] = set()
    for dense_weight in dense_weights:
        for bm25_weight in bm25_weights:
            for multiplier in candidate_multipliers:
                name = (
                    f"hybrid-dw{_number_label(dense_weight)}-"
                    f"bw{_number_label(bm25_weight)}-cm{multiplier}"
                )
                profile = RetrievalExperimentProfile(
                    name=name,
                    dense_weight=dense_weight,
                    bm25_weight=bm25_weight,
                    partition_candidate_multiplier=multiplier,
                    federated_candidate_multiplier=multiplier,
                )
                if profile.name in seen_names:
                    raise ValueError(
                        "retrieval profile grid produced duplicate names; use "
                        "distinct numeric values"
                    )
                seen_names.add(profile.name)
                profiles.append(profile)
    if not profiles:
        raise ValueError("retrieval profile grid cannot be empty")
    return tuple(profiles)


def validate_unique_profiles(
    profiles: Iterable[RetrievalExperimentProfile],
) -> tuple[RetrievalExperimentProfile, ...]:
    """Materialize profiles and reject report-key collisions."""

    materialized = tuple(profiles)
    if not materialized:
        raise ValueError("at least one retrieval profile is required")
    names = [profile.name for profile in materialized]
    if len(names) != len(set(names)):
        raise ValueError("retrieval profile names must be unique")
    return materialized


def _number_label(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("retrieval grid weights must be finite")
    return format(number, ".12g").replace("-", "m").replace(".", "p")
