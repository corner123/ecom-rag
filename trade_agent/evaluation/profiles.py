"""Explicit component ablations with a common retrieval budget."""
from dataclasses import dataclass
from enum import Enum


class EvaluationArm(str, Enum):
    DENSE = 'dense'
    BM25 = 'bm25'
    HYBRID_RRF = 'hybrid_rrf'
    WRRF = 'wrrf'
    WRRF_FILTER = 'wrrf_filter'
    FULL_RERANK = 'full_rerank'
    FULL_E2E = 'full_e2e'


COMPONENTS = ('dense', 'bm25', 'rrf', 'wrrf', 'filter', 'reranker', 'generation')


@dataclass(frozen=True)
class EvaluationBudget:
    candidate_limit: int = 100
    top_k: int = 10

    def __post_init__(self):
        if (type(self.candidate_limit) is not int or type(self.top_k) is not int
                or not 1 <= self.top_k <= self.candidate_limit <= 512):
            raise ValueError('budget requires 1 <= top_k <= candidate_limit <= 512')


@dataclass(frozen=True)
class ArmProfile:
    arm: EvaluationArm
    enabled_components: tuple[str, ...]
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    rrf_k: int = 60
    version: str = 'trade-eval-arms/v1'


def profile_for(arm: EvaluationArm) -> ArmProfile:
    arm = EvaluationArm(arm)
    if arm in (EvaluationArm.DENSE, EvaluationArm.BM25):
        return ArmProfile(arm, (arm.value,))
    fusion = 'rrf' if arm is EvaluationArm.HYBRID_RRF else 'wrrf'
    components = ('dense', 'bm25', fusion)
    if arm in (EvaluationArm.WRRF_FILTER, EvaluationArm.FULL_RERANK, EvaluationArm.FULL_E2E):
        components += ('filter',)
    if arm in (EvaluationArm.FULL_RERANK, EvaluationArm.FULL_E2E):
        components += ('reranker',)
    if arm is EvaluationArm.FULL_E2E:
        components += ('generation',)
    return ArmProfile(arm, components, sparse_weight=1.0 if fusion == 'rrf' else 0.7)
