"""High-level orchestration for the engineering knowledge RAG."""

from .grounding import GroundedAnswerer
from .deepseek_generation import (
    DeepSeekGenerationSettings,
    DeepSeekGroundedGenerator,
    build_deepseek_generator_from_env,
    build_grounded_answerer_from_env,
)
from .index import (
    DenseVectorRetriever,
    EngineeringIndex,
    HybridPartitionRetriever,
    infer_partition,
)
from .models import (
    AnswerOutcome,
    EvidenceCitation,
    GenerationStatus,
    RetrievedEvidence,
    RetrievalOutcome,
)
from .service import EngineeringRAGService
from .sufficiency import EvidenceSufficiencyGuard
from .workflows import (
    build_engineering_index,
    load_engineering_service,
    query_engineering_knowledge,
    sync_engineering_sources,
)

__all__ = [
    "AnswerOutcome",
    "DenseVectorRetriever",
    "DeepSeekGenerationSettings",
    "DeepSeekGroundedGenerator",
    "EngineeringIndex",
    "EngineeringRAGService",
    "EvidenceSufficiencyGuard",
    "EvidenceCitation",
    "GenerationStatus",
    "GroundedAnswerer",
    "HybridPartitionRetriever",
    "RetrievalOutcome",
    "RetrievedEvidence",
    "infer_partition",
    "build_engineering_index",
    "build_deepseek_generator_from_env",
    "build_grounded_answerer_from_env",
    "load_engineering_service",
    "query_engineering_knowledge",
    "sync_engineering_sources",
]
