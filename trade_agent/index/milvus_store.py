"""Build-scoped Milvus contracts and storage primitives for trade chunks."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal, Protocol

import numpy as np

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from trade_agent.data.manifest import BuildManifest, canonical_json
from trade_agent.index.contracts import EmbeddingContract
from trade_agent.schemas.source import ChunkRecord
from trade_agent.retrieval.filters import RetrievalFilter, compile_filter_binding

try:
    from pymilvus import DataType
except ImportError:  # pragma: no cover - exercised only in non-service environments
    DataType = None

if DataType is None:
    _DATATYPES: dict[str, Any] = {}
else:
    _DATATYPES = {
        "BOOL": DataType.BOOL,
        "INT64": DataType.INT64,
        "FLOAT": DataType.FLOAT,
        "FLOAT_VECTOR": DataType.FLOAT_VECTOR,
        "VARCHAR": DataType.VARCHAR,
    }

COLLECTION_OWNER = "trade-agent"
COLLECTION_SCHEMA_VERSION = "trade-milvus-v1"
CONTRACT_COLLECTION_NAME = "trade_intel_collection_contracts_v1"
CONTRACT_COLLECTION_SCHEMA_VERSION = "trade-milvus-contract-registry-v1"
VECTOR_INDEX_NAME = "dense_vector_hnsw"
CONTRACT_VECTOR_INDEX_NAME = "contract_vector_flat"
NULL_STRING_SENTINEL = "__trade_agent_null_v1__"
NULL_EPOCH_SENTINEL = -(2**63)
MAX_VARCHAR_BYTES = 65_535

MATERIALIZED_FIELDS = (
    "chunk_id",
    "dense_vector",
    "text",
    "document_id",
    "entity_id",
    "country_code",
    "region",
    "hs_code",
    "source_type",
    "source_weight",
    "fact_type",
    "publish_time_epoch",
    "valid_to_epoch",
    "file_type",
    "is_synthetic",
    "canonical_url_hash",
    "dedupe_cluster_id",
    "metadata_json",
)

_BUILD_ID = re.compile(r"build_[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def collection_name_for_build_id(build_id: str) -> str:
    if not isinstance(build_id, str) or not _BUILD_ID.fullmatch(build_id):
        raise ValueError("build_id must be build_ followed by 32 lowercase hex characters")
    return "trade_intel_chunks_" + build_id.removeprefix("build_")[:20]


def canonical_url_hash(
    *,
    canonical_url: str | None,
    source_url: str | None,
    document_id: str,
) -> str:
    """Hash canonical URL, then source URL, then a stable document URN.

    The document URN fallback keeps the materialized field non-null without
    pretending that a source without a URL has a web locator.
    """

    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id must not be blank")
    identity = canonical_url or source_url or f"urn:trade-agent:document:{document_id}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _utf8(value: str, *, field: str, maximum: int = MAX_VARCHAR_BYTES) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if value == NULL_STRING_SENTINEL:
        raise ValueError(f"{field} collides with the reserved null sentinel")
    if len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{field} exceeds the Milvus VARCHAR byte limit")
    return value


def _nullable_string(value: object | None, *, field: str, maximum: int) -> str:
    if value is None:
        return NULL_STRING_SENTINEL
    raw = value.value if hasattr(value, "value") else str(value)
    return _utf8(raw, field=field, maximum=maximum)


def _epoch(value: datetime | None) -> int:
    if value is None:
        return NULL_EPOCH_SENTINEL
    epoch = int(value.timestamp())
    if epoch == NULL_EPOCH_SENTINEL:
        raise ValueError("datetime collides with the reserved null epoch")
    return epoch


def chunk_ids_sha256(chunk_ids: Sequence[str]) -> str:
    if isinstance(chunk_ids, (str, bytes)):
        raise TypeError("chunk_ids must be a sequence, not a string")
    canonical = json.dumps(sorted(chunk_ids), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_build_chunks(
    build: BuildManifest,
    chunks: Sequence[ChunkRecord],
) -> tuple[ChunkRecord, ...]:
    """Return chunks in manifest order after exact frozen-payload validation."""

    if not isinstance(build, BuildManifest):
        raise TypeError("build must be a validated BuildManifest")
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
        raise TypeError("chunks must be a sequence of ChunkRecord objects")
    if any(not isinstance(chunk, ChunkRecord) for chunk in chunks):
        raise TypeError("every chunk must be a ChunkRecord")
    supplied_ids = [chunk.metadata.chunk_id for chunk in chunks]
    if len(set(supplied_ids)) != len(supplied_ids):
        raise ValueError("chunks contain duplicate chunk IDs")
    if len(chunks) != len(build.chunks) or set(supplied_ids) != set(build.chunk_ids):
        raise ValueError("chunks must contain exactly the complete frozen build set")
    by_id = {chunk.metadata.chunk_id: chunk for chunk in chunks}
    ordered: list[ChunkRecord] = []
    for snapshot in build.chunks:
        supplied = by_id[snapshot.chunk_id]
        if canonical_json(supplied.model_dump(mode="json")) != snapshot.payload:
            raise ValueError("chunk does not match its frozen manifest payload")
        ordered.append(supplied)
    return tuple(ordered)


def validate_query_vector(vector: np.ndarray, *, dimension: int) -> np.ndarray:
    if type(vector) is not np.ndarray:
        raise TypeError("query vector must be a NumPy array")
    if vector.dtype != np.float32:
        raise TypeError("query vector dtype must be float32")
    if vector.ndim != 1 or vector.shape != (dimension,):
        raise ValueError("query vector shape does not match the collection dimension")
    if not np.isfinite(vector).all():
        raise ValueError("query vector must be finite")
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if not np.isfinite(norm) or not np.isclose(norm, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("query vector must be normalized")
    return vector


def materialize_chunk(chunk: ChunkRecord, vector: np.ndarray) -> dict[str, object]:
    """Create the exact Milvus row, preserving canonical full metadata JSON."""

    if not isinstance(chunk, ChunkRecord):
        raise TypeError("chunk must be a validated ChunkRecord")
    dense = validate_query_vector(vector, dimension=vector.shape[0] if vector.ndim == 1 else 0)
    if dense.shape[0] != 1024:
        raise ValueError("production chunk vectors must have dimension 1024")
    metadata = chunk.metadata
    metadata_json = canonical_json(metadata.model_dump(mode="json"))
    _utf8(metadata_json, field="metadata_json")
    return {
        "chunk_id": _utf8(metadata.chunk_id, field="chunk_id", maximum=128),
        "dense_vector": dense.tolist(),
        "text": _utf8(chunk.content, field="text"),
        "document_id": _utf8(metadata.document_id, field="document_id", maximum=128),
        "entity_id": _nullable_string(metadata.entity_id, field="entity_id", maximum=256),
        "country_code": _nullable_string(metadata.country_code, field="country_code", maximum=16),
        "region": _nullable_string(metadata.region, field="region", maximum=128),
        "hs_code": _nullable_string(metadata.hs_code, field="hs_code", maximum=32),
        "source_type": _nullable_string(metadata.source_type, field="source_type", maximum=64),
        "source_weight": float(metadata.source_weight),
        "fact_type": _nullable_string(metadata.fact_type, field="fact_type", maximum=64),
        "publish_time_epoch": _epoch(metadata.publish_time),
        "valid_to_epoch": _epoch(metadata.valid_to),
        "file_type": _nullable_string(metadata.file_type, field="file_type", maximum=64),
        "is_synthetic": metadata.is_synthetic,
        "canonical_url_hash": canonical_url_hash(
            canonical_url=str(metadata.canonical_url) if metadata.canonical_url is not None else None,
            source_url=str(metadata.source_url) if metadata.source_url is not None else None,
            document_id=metadata.document_id,
        ),
        "dedupe_cluster_id": _nullable_string(
            metadata.dedupe_cluster_id,
            field="dedupe_cluster_id",
            maximum=256,
        ),
        "metadata_json": metadata_json,
    }


class HnswConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    m: StrictInt = Field(default=32, ge=4, le=64)
    ef_construction: StrictInt = Field(default=256, ge=8, le=512)
    ef_search: StrictInt = Field(default=128, ge=1, le=1024)


class CollectionContract(BaseModel):
    """Canonical compatibility contract persisted alongside one collection."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    owner: Literal["trade-agent"] = COLLECTION_OWNER
    schema_version: Literal["trade-milvus-v1"] = COLLECTION_SCHEMA_VERSION
    collection_name: StrictStr
    build_id: StrictStr
    build_fingerprint: StrictStr
    build_chunk_count: StrictInt = Field(ge=1)
    build_chunk_ids_sha256: StrictStr
    metadata_schema_version: StrictStr
    embedding_provider: Literal["sentence-transformers"]
    embedding_model: Literal["BAAI/bge-m3"]
    embedding_requested_revision: Literal[
        "5617a9f61b028005a4858fdac845db406aefb181"
    ]
    embedding_resolved_revision: Literal[
        "5617a9f61b028005a4858fdac845db406aefb181"
    ]
    embedding_dimension: Literal[1024]
    embedding_normalized: Literal[True]
    embedding_dtype: Literal["float32"]
    embedding_library_version: StrictStr
    embedding_artifact_manifest_sha256: Literal[
        "3a862f1d0a8543acc13e9faa5e6d6d1f916ee609b264960be8337e6ba509856b"
    ]
    embedding_verified_artifact_count: Literal[10]
    vector_field: Literal["dense_vector"] = "dense_vector"
    metric_type: Literal["COSINE"] = "COSINE"
    index_type: Literal["HNSW"] = "HNSW"
    hnsw_m: StrictInt = Field(ge=4, le=64)
    hnsw_ef_construction: StrictInt = Field(ge=8, le=512)
    default_ef_search: StrictInt = Field(ge=1, le=1024)
    filter_expression_version: Literal["trade-filter-v1"] = "trade-filter-v1"
    null_string_sentinel: Literal["__trade_agent_null_v1__"] = NULL_STRING_SENTINEL
    null_epoch_sentinel: Literal[-9223372036854775808] = NULL_EPOCH_SENTINEL
    materialized_fields: tuple[StrictStr, ...] = MATERIALIZED_FIELDS

    @field_validator(
        "build_fingerprint",
        "build_chunk_ids_sha256",
        "embedding_artifact_manifest_sha256",
    )
    @classmethod
    def sha256_fields(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("field must be a lowercase SHA-256 digest")
        return value

    @field_validator("embedding_library_version", "metadata_schema_version")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must not be blank")
        return value

    @model_validator(mode="after")
    def exact_identity(self) -> "CollectionContract":
        if self.collection_name != collection_name_for_build_id(self.build_id):
            raise ValueError("collection name does not match build ID")
        if self.materialized_fields != MATERIALIZED_FIELDS:
            raise ValueError("materialized field contract does not match schema version")
        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def contract_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class CollectionContractMismatch(ValueError):
    """Persisted Milvus state is incompatible with the requested build."""


class CollectionStats(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    collection_name: StrictStr
    build_id: StrictStr
    row_count: StrictInt = Field(ge=0)
    contract_state: Literal["created", "complete"]
    contract_source: Literal["companion_collection"] = "companion_collection"
    contract_sha256: StrictStr
    contract_row_count: Literal[1] = 1
    field_names: tuple[StrictStr, ...]
    index_type: Literal["HNSW"] = "HNSW"
    metric_type: Literal["COSINE"] = "COSINE"
    hnsw_m: StrictInt
    hnsw_ef_construction: StrictInt
    loaded: StrictBool
    exact_chunk_ids: StrictBool


class DenseHit(BaseModel):
    """One dense candidate with an auditable retrieval trace."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    chunk_id: StrictStr
    record: ChunkRecord
    score: float
    rank: StrictInt = Field(ge=1)
    build_id: StrictStr
    filter_expression: StrictStr
    filter_expression_version: StrictStr

    @property
    def metadata(self) -> object:
        return self.record.metadata


class IndexWriteSummary(BaseModel):
    """Immutable result of the one allowed bulk write for a build."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    collection_name: StrictStr
    status: Literal["inserted", "noop"]
    inserted_count: StrictInt = Field(ge=0)
    row_count: StrictInt = Field(ge=0)
    contract_sha256: StrictStr


def _main_description(build_id: str) -> str:
    return f"{COLLECTION_OWNER}:{COLLECTION_SCHEMA_VERSION}:{build_id}"


def _main_field_contracts(dimension: int) -> tuple[tuple[str, str, dict[str, int], bool], ...]:
    return (
        ("chunk_id", "VARCHAR", {"max_length": 128}, True),
        ("dense_vector", "FLOAT_VECTOR", {"dim": dimension}, False),
        ("text", "VARCHAR", {"max_length": 65_535}, False),
        ("document_id", "VARCHAR", {"max_length": 128}, False),
        ("entity_id", "VARCHAR", {"max_length": 256}, False),
        ("country_code", "VARCHAR", {"max_length": 32}, False),
        ("region", "VARCHAR", {"max_length": 128}, False),
        ("hs_code", "VARCHAR", {"max_length": 32}, False),
        ("source_type", "VARCHAR", {"max_length": 64}, False),
        ("source_weight", "FLOAT", {}, False),
        ("fact_type", "VARCHAR", {"max_length": 64}, False),
        ("publish_time_epoch", "INT64", {}, False),
        ("valid_to_epoch", "INT64", {}, False),
        ("file_type", "VARCHAR", {"max_length": 64}, False),
        ("is_synthetic", "BOOL", {}, False),
        ("canonical_url_hash", "VARCHAR", {"max_length": 64}, False),
        ("dedupe_cluster_id", "VARCHAR", {"max_length": 256}, False),
        ("metadata_json", "VARCHAR", {"max_length": 65_535}, False),
    )


def _contract_field_contracts() -> tuple[tuple[str, str, dict[str, int], bool], ...]:
    return (
        ("collection_name", "VARCHAR", {"max_length": 128}, True),
        ("owner", "VARCHAR", {"max_length": 64}, False),
        ("schema_version", "VARCHAR", {"max_length": 64}, False),
        ("contract_sha256", "VARCHAR", {"max_length": 64}, False),
        ("contract_json", "VARCHAR", {"max_length": 8192}, False),
        ("state", "VARCHAR", {"max_length": 32}, False),
        ("contract_vector", "FLOAT_VECTOR", {"dim": 2}, False),
    )


def _normalized_fields(description: dict[str, Any]) -> tuple[tuple[str, str, dict[str, int], bool], ...]:
    fields: list[tuple[str, str, dict[str, int], bool]] = []
    for field in description.get("fields", []):
        datatype = field.get("type")
        name = getattr(datatype, "name", str(datatype).split(".")[-1])
        params = {key: int(value) for key, value in field.get("params", {}).items()}
        fields.append((field["name"], name, params, bool(field.get("is_primary", False))))
    return tuple(fields)


def _exact_count(client: Any, collection_name: str) -> int:
    result = client.query(
        collection_name=collection_name,
        filter="",
        output_fields=["count(*)"],
        consistency_level="Strong",
    )
    if len(result) != 1 or type(result[0].get("count(*)")) is not int:
        raise CollectionContractMismatch("Milvus Strong count response is invalid")
    return result[0]["count(*)"]


def _contract_row(client: Any, collection_name: str) -> dict[str, Any] | None:
    rows = client.query(
        collection_name=CONTRACT_COLLECTION_NAME,
        ids=[collection_name],
        output_fields=[
            "collection_name",
            "owner",
            "schema_version",
            "contract_sha256",
            "contract_json",
            "state",
        ],
        consistency_level="Strong",
    )
    if len(rows) > 1:
        raise CollectionContractMismatch("collection contract registry contains duplicate rows")
    return dict(rows[0]) if rows else None


def _contract_from_build(
    build: BuildManifest,
    embedding: EmbeddingContract,
    hnsw: HnswConfig,
) -> CollectionContract:
    return CollectionContract(
        collection_name=collection_name_for_build_id(build.build_id),
        build_id=build.build_id,
        build_fingerprint=build.fingerprint,
        build_chunk_count=len(build.chunks),
        build_chunk_ids_sha256=chunk_ids_sha256(build.chunk_ids),
        metadata_schema_version=build.metadata_schema_version,
        embedding_provider=embedding.provider,
        embedding_model=embedding.model_name,
        embedding_requested_revision=embedding.requested_revision,
        embedding_resolved_revision=embedding.resolved_revision,
        embedding_dimension=embedding.dimension,
        embedding_normalized=embedding.normalized,
        embedding_dtype=embedding.dtype,
        embedding_library_version=embedding.library_version,
        embedding_artifact_manifest_sha256=embedding.artifact_manifest_sha256,
        embedding_verified_artifact_count=embedding.verified_artifact_count,
        hnsw_m=hnsw.m,
        hnsw_ef_construction=hnsw.ef_construction,
        default_ef_search=hnsw.ef_search,
    )


def _validated_persisted_contract(
    row: dict[str, Any],
    expected: CollectionContract,
) -> CollectionContract:
    if row.get("owner") != expected.owner or row.get("schema_version") != expected.schema_version:
        raise CollectionContractMismatch("collection contract ownership mismatch")
    if row.get("contract_sha256") != expected.contract_sha256:
        raise CollectionContractMismatch("collection contract hash mismatch")
    if row.get("contract_json") != expected.canonical_json():
        raise CollectionContractMismatch("collection contract payload mismatch")
    if row.get("collection_name") != expected.collection_name:
        raise CollectionContractMismatch("collection contract identity mismatch")
    if row.get("state") not in {"created", "complete"}:
        raise CollectionContractMismatch("collection contract state mismatch")
    return expected


def _chunk_ids_match(
    client: Any,
    collection_name: str,
    expected_sha256: str,
) -> bool:
    rows = client.query(
        collection_name=collection_name,
        filter="",
        output_fields=["chunk_id"],
        limit=16_384,
        consistency_level="Strong",
    )
    chunk_ids: list[str] = []
    for row in rows:
        value = row.get("chunk_id")
        if type(value) is not str:
            return False
        chunk_ids.append(value)
    return chunk_ids_sha256(chunk_ids) == expected_sha256


class _EmbeddingManager(Protocol):
    @property
    def contract(self) -> EmbeddingContract: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...


class TradeMilvusStore:
    """Real Milvus store; construction is inert until a live contract is proven."""

    def __init__(
        self,
        *,
        client: Any,
        embedding_manager: _EmbeddingManager,
        hnsw: HnswConfig | None = None,
    ) -> None:
        self.client = client
        self.embedding_manager = embedding_manager
        self.hnsw = hnsw or HnswConfig()
        self._build: BuildManifest | None = None
        self._contract: CollectionContract | None = None

    @property
    def contract(self) -> CollectionContract:
        if self._contract is None:
            raise RuntimeError("Milvus collection contract is not open")
        return self._contract

    def create(
        self,
        build: BuildManifest,
        embedding: EmbeddingContract,
    ) -> CollectionContract:
        embedding.require_production()
        if embedding is not self.embedding_manager.contract:
            raise ValueError("embedding contract must come from this store's live manager")
        if not build.metadata_complete:
            raise ValueError("Milvus builds require complete metadata")
        contract = _contract_from_build(build, embedding, self.hnsw)
        collection_name = contract.collection_name
        if self.client.has_collection(collection_name):
            raise CollectionContractMismatch(f"Milvus collection already exists: {collection_name}")
        if self.client.has_collection(CONTRACT_COLLECTION_NAME):
            self._validate_contract_collection_schema()
        else:
            self._create_contract_collection()

        if _contract_row(self.client, collection_name) is not None:
            raise CollectionContractMismatch("collection contract row already exists")
        self._create_collection(contract, build)
        self._insert_contract_row(contract, state="created")
        self._bind(contract, build)
        return contract

    def open(
        self,
        build: BuildManifest,
        embedding: EmbeddingContract,
    ) -> CollectionContract:
        embedding.require_production()
        if embedding is not self.embedding_manager.contract:
            raise ValueError("embedding contract must come from this store's live manager")
        expected = _contract_from_build(build, embedding, self.hnsw)
        row = _contract_row(self.client, expected.collection_name)
        if row is None:
            raise CollectionContractMismatch("collection contract row is absent")
        contract = _validated_persisted_contract(row, expected)
        self._validate_collection_schema(contract)
        self._validate_collection_indexes(contract)
        self._require_loaded(contract.collection_name)
        self._bind(contract, build)
        return contract

    def replace_chunks(
        self,
        chunks: Sequence[ChunkRecord],
    ) -> IndexWriteSummary:
        contract = self._require_contract()
        build = self._build
        if build is None:
            raise RuntimeError("Milvus build manifest is not open")
        ordered = validate_build_chunks(build, chunks)
        count = _exact_count(self.client, contract.collection_name)
        if count == contract.build_chunk_count:
            if count == 0 or _chunk_ids_match(
                self.client,
                contract.collection_name,
                contract.build_chunk_ids_sha256,
            ):
                return IndexWriteSummary(
                    collection_name=contract.collection_name,
                    status="noop",
                    inserted_count=0,
                    row_count=count,
                    contract_sha256=contract.contract_sha256,
                )
            raise CollectionContractMismatch("loaded chunk IDs do not match the immutable contract")
        if count != 0:
            raise CollectionContractMismatch("immutable Milvus build was partially written")

        vectors = self.embedding_manager.embed_documents([chunk.content for chunk in ordered])
        rows = [materialize_chunk(chunk, vectors[index]) for index, chunk in enumerate(ordered)]
        result = self.client.insert(
            collection_name=contract.collection_name,
            data=rows,
        )
        inserted_count = int(getattr(result, "insert_count", len(rows)))
        if inserted_count != len(rows):
            raise CollectionContractMismatch("Milvus insert count does not match the build")
        self.client.flush(contract.collection_name)
        self._mark_contract_complete(contract)
        self._load_collection(contract.collection_name)
        if not _chunk_ids_match(
            self.client,
            contract.collection_name,
            contract.build_chunk_ids_sha256,
        ):
            raise CollectionContractMismatch("flushed chunk IDs do not match the immutable contract")
        return IndexWriteSummary(
            collection_name=contract.collection_name,
            status="inserted",
            inserted_count=inserted_count,
            row_count=len(rows),
            contract_sha256=contract.contract_sha256,
        )

    def validate(
        self,
        contract: CollectionContract,
        *,
        require_complete: bool = True,
    ) -> CollectionStats:
        if self._contract != contract:
            raise CollectionContractMismatch("validated contract does not match the opened store")
        if not self.client.has_collection(contract.collection_name):
            raise CollectionContractMismatch("Milvus collection is absent")
        self._validate_collection_schema(contract)
        self._validate_collection_indexes(contract)
        loaded = self._require_loaded(contract.collection_name)
        row = _contract_row(self.client, contract.collection_name)
        if row is None:
            raise CollectionContractMismatch("collection contract row is absent")
        _validated_persisted_contract(row, contract)
        if row.get("state") not in {"created", "complete"}:
            raise CollectionContractMismatch("collection contract state is invalid")
        row_count = _exact_count(self.client, contract.collection_name)
        if require_complete:
            if row["state"] != "complete":
                raise CollectionContractMismatch("Milvus build is not complete")
            if row_count != contract.build_chunk_count:
                raise CollectionContractMismatch("Milvus exact row count does not match the contract")
            exact_chunk_ids = _chunk_ids_match(
                self.client,
                contract.collection_name,
                contract.build_chunk_ids_sha256,
            )
            if not exact_chunk_ids:
                raise CollectionContractMismatch("Milvus chunk IDs do not match the contract")
        else:
            if row_count != 0:
                raise CollectionContractMismatch("created Milvus collection is not empty")
            exact_chunk_ids = row_count == contract.build_chunk_count
        return CollectionStats(
            collection_name=contract.collection_name,
            build_id=contract.build_id,
            row_count=row_count,
            contract_state=row["state"],
            contract_sha256=row["contract_sha256"],
            field_names=tuple(name for name, _, _, _ in _main_field_contracts(1024)),
            hnsw_m=contract.hnsw_m,
            hnsw_ef_construction=contract.hnsw_ef_construction,
            loaded=loaded,
            exact_chunk_ids=exact_chunk_ids,
        )

    def search(
        self,
        vector: np.ndarray,
        *,
        top_k: int,
        filter_: RetrievalFilter | None,
    ) -> tuple[DenseHit, ...]:
        contract = self._require_contract()
        if type(top_k) is not int or not 1 <= top_k <= 512:
            raise ValueError("top_k must be an integer from 1 through 512")
        validated_vector = validate_query_vector(vector, dimension=contract.embedding_dimension)
        compiled = (
            compile_filter_binding(filter_)
            if filter_ is not None
            else None
        )
        raw = self.client.search(
            collection_name=contract.collection_name,
            data=[validated_vector.tolist()],
            limit=top_k,
            filter=compiled.expression if compiled else "",
            filter_params=compiled.parameters if compiled else None,
            output_fields=[
                "chunk_id",
                "text",
                "document_id",
                "metadata_json",
            ],
            search_params={
                "metric_type": contract.metric_type,
                "params": {"ef": contract.default_ef_search},
            },
            consistency_level="Strong",
        )
        raw_hits = raw[0]
        hits: list[DenseHit] = []
        seen_chunk_ids: set[str] = set()
        for rank, raw_hit in enumerate(raw_hits, start=1):
            entity = dict(raw_hit.get("entity", {}))
            chunk_id = entity.get("chunk_id")
            metadata_json = entity.get("metadata_json")
            text = entity.get("text")
            if type(chunk_id) is not str or type(metadata_json) is not str or type(text) is not str:
                raise CollectionContractMismatch("Milvus search response is missing required fields")
            if chunk_id in seen_chunk_ids:
                raise CollectionContractMismatch("Milvus search returned duplicate chunk IDs")
            seen_chunk_ids.add(chunk_id)
            try:
                record = ChunkRecord.model_validate_json(
                    json.dumps(
                        {"content": text, "metadata": json.loads(metadata_json)},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            except Exception as error:
                raise CollectionContractMismatch("Milvus metadata JSON is invalid") from error
            if record.metadata.chunk_id != chunk_id or record.content != text:
                raise CollectionContractMismatch("Milvus search payload does not match stored metadata")
            hits.append(
                DenseHit(
                    chunk_id=chunk_id,
                    record=record,
                    score=float(raw_hit["distance"]),
                    rank=rank,
                    build_id=contract.build_id,
                    filter_expression=compiled.expression if compiled else "",
                    filter_expression_version=(
                        compiled.version if compiled else contract.filter_expression_version
                    ),
                )
            )
        return tuple(hits)

    def drop_owned_collection(
        self,
        contract: CollectionContract,
        *,
        allow_incomplete_created_here: bool = False,
    ) -> None:
        if self._contract != contract:
            raise CollectionContractMismatch("refusing to drop a collection not owned by this store")
        row = _contract_row(self.client, contract.collection_name)
        if row is not None:
            _validated_persisted_contract(row, contract)
            state = row.get("state")
            if state not in {"created", "complete"}:
                raise CollectionContractMismatch("Milvus contract state prevents cleanup")
        if self.client.has_collection(contract.collection_name):
            self.client.drop_collection(contract.collection_name)
        self.client.delete(
            collection_name=CONTRACT_COLLECTION_NAME,
            ids=[contract.collection_name],
        )
        self.client.flush(CONTRACT_COLLECTION_NAME)

    def _bind(self, contract: CollectionContract, build: BuildManifest) -> None:
        if contract.build_id != build.build_id:
            raise CollectionContractMismatch("collection contract belongs to another build")
        self._contract = contract
        self._build = build

    def _require_contract(self) -> CollectionContract:
        if self._contract is None:
            raise RuntimeError("open a Milvus collection before performing this operation")
        return self._contract

    def _create_contract_collection(self) -> None:
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="collection_name",
            datatype=_DATATYPES["VARCHAR"],
            is_primary=True,
            max_length=128,
        )
        schema.add_field(field_name="owner", datatype=_DATATYPES["VARCHAR"], max_length=64)
        schema.add_field(field_name="schema_version", datatype=_DATATYPES["VARCHAR"], max_length=64)
        schema.add_field(field_name="contract_sha256", datatype=_DATATYPES["VARCHAR"], max_length=64)
        schema.add_field(field_name="contract_json", datatype=_DATATYPES["VARCHAR"], max_length=8192)
        schema.add_field(field_name="state", datatype=_DATATYPES["VARCHAR"], max_length=32)
        schema.add_field(field_name="contract_vector", datatype=_DATATYPES["FLOAT_VECTOR"], dim=2)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="contract_vector",
            index_name=CONTRACT_VECTOR_INDEX_NAME,
            index_type="FLAT",
            metric_type="COSINE",
        )
        self.client.create_collection(
            collection_name=CONTRACT_COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
            description=f"{COLLECTION_OWNER}:{CONTRACT_COLLECTION_SCHEMA_VERSION}",
        )
        self._validate_contract_collection_schema()

    def _validate_contract_collection_schema(self) -> None:
        self._validate_schema(
            self.client.describe_collection(CONTRACT_COLLECTION_NAME),
            _contract_field_contracts(),
            CONTRACT_COLLECTION_NAME,
        )

    def _create_collection(
        self,
        contract: CollectionContract,
        build: BuildManifest,
    ) -> None:
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        for field_name, field_type, params, primary in _main_field_contracts(
            contract.embedding_dimension
        ):
            kwargs: dict[str, object] = dict(params)
            if primary:
                kwargs["is_primary"] = True
            schema.add_field(
                field_name=field_name,
                datatype=_DATATYPES[field_type],
                **kwargs,
            )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=contract.vector_field,
            index_name=VECTOR_INDEX_NAME,
            index_type=contract.index_type,
            metric_type=contract.metric_type,
            params={
                "M": contract.hnsw_m,
                "efConstruction": contract.hnsw_ef_construction,
            },
        )
        self.client.create_collection(
            collection_name=contract.collection_name,
            schema=schema,
            index_params=index_params,
            description=_main_description(contract.build_id),
        )
        if not self.client.has_collection(contract.collection_name):
            raise CollectionContractMismatch("Milvus collection creation did not persist")

    def _insert_contract_row(
        self,
        contract: CollectionContract,
        *,
        state: Literal["created", "complete"],
    ) -> None:
        result = self.client.insert(
            collection_name=CONTRACT_COLLECTION_NAME,
            data=[
                {
                    "collection_name": contract.collection_name,
                    "owner": contract.owner,
                    "schema_version": contract.schema_version,
                    "contract_sha256": contract.contract_sha256,
                    "contract_json": contract.canonical_json(),
                    "state": state,
                    "contract_vector": [1.0, 0.0],
                }
            ],
        )
        if int(getattr(result, "insert_count", 1)) != 1:
            raise CollectionContractMismatch("Milvus contract registry insert failed")
        self.client.flush(CONTRACT_COLLECTION_NAME)

    def _mark_contract_complete(self, contract: CollectionContract) -> None:
        self.client.delete(
            collection_name=CONTRACT_COLLECTION_NAME,
            ids=[contract.collection_name],
        )
        self.client.flush(CONTRACT_COLLECTION_NAME)
        self._insert_contract_row(contract, state="complete")

    def _validate_collection_schema(self, contract: CollectionContract) -> None:
        self._validate_schema(
            self.client.describe_collection(contract.collection_name),
            _main_field_contracts(contract.embedding_dimension),
            contract.collection_name,
        )

    def _validate_schema(
        self,
        description: dict[str, Any],
        expected: tuple[tuple[str, str, dict[str, int], bool], ...],
        collection_name: str,
    ) -> None:
        normalized = _normalized_fields(description)
        if normalized != expected:
            raise CollectionContractMismatch(
                f"Milvus schema mismatch for {collection_name}: {normalized} != {expected}"
            )

    def _validate_collection_indexes(self, contract: CollectionContract) -> None:
        indexes = self.client.list_indexes(contract.collection_name)
        if indexes != [VECTOR_INDEX_NAME]:
            raise CollectionContractMismatch("Milvus vector index contract mismatch")
        index = self.client.describe_index(contract.collection_name, VECTOR_INDEX_NAME)
        if (
            index.get("index_type") != contract.index_type
            or index.get("metric_type") != contract.metric_type
            or index.get("field_name") != contract.vector_field
        ):
            raise CollectionContractMismatch("Milvus vector index settings mismatch")

    def _load_collection(self, collection_name: str) -> None:
        self.client.load_collection(collection_name)
        self._require_loaded(collection_name)

    def _require_loaded(self, collection_name: str) -> bool:
        state = self.client.get_load_state(collection_name)
        value = state.get("state") if isinstance(state, dict) else getattr(state, "state", state)
        loaded = getattr(value, "name", str(value)) == "Loaded"
        if not loaded:
            raise CollectionContractMismatch(f"Milvus collection is not loaded: {collection_name}")
        return getattr(value, "name", str(value)) == "Loaded"
