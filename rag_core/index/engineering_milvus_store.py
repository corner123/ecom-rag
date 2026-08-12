"""Milvus dense-vector backend for immutable engineering-index builds.

This adapter deliberately uses ``MilvusClient`` instead of the legacy global
``connections``/``Collection`` API.  A collection belongs to one immutable
engineering build and one partition; local FAISS artifacts remain the durable
portable snapshot and BM25 source.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from langchain_core.documents import Document

from .embeddings import EmbeddingManager


class MilvusBackendError(RuntimeError):
    """Base error for the engineering Milvus adapter."""


class MilvusAvailabilityError(MilvusBackendError):
    """The configured Milvus backend cannot currently serve requests."""


class MilvusIntegrityError(MilvusBackendError):
    """The remote artifact exists but does not match its catalog contract."""


def resolve_milvus_owner_namespace(
    value: str | None = None,
    *,
    required: bool = False,
) -> str | None:
    """Return a non-secret deployment owner recorded in remote artifacts."""

    selected = str(
        value
        or os.getenv("ENGINEERING_MILVUS_NAMESPACE", "")
    ).strip()
    if not selected:
        if required:
            raise ValueError(
                "ENGINEERING_MILVUS_NAMESPACE is required for Milvus builds; "
                "choose a stable deployment-unique value"
            )
        return None
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,63}", selected):
        raise ValueError(
            "ENGINEERING_MILVUS_NAMESPACE must be 3-64 characters using "
            "letters, digits, dot, underscore or hyphen"
        )
    return selected


@dataclass(frozen=True, slots=True)
class MilvusConnectionSettings:
    """Connection settings with a secret-safe public representation."""

    uri: str = "http://127.0.0.1:19530"
    token: str | None = None
    database: str = "default"
    timeout_seconds: float = 5.0

    @classmethod
    def from_env(cls) -> "MilvusConnectionSettings":
        uri = os.getenv("ENGINEERING_MILVUS_URI", "").strip()
        if not uri:
            host = os.getenv("MILVUS_HOST", "127.0.0.1").strip()
            port = os.getenv("MILVUS_PORT", "19530").strip()
            uri = f"http://{host}:{port}"
        token = os.getenv("ENGINEERING_MILVUS_TOKEN", "").strip() or None
        if token is None:
            user = os.getenv("MILVUS_USER", "").strip()
            password = os.getenv("MILVUS_PASSWORD", "").strip()
            if user or password:
                token = f"{user}:{password}"
        database = (
            os.getenv("ENGINEERING_MILVUS_DATABASE", "").strip()
            or os.getenv("MILVUS_DATABASE", "").strip()
            or "default"
        )
        raw_timeout = os.getenv("ENGINEERING_MILVUS_TIMEOUT_SECONDS", "5").strip()
        try:
            timeout = float(raw_timeout)
        except ValueError as exc:
            raise ValueError(
                "ENGINEERING_MILVUS_TIMEOUT_SECONDS must be numeric"
            ) from exc
        if timeout <= 0:
            raise ValueError("ENGINEERING_MILVUS_TIMEOUT_SECONDS must be positive")
        return cls(uri=uri, token=token, database=database, timeout_seconds=timeout)

    def public_dict(self) -> dict[str, Any]:
        return {
            "database": self.database,
            "timeout_seconds": self.timeout_seconds,
            "authenticated": bool(self.token),
        }


MilvusClientFactory = Callable[[MilvusConnectionSettings], Any]


def create_milvus_client(settings: MilvusConnectionSettings) -> Any:
    """Create a real Milvus client lazily so FAISS-only installs still load."""

    try:
        from pymilvus import MilvusClient
    except (ImportError, OSError) as exc:
        raise MilvusBackendError(
            "pymilvus is unavailable; install the declared Milvus dependency"
        ) from exc
    kwargs: dict[str, Any] = {
        "uri": settings.uri,
        "db_name": settings.database,
        "timeout": settings.timeout_seconds,
    }
    if settings.token:
        kwargs["token"] = settings.token
    try:
        return MilvusClient(**kwargs)
    except Exception as exc:  # SDK transports expose several vendor exceptions.
        raise _classify_sdk_error(
            "create client for the configured Milvus service", exc
        ) from exc


def build_collection_name(
    build_id: str,
    partition: str,
    build_token: str,
    *,
    prefix: str | None = None,
) -> str:
    """Return a Milvus-safe, build-scoped immutable collection name."""

    raw_prefix = prefix or os.getenv(
        "ENGINEERING_MILVUS_COLLECTION_PREFIX", "engineering_rag"
    )
    safe_prefix = re.sub(r"[^A-Za-z0-9_]", "_", raw_prefix).strip("_") or "engineering_rag"
    safe_partition = re.sub(r"[^A-Za-z0-9_]", "_", partition).strip("_") or "partition"
    build_digest = hashlib.sha256(build_id.encode("utf-8")).hexdigest()[:12]
    token = re.sub(r"[^A-Za-z0-9]", "", build_token)[:10] or "build"
    value = f"{safe_prefix}_{build_digest}_{token}_{safe_partition}"
    if value[0].isdigit():
        value = f"c_{value}"
    return value[:255]


class EngineeringMilvusVectorStore:
    """Dense store backed by one immutable Milvus partition collection."""

    TEXT_MAX_LENGTH = 65_535
    METADATA_MAX_LENGTH = 65_535

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        collection_name: str,
        *,
        expected_dimension: int,
        expected_count: int,
        settings: MilvusConnectionSettings | None = None,
        client_factory: MilvusClientFactory | None = None,
        client: Any | None = None,
        data_types: Mapping[str, Any] | None = None,
        build_id: str = "",
        embedding_model: str = "",
        owner_namespace: str | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,254}", collection_name):
            raise ValueError(f"invalid Milvus collection name: {collection_name}")
        if expected_dimension <= 0 or expected_count < 0:
            raise ValueError("invalid expected Milvus artifact dimensions")
        self.embedding_manager = embedding_manager
        self.collection_name = collection_name
        self.expected_dimension = int(expected_dimension)
        self.expected_count = int(expected_count)
        self.build_id = str(build_id)
        self.embedding_model = str(embedding_model)
        resolved_owner = resolve_milvus_owner_namespace(
            owner_namespace,
            required=True,
        )
        assert resolved_owner is not None
        self.owner_namespace = resolved_owner
        self.settings = settings or MilvusConnectionSettings.from_env()
        self._client_factory = client_factory or create_milvus_client
        self._client = client
        self._provided_data_types = dict(data_types or {})

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                self._client = self._client_factory(self.settings)
            except MilvusBackendError:
                raise
            except Exception as exc:
                raise _classify_sdk_error(
                    "create the configured Milvus client", exc
                ) from exc
        return self._client

    def replace_documents(
        self,
        docs: list[Document],
        batch_size: int = 256,
    ) -> dict[str, int]:
        """Create and populate a new immutable collection."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        records = self._deduplicate(docs)
        if len(records) != self.expected_count:
            raise MilvusIntegrityError(
                "Milvus document count does not match the catalog build contract"
            )
        texts = [record["text"] for record in records]
        vectors = self._embed(texts, batch_size=batch_size)
        if vectors.shape[1] != self.expected_dimension:
            raise MilvusIntegrityError(
                f"embedding dimension {vectors.shape[1]} does not match "
                f"catalog dimension {self.expected_dimension}"
            )
        client = self.client
        creation_attempted = False
        try:
            if client.has_collection(collection_name=self.collection_name):
                raise MilvusIntegrityError(
                    f"immutable Milvus collection already exists: {self.collection_name}"
                )
            schema = client.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
                description=self._artifact_description(),
            )
            types = self._data_types()
            schema.add_field(
                field_name="chunk_id",
                datatype=types["VARCHAR"],
                is_primary=True,
                max_length=256,
            )
            schema.add_field(
                field_name="dense_vector",
                datatype=types["FLOAT_VECTOR"],
                dim=self.expected_dimension,
            )
            schema.add_field(
                field_name="text",
                datatype=types["VARCHAR"],
                max_length=self.TEXT_MAX_LENGTH,
            )
            schema.add_field(
                field_name="metadata_json",
                datatype=types["VARCHAR"],
                max_length=self.METADATA_MAX_LENGTH,
            )
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200},
            )
            # Mark the attempt before crossing the network boundary. The
            # server may commit creation even if the client loses the response.
            creation_attempted = True
            client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
            for start in range(0, len(records), batch_size):
                batch = []
                for record, vector in zip(
                    records[start : start + batch_size],
                    vectors[start : start + batch_size],
                ):
                    batch.append(
                        {
                            "chunk_id": record["chunk_id"],
                            "dense_vector": vector.tolist(),
                            "text": record["text"],
                            "metadata_json": json.dumps(
                                record["metadata"],
                                ensure_ascii=False,
                                separators=(",", ":"),
                                sort_keys=True,
                            ),
                        }
                    )
                if batch:
                    client.insert(collection_name=self.collection_name, data=batch)
            flush = getattr(client, "flush", None)
            if callable(flush):
                flush(collection_name=self.collection_name)
            load = getattr(client, "load_collection", None)
            if callable(load):
                load(collection_name=self.collection_name)
            # Validation belongs to the same transaction boundary as creation
            # and insertion.  If Milvus accepted the writes but exposes an
            # unexpected row count, schema or artifact description, this
            # unpublished build must not leave an orphan collection behind.
            self.validate()
        except MilvusBackendError as exc:
            if creation_attempted:
                self._discard_unpublished_collection(exc)
            raise
        except Exception as exc:
            classified = _classify_sdk_error(
                f"create Milvus collection {self.collection_name}", exc
            )
            if creation_attempted:
                self._discard_unpublished_collection(classified)
            raise classified from exc
        return {"indexed": len(records), "duplicates_removed": len(docs) - len(records)}

    def _discard_unpublished_collection(self, original: BaseException) -> None:
        """Best-effort rollback that cannot replace the transaction failure."""

        try:
            self.drop_collection(ignore_missing=True)
        except Exception as cleanup_error:
            original.add_note(
                "failed to remove unpublished Milvus collection "
                f"{self.collection_name}: {type(cleanup_error).__name__}: "
                f"{cleanup_error}"
            )

    def validate(self) -> dict[str, Any]:
        """Validate existence, row count and vector dimension against catalog."""

        client = self.client
        try:
            if not client.has_collection(collection_name=self.collection_name):
                raise MilvusIntegrityError(
                    f"Milvus collection is missing: {self.collection_name}"
                )
            stats = client.get_collection_stats(collection_name=self.collection_name)
            description = client.describe_collection(collection_name=self.collection_name)
        except MilvusBackendError:
            raise
        except Exception as exc:
            raise _classify_sdk_error(
                f"inspect Milvus collection {self.collection_name}", exc
            ) from exc
        row_count = self._row_count(stats)
        if row_count != self.expected_count:
            raise MilvusIntegrityError(
                f"Milvus row count mismatch for {self.collection_name}: "
                f"catalog={self.expected_count}, remote={row_count}"
            )
        dimension = self._vector_dimension(description)
        if dimension != self.expected_dimension:
            raise MilvusIntegrityError(
                f"Milvus vector dimension mismatch for {self.collection_name}: "
                f"catalog={self.expected_dimension}, remote={dimension}"
            )
        artifact = self._artifact_from_description(description)
        expected_artifact = {
            "build_id": self.build_id,
            "embedding_model": self.embedding_model,
            "dimension": self.expected_dimension,
            "owner_namespace": self.owner_namespace,
        }
        if artifact != expected_artifact:
            raise MilvusIntegrityError(
                f"Milvus artifact metadata mismatch for {self.collection_name}"
            )
        return self.get_stats()

    def search_dense(
        self,
        query: str,
        top_k: int = 5,
        filter_expr: str = "",
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        vector = self._normalise(np.asarray([self.embedding_manager.embed_query(query)], dtype="float32"))[0]
        if vector.shape[0] != self.expected_dimension:
            raise MilvusIntegrityError(
                "query embedding dimension does not match the Milvus collection"
            )
        kwargs: dict[str, Any] = {
            "collection_name": self.collection_name,
            "data": [vector.tolist()],
            "anns_field": "dense_vector",
            "limit": int(top_k),
            "output_fields": ["chunk_id", "text", "metadata_json"],
            "search_params": {"metric_type": "COSINE", "params": {"ef": 128}},
        }
        if filter_expr:
            kwargs["filter"] = filter_expr
        try:
            response = self.client.search(**kwargs)
        except Exception as exc:
            raise _classify_sdk_error(
                f"search Milvus collection {self.collection_name}", exc
            ) from exc
        hits = response[0] if response else []
        formatted: list[dict[str, Any]] = []
        for hit in hits:
            entity = self._hit_value(hit, "entity", {}) or {}
            raw_metadata = entity.get("metadata_json", "{}")
            try:
                metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else dict(raw_metadata)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise MilvusIntegrityError("Milvus metadata_json is corrupt") from exc
            chunk_id = str(
                entity.get("chunk_id")
                or self._hit_value(hit, "id", "")
                or metadata.get("chunk_id", "")
            )
            metadata["chunk_id"] = chunk_id
            formatted.append(
                {
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "text": str(entity.get("text", "")),
                    "score": float(
                        self._hit_value(
                            hit,
                            "distance",
                            self._hit_value(hit, "score", 0.0),
                        )
                    ),
                    "metadata": metadata,
                }
            )
        return formatted

    def drop_collection(self, *, ignore_missing: bool = False) -> None:
        try:
            exists = self.client.has_collection(collection_name=self.collection_name)
        except MilvusBackendError:
            raise
        except Exception as exc:
            raise _classify_sdk_error(
                f"inspect Milvus collection {self.collection_name} before deletion",
                exc,
            ) from exc
        if not exists:
            if ignore_missing:
                return
            raise MilvusIntegrityError(
                f"Milvus collection is missing: {self.collection_name}"
            )
        try:
            self.client.drop_collection(collection_name=self.collection_name)
        except MilvusBackendError:
            raise
        except Exception as exc:
            raise _classify_sdk_error(
                f"drop Milvus collection {self.collection_name}", exc
            ) from exc

    def get_stats(self) -> dict[str, Any]:
        return {
            "store_type": "distributed_milvus",
            "collection_name": self.collection_name,
            "num_entities": self.expected_count,
            "dimension": self.expected_dimension,
            "metric_type": "COSINE",
            "connection": self.settings.public_dict(),
        }

    def _artifact_description(self) -> str:
        return "engineering-rag:" + json.dumps(
            {
                "build_id": self.build_id,
                "embedding_model": self.embedding_model,
                "dimension": self.expected_dimension,
                "owner_namespace": self.owner_namespace,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @staticmethod
    def _artifact_from_description(description: Any) -> dict[str, Any]:
        raw = description.get("description") if isinstance(description, Mapping) else None
        prefix = "engineering-rag:"
        if not isinstance(raw, str) or not raw.startswith(prefix):
            raise MilvusIntegrityError("Milvus collection has no engineering artifact metadata")
        try:
            payload = json.loads(raw[len(prefix) :])
        except json.JSONDecodeError as exc:
            raise MilvusIntegrityError("Milvus collection artifact metadata is corrupt") from exc
        if not isinstance(payload, dict):
            raise MilvusIntegrityError("Milvus collection artifact metadata is invalid")
        try:
            dimension = int(payload.get("dimension"))
        except (TypeError, ValueError) as exc:
            raise MilvusIntegrityError("Milvus artifact dimension is invalid") from exc
        return {
            "build_id": str(payload.get("build_id") or ""),
            "embedding_model": str(payload.get("embedding_model") or ""),
            "dimension": dimension,
            "owner_namespace": str(payload.get("owner_namespace") or ""),
        }

    def _data_types(self) -> dict[str, Any]:
        if self._provided_data_types:
            return self._provided_data_types
        client_types = getattr(self.client, "data_types", None)
        if isinstance(client_types, Mapping) and {
            "VARCHAR",
            "FLOAT_VECTOR",
        }.issubset(client_types):
            return dict(client_types)
        try:
            from pymilvus import DataType
        except (ImportError, OSError) as exc:
            raise MilvusAvailabilityError("pymilvus DataType is unavailable") from exc
        return {"VARCHAR": DataType.VARCHAR, "FLOAT_VECTOR": DataType.FLOAT_VECTOR}

    def _embed(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(self.embedding_manager.embed_documents(texts[start : start + batch_size]))
        array = np.asarray(vectors, dtype="float32")
        if array.ndim != 2:
            raise MilvusIntegrityError("embedding manager returned an invalid vector matrix")
        return self._normalise(array)

    @staticmethod
    def _normalise(array: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return array / norms

    @classmethod
    def _deduplicate(cls, docs: Iterable[Document]) -> list[dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for doc in docs:
            metadata = cls._json_safe(dict(doc.metadata))
            chunk_id = str(metadata.get("chunk_id") or cls._stable_chunk_id(doc))
            if len(chunk_id) > 256:
                chunk_id = hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()
            metadata["chunk_id"] = chunk_id
            text = str(doc.page_content)
            encoded_metadata = json.dumps(metadata, ensure_ascii=False)
            if len(text.encode("utf-8")) > cls.TEXT_MAX_LENGTH:
                raise MilvusIntegrityError("document text exceeds the Milvus VARCHAR limit")
            if len(encoded_metadata.encode("utf-8")) > cls.METADATA_MAX_LENGTH:
                raise MilvusIntegrityError("document metadata exceeds the Milvus VARCHAR limit")
            if chunk_id not in records:
                order.append(chunk_id)
            records[chunk_id] = {"chunk_id": chunk_id, "text": text, "metadata": metadata}
        return [records[chunk_id] for chunk_id in order]

    @staticmethod
    def _stable_chunk_id(doc: Document) -> str:
        payload = json.dumps(
            {"text": doc.page_content, "metadata": doc.metadata},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._json_safe(item) for item in value]
        return str(value)

    @staticmethod
    def _row_count(stats: Any) -> int:
        if isinstance(stats, Mapping):
            raw = stats.get("row_count", stats.get("num_entities"))
        else:
            raw = getattr(stats, "row_count", getattr(stats, "num_entities", None))
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise MilvusIntegrityError("Milvus stats omitted a valid row count") from exc

    @staticmethod
    def _vector_dimension(description: Any) -> int:
        fields = description.get("fields", []) if isinstance(description, Mapping) else []
        for field in fields:
            if not isinstance(field, Mapping) or field.get("name") != "dense_vector":
                continue
            params = field.get("params") or field.get("type_params") or {}
            raw = params.get("dim") if isinstance(params, Mapping) else None
            try:
                return int(raw)
            except (TypeError, ValueError):
                break
        raise MilvusIntegrityError("Milvus collection description omitted vector dimension")

    @staticmethod
    def _hit_value(hit: Any, name: str, default: Any) -> Any:
        if isinstance(hit, Mapping):
            return hit.get(name, default)
        return getattr(hit, name, default)


class AvailabilityFailoverDenseStore:
    """Sticky Milvus-to-FAISS failover for availability failures only."""

    def __init__(
        self,
        primary: EngineeringMilvusVectorStore,
        fallback: Any,
        *,
        on_fallback: Callable[[str], None] | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.on_fallback = on_fallback
        self._using_fallback = False

    def search_dense(self, query: str, top_k: int = 5, filter_expr: str = "") -> list[dict[str, Any]]:
        if self._using_fallback:
            return self.fallback.search_dense(query, top_k=top_k, filter_expr=filter_expr)
        try:
            return self.primary.search_dense(query, top_k=top_k, filter_expr=filter_expr)
        except MilvusAvailabilityError as exc:
            self._using_fallback = True
            if self.on_fallback is not None:
                self.on_fallback(str(exc))
            return self.fallback.search_dense(query, top_k=top_k, filter_expr=filter_expr)

    def get_stats(self) -> dict[str, Any]:
        if self._using_fallback:
            return self.fallback.get_stats()
        return self.primary.get_stats()


def cleanup_stale_engineering_collections(
    active_collections: Iterable[str],
    *,
    execute: bool = False,
    settings: MilvusConnectionSettings | None = None,
    client_factory: MilvusClientFactory | None = None,
    prefix: str | None = None,
    owner_namespace: str | None = None,
    expected_candidates: Iterable[str] | None = None,
) -> dict[str, Any]:
    """List or explicitly delete unreachable build-scoped collections.

    Cleanup is intentionally not automatic at publication time: an older
    process may still be serving the previous catalog. Operators first run a
    dry run and execute deletion only after that reader-drain boundary.
    """

    owner = resolve_milvus_owner_namespace(owner_namespace, required=True)
    assert owner is not None
    resolved_settings = settings or MilvusConnectionSettings.from_env()
    factory = client_factory or create_milvus_client
    try:
        client = factory(resolved_settings)
        listed = client.list_collections()
    except Exception as exc:
        raise _classify_sdk_error("list engineering Milvus collections", exc) from exc
    safe_prefix = re.sub(
        r"[^A-Za-z0-9_]",
        "_",
        prefix or os.getenv("ENGINEERING_MILVUS_COLLECTION_PREFIX", "engineering_rag"),
    ).strip("_") or "engineering_rag"
    pattern = re.compile(
        rf"^{re.escape(safe_prefix)}_[0-9a-f]{{12}}_[A-Za-z0-9]{{1,10}}_[A-Za-z0-9_]+$"
    )
    active = {str(name) for name in active_collections}
    candidates: list[str] = []
    for raw_name in listed:
        name = str(raw_name)
        if not pattern.fullmatch(name) or name in active:
            continue
        try:
            description = client.describe_collection(collection_name=name)
        except MilvusBackendError:
            raise
        except Exception as exc:
            raise _classify_sdk_error(
                f"inspect engineering Milvus collection {name}", exc
            ) from exc
        try:
            artifact = EngineeringMilvusVectorStore._artifact_from_description(
                description
            )
        except (MilvusIntegrityError, KeyError, TypeError, ValueError):
            # A matching name is not ownership proof. Unknown or malformed
            # collections are never deletion candidates.
            continue
        if artifact.get("owner_namespace") == owner:
            candidates.append(name)
    candidates.sort()
    expected = (
        sorted({str(name) for name in expected_candidates})
        if expected_candidates is not None
        else None
    )
    if execute and expected is not None and candidates != expected:
        raise RuntimeError(
            "Milvus cleanup candidates changed after dry-run; inspect and retry"
        )
    deleted: list[str] = []
    if execute:
        for name in candidates:
            try:
                client.drop_collection(collection_name=name)
            except Exception as exc:
                raise _classify_sdk_error(
                    f"drop stale engineering collection {name}", exc
                ) from exc
            deleted.append(name)
    return {
        "dry_run": not execute,
        "active_collections": sorted(active),
        "owner_namespace": owner,
        "candidates": candidates,
        "deleted": deleted,
    }


_AVAILABILITY_MARKERS = (
    "connection refused",
    "connection reset",
    "connection aborted",
    "failed to connect",
    "network is unreachable",
    "no route to host",
    "service unavailable",
    "statuscode.unavailable",
    "status_code: unavailable",
    "deadline exceeded",
    "statuscode.deadline_exceeded",
    "timed out",
    "timeout",
    "transport is closing",
)
_FAIL_CLOSED_MARKERS = (
    "unauthenticated",
    "authentication",
    "unauthorized",
    "permission denied",
    "forbidden",
    "invalid argument",
    "invalid token",
    "schema",
    "dimension",
    "database not found",
    "illegal",
)


def _classify_sdk_error(action: str, exc: Exception) -> MilvusBackendError:
    """Classify only clear transport outages as safe automatic failover.

    Authentication, authorization, schema and other configuration failures are
    deliberately fail-closed. Unknown SDK failures are also not treated as
    availability faults; silently hiding a programming error behind FAISS would
    make health reporting misleading.
    """

    if isinstance(exc, MilvusBackendError):
        return exc
    details = f"{exc.__class__.__module__}.{exc.__class__.__name__}: {exc}".casefold()
    message = f"Milvus failed to {action}"
    if any(marker in details for marker in _FAIL_CLOSED_MARKERS):
        return MilvusIntegrityError(message)
    if any(marker in details for marker in _AVAILABILITY_MARKERS):
        return MilvusAvailabilityError(message)
    return MilvusBackendError(message)
