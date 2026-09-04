"""Publish and reopen immutable, jointly verified dense/sparse index bundles."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from trade_agent.data.manifest import BuildManifest, canonical_json
from trade_agent.index.milvus_store import (
    CollectionContract, MATERIALIZED_FIELDS, chunk_ids_sha256,
    collection_name_for_build_id, materialize_chunk,
)
from trade_agent.retrieval.bm25 import BM25Index


class BundleDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["trade-index-bundle-v1"] = "trade-index-bundle-v1"
    build_id: str
    build_fingerprint: str
    chunk_count: int
    chunk_ids_sha256: str
    manifest_sha256: str
    bm25_sha256: str
    dense_payload_sha256: str
    collection_contract: CollectionContract


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_sparse_build(build: BuildManifest, bm25: BM25Index) -> None:
    expected = {snapshot.chunk_id: snapshot.restore() for snapshot in build.chunks}
    actual = {record.metadata.chunk_id: record for record in bm25.records}
    if bm25.build_id != build.build_id or bm25.chunk_count != len(expected) or actual != expected:
        raise ValueError("BM25 records do not exactly match the frozen build")
    bm25.validate_records(tuple(snapshot.restore() for snapshot in build.chunks))


def _validate_dense(build: BuildManifest, milvus: Any, contract: CollectionContract) -> str:
    if (contract.build_id != build.build_id or contract.build_fingerprint != build.fingerprint
        or contract.build_chunk_count != len(build.chunks)
        or contract.build_chunk_ids_sha256 != chunk_ids_sha256(build.chunk_ids)
        or contract.metadata_schema_version != build.metadata_schema_version):
        raise ValueError("dense collection contract does not match frozen build")
    stats = milvus.validate(contract)
    if (stats.row_count != len(build.chunks) or not stats.exact_chunk_ids
        or not stats.loaded or stats.contract_state != "complete"):
        raise ValueError("dense collection is not a complete exact build")
    expected_records = {s.chunk_id: s.restore() for s in build.chunks}
    seen = set()
    row_hashes = {}
    iterator = milvus.client.query_iterator(collection_name=contract.collection_name,
        filter="", output_fields=list(MATERIALIZED_FIELDS), batch_size=256,
        consistency_level="Strong")
    try:
        while rows := iterator.next():
            for row in rows:
                chunk_id = row.get("chunk_id")
                if chunk_id not in expected_records or chunk_id in seen:
                    raise ValueError("dense payload IDs do not match frozen build")
                vector = np.asarray(row.get("dense_vector"), dtype=np.float32)
                expected = materialize_chunk(expected_records[chunk_id], vector)
                # Milvus FLOAT stores a float32 scalar; metadata_json retains
                # the original exact Python metadata representation.
                expected["source_weight"] = float(np.float32(expected["source_weight"]))
                if any(row.get(field) != value for field, value in expected.items()
                       if field != "dense_vector"):
                    raise ValueError("dense stored payload does not match frozen build")
                seen.add(chunk_id)
                row_hashes[chunk_id] = hashlib.sha256(canonical_json(expected).encode("utf-8")).hexdigest()
    finally:
        iterator.close()
    if seen != set(expected_records):
        raise ValueError("dense payload IDs do not match frozen build")
    return hashlib.sha256(canonical_json(row_hashes).encode("utf-8")).hexdigest()


def _live_embedding(manager: Any) -> Any:
    # A real observed vector is required before the manager exposes provenance.
    manager.embed_query("trade index contract verification")
    return manager.contract


@dataclass(frozen=True)
class TradeIndexBundle:
    descriptor_path: Path
    descriptor: BundleDescriptor
    build: BuildManifest
    bm25: BM25Index

    @property
    def bm25_path(self) -> Path:
        return self.descriptor_path.parent / "bm25.json"

    @classmethod
    def load(cls, path: str | Path, *, milvus: Any, embedding_manager: Any) -> "TradeIndexBundle":
        path = Path(path)
        descriptor = BundleDescriptor.model_validate_json(path.read_text(encoding="utf-8"))
        manifest_path = path.parent / "manifest.json"
        bm25_path = path.parent / "bm25.json"
        if _sha(manifest_path) != descriptor.manifest_sha256 or _sha(bm25_path) != descriptor.bm25_sha256:
            raise ValueError("bundle artifact checksum mismatch")
        build = BuildManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if (descriptor.build_id != build.build_id or descriptor.build_fingerprint != build.fingerprint
            or descriptor.chunk_count != len(build.chunks)
            or descriptor.chunk_ids_sha256 != chunk_ids_sha256(build.chunk_ids)):
            raise ValueError("bundle descriptor does not match frozen build")
        bm25 = BM25Index.load(bm25_path)
        validate_sparse_build(build, bm25)
        contract = milvus.open(build, _live_embedding(embedding_manager))
        if contract != descriptor.collection_contract:
            raise ValueError("bundle collection contract mismatch")
        if _validate_dense(build, milvus, contract) != descriptor.dense_payload_sha256:
            raise ValueError("dense payload checksum mismatch")
        return cls(path, descriptor, build, bm25)


class TradeIndexBuilder:
    """Own a new build directory; publish only after both indexes validate."""

    def __init__(self, *, output_dir: str | Path, milvus: Any, embedding_manager: Any):
        self.output_dir = Path(output_dir)
        self.milvus = milvus
        self.embedding_manager = embedding_manager

    def build(self, manifest: BuildManifest) -> TradeIndexBundle:
        # Re-parse to reject objects produced through unchecked model_construct.
        build = BuildManifest.model_validate_json(manifest.model_dump_json())
        if not build.chunks or not build.metadata_complete:
            raise ValueError("index bundles require nonempty metadata-complete builds")
        chunks = tuple(snapshot.restore() for snapshot in build.chunks)
        bm25 = BM25Index.build(chunks, build_id=build.build_id)
        validate_sparse_build(build, bm25)
        directory = self.output_dir / build.build_id
        if directory.exists():
            existing = TradeIndexBundle.load(directory / "bundle.json", milvus=self.milvus,
                                             embedding_manager=self.embedding_manager)
            if existing.build != build:
                raise ValueError("existing bundle does not match requested manifest")
            return existing
        directory.mkdir(parents=True, exist_ok=False)
        contract = None
        created_here = False
        try:
            embedding = _live_embedding(self.embedding_manager)
            if self.milvus.client.has_collection(collection_name_for_build_id(build.build_id)):
                contract = self.milvus.open(build, embedding)
            else:
                contract = self.milvus.create(build, embedding)
                created_here = True
                # BgeEmbeddingManager splits documents into its configured batches.
                self.milvus.replace_chunks(chunks)
            dense_sha256 = _validate_dense(build, self.milvus, contract)
            manifest_path = directory / "manifest.json"
            with manifest_path.open("x", encoding="utf-8") as handle:
                handle.write(build.model_dump_json())
                handle.flush()
                os.fsync(handle.fileno())
            bm25_path = directory / "bm25.json"
            bm25.save(bm25_path)
            validate_sparse_build(build, BM25Index.load(bm25_path))
            descriptor = BundleDescriptor(
                build_id=build.build_id, build_fingerprint=build.fingerprint,
                chunk_count=len(chunks), chunk_ids_sha256=chunk_ids_sha256(build.chunk_ids),
                manifest_sha256=_sha(manifest_path), bm25_sha256=_sha(bm25_path),
                collection_contract=contract, dense_payload_sha256=dense_sha256,
            )
            temporary = directory / "bundle.json.tmp"
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(canonical_json(descriptor.model_dump(mode="json")))
                handle.flush()
                os.fsync(handle.fileno())
            destination = directory / "bundle.json"
            os.replace(temporary, destination)
            return TradeIndexBundle(destination, descriptor, build, bm25)
        except BaseException:
            # Only clean a collection whose successful create proved ownership.
            try:
                if created_here and contract is not None:
                    self.milvus.drop_owned_collection(contract, allow_incomplete_created_here=True)
            finally:
                shutil.rmtree(directory)
            raise
