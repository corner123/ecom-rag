"""Read-only, fail-closed verification of the trade-agent foundation.

The command intentionally uses the query database role.  Migration and root
credentials are never read by this module, included in its return value, or
passed to a service client.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

from sqlalchemy import create_engine, text
from sqlalchemy.dialects import mysql

from trade_agent.config.settings import Settings
from trade_agent.data.demo_generator import generate_demo_corpus
from trade_agent.data.manifest import BuildManifest, canonical_hash, canonical_json
from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog
from trade_agent.db.models import Base
from trade_agent.db.seed import generate_trade_seed
from trade_agent.db.session import database_url_from_environment

SEED = 20260830
EXPECTED_TABLES = (
    "countries", "companies", "hs_codes", "products", "data_sources",
    "company_products", "trade_records",
)
EXPECTED_COUNTS = {
    "countries": 8, "companies": 60, "hs_codes": 12, "products": 30,
    "data_sources": 2, "company_products": 60, "trade_records": 825,
}
EXPECTED_MONTHS = 18
EXPECTED_SOURCE_TYPES = {
    "official_website", "b2b", "industry_news", "social", "regulator",
    "customs_profile",
}
EXPECTED_SCANNED_PATH = "pdf/scanned-regulator-notice.pdf"
EXPECTED_SOURCE_COUNTS = {"b2b": 1, "customs_profile": 54, "industry_news": 3, "official_website": 13, "regulator": 2, "social": 1}
EXPECTED_FILE_COUNTS = {"generated_profile": 54, "html": 13, "json": 2, "jsonl": 1, "markdown": 1, "pdf": 3}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resource_path(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("resource path must be relative")
    return _repo_root() / path


class FoundationSmokeFailure(RuntimeError):
    """Internal stage wrapper whose public rendering is intentionally redacted."""

    def __init__(self, stage: str, cause: BaseException):
        super().__init__(type(cause).__name__)
        self.stage = stage
        self.exception_class = type(cause).__name__


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise argparse.ArgumentError(None, "invalid command arguments")


def failure_payload(stage: str, cause: BaseException | str) -> dict[str, str]:
    """Return the only details allowed on the smoke command's failure boundary."""
    exception_class = cause if isinstance(cause, str) else type(cause).__name__
    return {"error": exception_class, "stage": stage, "status": "error"}


def validate_foundation_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Validate the minimum machine-readable contract without truthy shortcuts."""
    if not isinstance(summary, dict):
        raise ValueError("summary must be an object")
    required = {"status", "synthetic_only", "mysql_tables", "trade_records", "months",
                "corpus", "metadata_required_completeness", "quarantine_unexpected"}
    missing = sorted(required - set(summary))
    if missing:
        raise ValueError(f"summary missing {','.join(missing)}")
    if summary["status"] != "ok" or summary["synthetic_only"] is not True:
        raise ValueError("summary status or synthetic flag is invalid")
    if summary["mysql_tables"] != len(EXPECTED_TABLES) or summary["trade_records"] < EXPECTED_COUNTS["trade_records"] or summary["months"] != EXPECTED_MONTHS:
        raise ValueError("summary database contract is invalid")
    corpus = summary["corpus"]
    if not isinstance(corpus, dict) or corpus.get("sources") != 74 or corpus.get("documents") != 116 or corpus.get("chunks") != 120:
        raise ValueError("summary corpus chunks contract is invalid")
    completeness = summary["metadata_required_completeness"]
    if type(completeness) not in {int, float} or completeness != 1.0:
        raise ValueError("metadata_required_completeness must equal 1.0")
    if summary["quarantine_unexpected"] != 0:
        raise ValueError("unexpected quarantine records present")
    return summary


def _stage(stage: str, operation: Callable[[], Any]) -> Any:
    try:
        return operation()
    except FoundationSmokeFailure:
        raise
    except Exception as exc:
        raise FoundationSmokeFailure(stage, exc) from None


def _normalize_type(value: str) -> str:
    return value.lower().replace(" ", "").replace("numeric", "decimal")


def _expected_schema() -> dict[str, Any]:
    result: dict[str, Any] = {"tables": sorted(EXPECTED_TABLES), "columns": {}, "indexes": {}, "foreign_keys": {}}
    for table in Base.metadata.sorted_tables:
        columns = {}
        for column in table.columns:
            kind = _normalize_type(column.type.compile(dialect=mysql.dialect()))
            if kind in {"bool", "boolean"}:
                kind = "tinyint(1)"
            default = "current_timestamp" if column.server_default is not None else None
            columns[column.name] = {"type": kind, "nullable": bool(column.nullable), "default": default}
        result["columns"][table.name] = columns
        indexes: dict[str, Any] = {"PRIMARY": {"unique": True, "columns": [c.name for c in table.primary_key.columns]}}
        for index in table.indexes:
            if index.name:
                indexes[index.name] = {"unique": bool(index.unique), "columns": [c.name for c in index.columns]}
        for constraint in table.constraints:
            if constraint.name and constraint.__class__.__name__ == "UniqueConstraint":
                indexes[constraint.name] = {"unique": True, "columns": [c.name for c in constraint.columns]}
        result["indexes"][table.name] = indexes
        result["foreign_keys"][table.name] = sorted(
            [{"name": fk.constraint.name, "column": fk.parent.name, "table": fk.column.table.name, "referenced_column": fk.column.name, "on_update": "RESTRICT", "on_delete": "RESTRICT"} for fk in table.foreign_keys],
            key=canonical_json,
        )
    return result


def _live_schema(connection: Any) -> dict[str, Any]:
    tables = sorted(row[0] for row in connection.execute(text("SHOW TABLES")))
    columns: dict[str, dict[str, Any]] = {}
    indexes: dict[str, dict[str, Any]] = {}
    foreign_keys: dict[str, list[dict[str, str]]] = {}
    for table in tables:
        rows = connection.execute(text("""
            SELECT column_name, column_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = DATABASE() AND table_name = :table
            ORDER BY ordinal_position
        """), {"table": table})
        columns[table] = {row[0]: {"type": _normalize_type(row[1]), "nullable": row[2] == "YES", "default": (str(row[3]).lower().replace("()", "") if row[3] is not None else None)} for row in rows}
        rows = connection.execute(text("""
            SELECT index_name, non_unique, seq_in_index, column_name
            FROM information_schema.statistics
            WHERE table_schema = DATABASE() AND table_name = :table
            ORDER BY index_name, seq_in_index
        """), {"table": table})
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            current = grouped.setdefault(row[0], {"unique": not bool(row[1]), "columns": []})
            current["columns"].append(row[3])
        indexes[table] = grouped
        rows = connection.execute(text("""
            SELECT k.constraint_name AS constraint_name, k.column_name AS column_name,
                   k.referenced_table_name AS ref_table, k.referenced_column_name AS ref_column,
                   r.update_rule AS on_update, r.delete_rule AS on_delete
            FROM information_schema.key_column_usage k
            JOIN information_schema.referential_constraints r
              ON r.constraint_schema=k.constraint_schema AND r.constraint_name=k.constraint_name
             AND r.table_name=k.table_name
            WHERE k.table_schema = DATABASE() AND k.table_name = :table
              AND k.referenced_table_name IS NOT NULL
            ORDER BY k.constraint_name, k.ordinal_position
        """), {"table": table})
        foreign_keys[table] = sorted(({"name": row[0], "column": row[1], "table": row[2], "referenced_column": row[3], "on_update": "RESTRICT" if row[4] == "NO ACTION" else row[4], "on_delete": "RESTRICT" if row[5] == "NO ACTION" else row[5]} for row in rows), key=canonical_json)
    return {"tables": tables, "columns": columns, "indexes": indexes, "foreign_keys": foreign_keys}


def _json_safe(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value) if value.__class__.__name__ == "Decimal" else value


def _seed_snapshot(connection: Any, bundle: Any) -> tuple[dict[str, list[dict[str, Any]]], str]:
    models = {
        "countries": bundle.countries, "companies": bundle.companies,
        "hs_codes": bundle.hs_codes, "products": bundle.products,
        "data_sources": bundle.data_sources, "company_products": bundle.company_products,
        "trade_records": bundle.trade_records,
    }
    expected = {table: [_json_safe(item.model_dump(mode="json")) for item in rows] for table, rows in models.items()}
    actual: dict[str, list[dict[str, Any]]] = {}
    for table in EXPECTED_TABLES:
        rows = []
        for row in connection.execute(text(f"SELECT * FROM `{table}` ORDER BY id")):
            value = _json_safe(dict(row._mapping))
            if "is_synthetic" in value:
                value["is_synthetic"] = bool(value["is_synthetic"])
            rows.append(value)
        actual[table] = rows
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError("seed rows drift from deterministic bundle")
    return actual, canonical_hash(actual)


def _probe_mysql(settings: Settings) -> dict[str, Any]:
    # This is the only database URL constructed by the smoke runner.
    if settings.mysql.user != "trade_query":
        raise ValueError("runtime database user must be trade_query")
    engine = create_engine(database_url_from_environment(role="query"), pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            version = str(connection.scalar(text("SELECT VERSION()")))
            live_schema = _live_schema(connection)
            if live_schema["tables"] != sorted(EXPECTED_TABLES):
                raise ValueError("unexpected MySQL table set")
            expected_schema = _expected_schema()
            if live_schema != expected_schema:
                raise ValueError("MySQL schema contract drift")
            counts = {table: int(connection.scalar(text(f"SELECT COUNT(*) FROM `{table}`"))) for table in EXPECTED_TABLES}
            months = int(connection.scalar(text("SELECT COUNT(DISTINCT DATE_FORMAT(trade_date, '%Y-%m')) FROM trade_records")))
            bundle = generate_trade_seed(SEED)
            _, seed_hash = _seed_snapshot(connection, bundle)
            synthetic_flags = {
                "companies": int(connection.scalar(text("SELECT COUNT(*) FROM companies WHERE is_synthetic = 1"))) == counts["companies"],
                "data_sources": int(connection.scalar(text("SELECT COUNT(*) FROM data_sources WHERE is_synthetic = 1"))) == counts["data_sources"],
                "trade_record_ids": int(connection.scalar(text("SELECT COUNT(*) FROM trade_records WHERE raw_record_id LIKE 'SYN-%'"))) == counts["trade_records"],
            }
            if not all(synthetic_flags.values()):
                raise ValueError("database synthetic flags are incomplete")
    except Exception:
        engine.dispose()
        raise
    return {"engine": engine, "version": version, "schema": live_schema, "schema_fingerprint": canonical_hash(live_schema), "counts": counts, "months": months, "seed_hash": seed_hash, "synthetic_flags": synthetic_flags, "before_counts": counts.copy()}


def _assert_mysql_unchanged(mysql_state: dict[str, Any]) -> None:
    engine = mysql_state["engine"]
    with engine.connect() as connection:
        after = {table: int(connection.scalar(text(f"SELECT COUNT(*) FROM `{table}`"))) for table in EXPECTED_TABLES}
    if after != mysql_state["before_counts"]:
        raise ValueError("read-only smoke changed MySQL row counts")


def _http_json(host: str, port: int, path: str) -> dict[str, Any]:
    request = Request(f"http://{host}:{port}{path}", headers={"Accept": "application/json"})
    with urlopen(request, timeout=3) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("service response is not an object")
    return payload


def _http_ok(host: str, port: int, path: str) -> None:
    request = Request(f"http://{host}:{port}{path}")
    with urlopen(request, timeout=3) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")


def _probe_redis(settings: Settings) -> dict[str, Any]:
    import redis
    client = redis.Redis(host=settings.redis.host, port=settings.redis.port, db=settings.redis.database, socket_timeout=3, decode_responses=True)
    if client.ping() is not True:
        raise RuntimeError("Redis PING failed")
    info = client.info(section="server")
    return {"ready": True, "live": str(info.get("redis_version", "unknown"))}


def _probe_milvus(settings: Settings) -> dict[str, Any]:
    from pymilvus import MilvusClient
    client = MilvusClient(uri=f"http://{settings.milvus.host}:{settings.milvus.port}", db_name=settings.milvus.database)
    try:
        collections = client.list_collections()
        if not isinstance(collections, list):
            raise ValueError("Milvus list_collections response is malformed")
        version = client.get_server_version() if hasattr(client, "get_server_version") else "unknown"
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    return {"ready": True, "live": str(version), "collections": sorted(str(item) for item in collections)}


def _file_hashes(root: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = exclude or set()
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.rglob("*")) if path.is_file() and path.relative_to(root).as_posix() not in excluded}


def _tree_hash(files: dict[str, str]) -> str:
    return hashlib.sha256(canonical_json(files).encode("utf-8")).hexdigest()


def _run_corpus_and_ingestion(temp_root: Path) -> dict[str, Any]:
    generated_root = temp_root / "generated-corpus"
    generate_demo_corpus(generated_root, seed=SEED, clean=True)
    checked_in_root = _resource_path("demo/trade_intel_seed")
    # README.md is checked in as explanatory documentation and intentionally
    # preserved by the generator, but is not a generated source/manifest file.
    generated_files = _file_hashes(generated_root, exclude={"README.md"})
    checked_in_files = _file_hashes(checked_in_root, exclude={"README.md"})
    if generated_files != checked_in_files:
        raise ValueError("generated corpus differs from checked-in deterministic corpus")
    import yaml
    payload = yaml.safe_load(_resource_path("data/sources/trade_intel_demo.yaml").read_text(encoding="utf-8"))
    payload["root"] = generated_root.as_posix()
    catalog_path = temp_root / "generated-catalog.yaml"
    catalog_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    first = IngestionPipeline().run(SourceCatalog.from_yaml(catalog_path), temp_root / "build-a.json")
    second = IngestionPipeline().run(SourceCatalog.from_yaml(catalog_path), temp_root / "build-b.json")
    if (temp_root / "build-a.json").read_bytes() != (temp_root / "build-b.json").read_bytes() or first.build_id != second.build_id or first.chunk_ids != second.chunk_ids:
        raise ValueError("ingestion is not deterministic")
    reloaded = BuildManifest.model_validate_json((temp_root / "build-a.json").read_text(encoding="utf-8"))
    if reloaded != first:
        raise ValueError("persisted manifest reload differs from live build")
    _validate_ingestion_counts(first)
    source_types = {item.source_type for item in first.sources if item.source_type}
    if source_types != EXPECTED_SOURCE_TYPES or any(item.source_type == "trade_ledger" for item in first.sources):
        raise ValueError("corpus source type contract is invalid")
    if first.source_type_counts != EXPECTED_SOURCE_COUNTS or first.file_type_counts != EXPECTED_FILE_COUNTS:
        raise ValueError("corpus file/source type counts are invalid")
    if not all(item.restore().is_synthetic for item in first.documents) or not all(item.restore().metadata.is_synthetic for item in first.chunks):
        raise ValueError("corpus synthetic flags are not complete")
    expected_quarantine = [item for item in first.quarantined if item.source_path == EXPECTED_SCANNED_PATH and item.error_code == "scanned_pdf_ocr_unavailable"]
    unexpected = [item for item in first.quarantined if item not in expected_quarantine]
    if len(expected_quarantine) != 1 or unexpected:
        raise ValueError("corpus quarantine contract is invalid")
    scanned = next(item for item in first.sources if item.path == EXPECTED_SCANNED_PATH)
    if scanned.parser_backend != "pymupdf" or not scanned.degraded or first.parser_backends.mineru_statuses != ("unavailable",) or "mineru_unavailable" not in first.parser_backends.degraded_components:
        raise ValueError("PDF parser degradation state is not truthful")
    if not first.metadata_complete or not first.chunks:
        raise ValueError("chunk metadata completeness is not proven")
    return {
        "manifest": first,
        "tree_hash": _tree_hash(generated_files),
        "manifest_hash": hashlib.sha256((generated_root / "manifests/corpus_manifest.json").read_bytes()).hexdigest(),
        "quarantine_expected": len(expected_quarantine),
        "quarantine_unexpected": len(unexpected),
    }


def _validate_ingestion_counts(manifest: BuildManifest) -> None:
    if (len(manifest.sources), len(manifest.documents), len(manifest.chunks)) != (74, 116, 120):
        raise ValueError("ingestion corpus counts are not exact")


def _run_corpus_in_temp() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="trade-foundation-smoke-") as directory:
        return _run_corpus_and_ingestion(Path(directory))


def _declared_versions() -> dict[str, str]:
    import yaml
    compose = yaml.safe_load(_resource_path("docker-compose.yml").read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for name in ("mysql", "etcd", "minio", "milvus", "redis"):
        service = compose["services"][name]
        result[name] = str(service.get("image") or service.get("build", {}).get("args", {}).get("MYSQL_BASE_IMAGE", "unknown"))
    return result


def run_foundation_smoke(*, settings: Settings | None = None) -> dict[str, Any]:
    """Run all foundation gates and return a strict JSON-serializable summary."""
    config = _stage("config", lambda: settings or Settings.load(runtime="compose"))
    mysql_state: dict[str, Any] | None = None
    try:
        mysql_state = _stage("mysql", lambda: _probe_mysql(config))
        # Keep each dependency as a distinct stage so a malformed or unavailable
        # service response cannot be hidden behind a generic "services" failure.
        services = {
            "mysql": {"ready": True, "live": mysql_state["version"]},
            "redis": _stage("redis", lambda: _probe_redis(config)),
            "milvus": _stage("milvus", lambda: _probe_milvus(config)),
        }
        import os
        etcd_host = os.environ.get("ETCD__HOST", "etcd")
        etcd_port = int(os.environ.get("ETCD__PORT", "2379"))
        etcd_health = _stage("etcd", lambda: _http_json(etcd_host, etcd_port, "/health"))
        etcd_version = _stage("etcd", lambda: _http_json(etcd_host, etcd_port, "/version"))
        if str(etcd_health.get("health", "")).lower() not in {"true", "ok"}:
            raise FoundationSmokeFailure("etcd", RuntimeError("health not ready"))
        minio_host = os.environ.get("MINIO__HOST", "minio")
        minio_port = int(os.environ.get("MINIO__PORT", "9000"))
        _stage("minio", lambda: _http_ok(minio_host, minio_port, "/minio/health/live"))
        services["etcd"] = {"ready": True, "live": etcd_version}
        services["minio"] = {"ready": True, "live": "healthy"}
        corpus_state = _stage("corpus", _run_corpus_in_temp)
        manifest = corpus_state["manifest"]
        _stage("read_only", lambda: _assert_mysql_unchanged(mysql_state))
        summary: dict[str, Any] = {
            "status": "ok", "synthetic_only": True,
            "service_versions": {name: {"declared": declared, **services.get(name, {})} for name, declared in _declared_versions().items()},
            "service_readiness": {name: bool(value.get("ready")) for name, value in services.items()},
            "mysql_tables": len(mysql_state["schema"]["tables"]), "row_counts": mysql_state["counts"],
            "trade_records": mysql_state["counts"]["trade_records"], "months": mysql_state["months"],
            "schema_fingerprint": mysql_state["schema_fingerprint"], "seed_hash": mysql_state["seed_hash"],
            "synthetic_flags": mysql_state["synthetic_flags"],
            "corpus": {"sources": len(manifest.sources), "documents": len(manifest.documents), "chunks": len(manifest.chunks), "tree_hash": corpus_state["tree_hash"], "manifest_hash": corpus_state["manifest_hash"]},
            "build_id": manifest.build_id, "config_hash": manifest.config_hash, "fingerprint": manifest.fingerprint,
            "parser_backends": manifest.parser_backends.model_dump(mode="json"),
            "chunk_counts_by_source_type": dict(sorted(Counter(chunk.restore().metadata.source_type.value for chunk in manifest.chunks).items())),
            "chunk_counts_by_file_type": dict(sorted(Counter(chunk.restore().metadata.file_type.value for chunk in manifest.chunks).items())),
            "quarantine_expected": corpus_state["quarantine_expected"], "quarantine_unexpected": corpus_state["quarantine_unexpected"],
            "metadata_required_completeness": 1.0 if manifest.chunks and manifest.metadata_complete else 0.0,
            "deterministic": {"seed": True, "corpus": True, "ingestion": True, "output_paths_excluded": True},
        }
        return validate_foundation_summary(summary)
    finally:
        if mysql_state is not None:
            mysql_state["engine"].dispose()


def main() -> int:
    parser = _SafeArgumentParser(description="Verify the synthetic trade-agent foundation.", add_help=False)
    try:
        parser.parse_args()
        print(json.dumps(run_foundation_smoke(), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0
    except argparse.ArgumentError:
        print(json.dumps(failure_payload("arguments", "ArgumentError"), sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    except FoundationSmokeFailure as exc:
        print(json.dumps(failure_payload(exc.stage, exc.exception_class), sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 1
    except Exception as exc:
        print(json.dumps(failure_payload("validation", exc), sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
