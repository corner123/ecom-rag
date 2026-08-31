from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def test_compose_has_exact_pinned_foundation_services_and_loopback_ports() -> None:
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert set(services) == {"mysql", "etcd", "minio", "milvus", "redis", "api"}
    assert services["milvus"]["image"] == "milvusdb/milvus:v2.6.22"
    assert services["etcd"]["image"] == "quay.io/coreos/etcd:v3.5.25"
    assert services["minio"]["image"] == "minio/minio:RELEASE.2024-12-18T13-15-44Z"
    assert services["redis"]["image"] == "redis:7.4-alpine"
    assert services["mysql"]["build"]["args"]["MYSQL_BASE_IMAGE"] == "mysql:8.4"
    for service in services.values():
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1:")


def test_compose_api_receives_only_query_database_secret() -> None:
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    api_environment = compose["services"]["api"]["environment"]
    assert "MYSQL__QUERY_PASSWORD" in api_environment
    assert not any("ROOT_PASSWORD" in key or "MIGRATION_PASSWORD" in key for key in api_environment)


@pytest.mark.integration
def test_foundation_smoke_returns_machine_readable_summary() -> None:
    from scripts.smoke_foundation import run_foundation_smoke

    summary = run_foundation_smoke()
    assert summary["mysql_tables"] == 7
    assert summary["trade_records"] >= 800
    assert summary["quarantine_unexpected"] == 0
    assert summary["metadata_required_completeness"] == 1.0
