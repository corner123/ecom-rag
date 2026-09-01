from __future__ import annotations

import os
import re
import shutil
import subprocess
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


def test_etcd_persists_to_the_declared_data_directory() -> None:
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    etcd = compose["services"]["etcd"]
    assert "--data-dir=/etcd-data" in etcd["command"]
    assert etcd["volumes"] == ["etcd_data:/etcd-data"]


def test_readme_does_not_source_env_or_expand_secret_values_in_argv() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "source .env" not in readme
    assert ". ./.env" not in readme
    assert re.search(r"\bread[^\n]*-p", readme) is None
    assert '-e MYSQL__MIGRATION_PASSWORD="$' not in readme
    assert "-e MYSQL__MIGRATION_PASSWORD" in readme
    assert "getpass.getpass" in readme


@pytest.mark.parametrize("shell", ["bash", "zsh"])
def test_readme_secret_setup_runs_in_bash_and_zsh_without_env_placeholders(shell: str, tmp_path: Path) -> None:
    shell_path = shutil.which(shell)
    assert shell_path is not None, f"required shell is unavailable: {shell}"
    readme = Path("README.md").read_text(encoding="utf-8")
    setup = readme.split("```bash", 1)[1].split("```", 1)[0]
    setup = setup.replace("$EDITOR .env", ":")
    shutil.copyfile(".env.example", tmp_path / ".env.example")
    script = """
python3() { printf '%s\\n' 'simulated-secret'; }
""" + setup + """
test "$MYSQL__ROOT_PASSWORD" = 'simulated-secret'
test "$MYSQL__MIGRATION_PASSWORD" = 'simulated-secret'
test "$MYSQL__QUERY_PASSWORD" = 'simulated-secret'
test "$MINIO_ROOT_PASSWORD" = 'simulated-secret'
printf '%s\\n' ready
"""
    result = subprocess.run(
        [shell_path, "-c", script], cwd=tmp_path, env={"PATH": os.environ["PATH"]},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ready"
    assert "simulated-secret" not in result.stdout
    assert "replace-with-" not in result.stdout


@pytest.mark.integration
def test_foundation_smoke_returns_machine_readable_summary() -> None:
    from scripts.smoke_foundation import run_foundation_smoke

    summary = run_foundation_smoke()
    assert summary["mysql_tables"] == 7
    assert summary["trade_records"] >= 800
    assert summary["quarantine_unexpected"] == 0
    assert summary["metadata_required_completeness"] == 1.0
