from pathlib import Path

import yaml


def test_milvus_uses_minio_credentials_from_same_compose_variables():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    minio = compose["services"]["minio"]["environment"]
    milvus = compose["services"]["milvus"]["environment"]
    mysql = compose["services"]["mysql"]["environment"]
    api = compose["services"]["api"]["environment"]
    assert minio["MINIO_ROOT_USER"] == milvus["MINIO_ACCESS_KEY_ID"] == "minioadmin"
    assert milvus["MINIO_SECRET_ACCESS_KEY"] == "${MINIO_ROOT_PASSWORD:?set MINIO_ROOT_PASSWORD in .env}"
    assert "MQ_TYPE" in milvus
    assert "MYSQL_ROOT_PASSWORD" in mysql
    assert "MYSQL__ROOT_PASSWORD" not in api


def test_mysql_healthcheck_uses_the_migration_account():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    healthcheck = compose["services"]["mysql"]["healthcheck"]["test"]
    assert "-utrade_migrator" in healthcheck[-1]
    assert "MYSQL_MIGRATION_PASSWORD" in healthcheck[-1]
    assert "SELECT 1" in healthcheck[-1]


def test_mysql_base_image_contract_is_explicit():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    assert compose["services"]["mysql"]["build"]["args"]["MYSQL_BASE_IMAGE"] == "mysql:8.4"
    assert "ARG MYSQL_BASE_IMAGE" in Path("docker/mysql.Dockerfile").read_text()
