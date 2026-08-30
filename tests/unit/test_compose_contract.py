from pathlib import Path

import yaml


def test_milvus_uses_minio_credentials_from_same_compose_variables():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    minio = compose["services"]["minio"]["environment"]
    milvus = compose["services"]["milvus"]["environment"]
    mysql = compose["services"]["mysql"]["environment"]
    api = compose["services"]["api"]["environment"]
    assert minio["MINIO_ROOT_USER"] == milvus["MINIO_ACCESS_KEY_ID"] == "minioadmin"
    assert milvus["MINIO_SECRET_ACCESS_KEY"] == "${MINIO_ROOT_PASSWORD:-synthetic-demo-minio-only}"
    assert mysql["MYSQL_ROOT_PASSWORD"] == "${MYSQL__ROOT_PASSWORD:-synthetic-demo-root-only}"
    assert mysql["MYSQL_MIGRATION_PASSWORD"] == "${MYSQL__MIGRATION_PASSWORD:-synthetic-demo-migration-only}"
    assert mysql["MYSQL_QUERY_PASSWORD"] == "${MYSQL__QUERY_PASSWORD:-synthetic-demo-query-only}"
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


def test_api_has_only_the_query_database_secret():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    api = compose["services"]["api"]["environment"]
    mysql = compose["services"]["mysql"]["environment"]
    assert "MYSQL__QUERY_PASSWORD" in api
    assert "MYSQL__MIGRATION_PASSWORD" not in api
    assert "MYSQL__ROOT_PASSWORD" not in api
    assert {"MYSQL_ROOT_PASSWORD", "MYSQL_MIGRATION_PASSWORD", "MYSQL_QUERY_PASSWORD"} <= set(mysql)
