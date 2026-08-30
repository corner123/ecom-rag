from pathlib import Path

import yaml


def test_milvus_uses_minio_credentials_from_same_compose_variables():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    minio = compose["services"]["minio"]["environment"]
    milvus = compose["services"]["milvus"]["environment"]
    assert minio["MINIO_ROOT_USER"] == milvus["MINIO_ACCESS_KEY_ID"] == "minioadmin"
    assert milvus["MINIO_SECRET_ACCESS_KEY"] == "${MINIO_ROOT_PASSWORD:?set MINIO_ROOT_PASSWORD in .env}"
    assert "MQ_TYPE" in milvus
