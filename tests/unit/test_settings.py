import pytest

from trade_agent.config.settings import ModelSettings, RuntimeLimits, Settings


def test_settings_reject_placeholder_password(monkeypatch):
    monkeypatch.setenv("MYSQL_APP_PASSWORD", "change-me")
    with pytest.raises(ValueError, match="placeholder"):
        Settings.load(runtime="compose")


def test_nested_environment_is_loaded(monkeypatch):
    monkeypatch.setenv("MYSQL__PASSWORD", "safe-password")
    monkeypatch.setenv("MYSQL__ROOT_PASSWORD", "safe-root-password")
    settings = Settings.load(runtime="compose")
    assert settings.mysql.password == "safe-password"
    assert not hasattr(settings.mysql, "root_password")
    assert settings.environment == "compose"


def test_query_password_is_runtime_default_and_migration_values_are_not_loaded(monkeypatch):
    monkeypatch.delenv("MYSQL__PASSWORD", raising=False)
    monkeypatch.delenv("MYSQL_APP_PASSWORD", raising=False)
    monkeypatch.setenv("MYSQL__QUERY_PASSWORD", "query-only")
    monkeypatch.setenv("MYSQL__MIGRATION_PASSWORD", "migration-only")
    monkeypatch.setenv("MYSQL__ROOT_PASSWORD", "root-only")
    settings = Settings.load(runtime="compose")
    assert settings.mysql.user == "trade_query"
    assert settings.mysql.password == "query-only"
    assert not hasattr(settings.mysql, "migration_password")
    assert not hasattr(settings.mysql, "root_password")



def test_query_password_from_env_file_populates_runtime_password(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MYSQL__PASSWORD", raising=False)
    monkeypatch.delenv("MYSQL_APP_PASSWORD", raising=False)
    monkeypatch.delenv("MYSQL__QUERY_PASSWORD", raising=False)
    (tmp_path / ".env").write_text("MYSQL__QUERY_PASSWORD=file-query-secret\n", encoding="utf-8")

    settings = Settings.load(runtime="compose")

    assert settings.mysql.password == "file-query-secret"
    assert not hasattr(settings.mysql, "query_password")


def test_limits_are_positive():
    limits = RuntimeLimits(max_graph_steps=12, max_retries=1, max_llm_calls=5)
    assert limits.max_graph_steps == 12


def test_limits_reject_zero_graph_steps():
    with pytest.raises(ValueError):
        RuntimeLimits(max_graph_steps=0)


@pytest.mark.parametrize("value", ["replace-with-a-local-secret", "replace-with-a-local-root-secret"])
def test_committed_example_passwords_are_rejected(monkeypatch, value):
    monkeypatch.setenv("MYSQL__PASSWORD", value)
    monkeypatch.setenv("MYSQL__ROOT_PASSWORD", "safe-root-password")
    with pytest.raises(ValueError, match="placeholder"):
        Settings.load(runtime="development")


def test_missing_mysql_password_is_rejected(monkeypatch):
    monkeypatch.delenv("MYSQL__PASSWORD", raising=False)
    with pytest.raises(ValueError, match="required|placeholder"):
        Settings.load(runtime="test")


def test_model_settings_pin_bge_m3_and_expose_embedding_runtime_controls():
    models = ModelSettings()

    assert models.embedding_model == "BAAI/bge-m3"
    assert models.embedding_revision == "5617a9f61b028005a4858fdac845db406aefb181"
    assert models.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert models.reranker_revision == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    assert models.embedding_device == "cpu"
    assert models.embedding_batch_size > 0
    assert models.embedding_offline is False


def test_model_settings_reject_floating_embedding_revision():
    with pytest.raises(ValueError, match="immutable"):
        ModelSettings(embedding_revision="main")


def test_empty_embedding_cache_environment_is_ignored(monkeypatch):
    monkeypatch.setenv("MYSQL__QUERY_PASSWORD", "query-only")
    monkeypatch.setenv("MODELS__EMBEDDING_CACHE_DIR", "")

    assert Settings.load(runtime="compose").models.embedding_cache_dir is None
