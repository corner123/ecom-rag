import pytest

from trade_agent.config.settings import RuntimeLimits, Settings


def test_settings_reject_placeholder_password(monkeypatch):
    monkeypatch.setenv("MYSQL_APP_PASSWORD", "change-me")
    with pytest.raises(ValueError, match="placeholder"):
        Settings.load(runtime="compose")


def test_nested_environment_is_loaded(monkeypatch):
    monkeypatch.setenv("MYSQL__PASSWORD", "safe-password")
    settings = Settings.load(runtime="compose")
    assert settings.mysql.password == "safe-password"
    assert settings.environment == "compose"


def test_limits_are_positive():
    limits = RuntimeLimits(max_graph_steps=12, max_retries=1, max_llm_calls=5)
    assert limits.max_graph_steps == 12


def test_limits_reject_zero_graph_steps():
    with pytest.raises(ValueError):
        RuntimeLimits(max_graph_steps=0)
