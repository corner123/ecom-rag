from trade_agent.server import readiness_payload


def test_readiness_requires_api_and_dependencies():
    payload = readiness_payload(lambda host, port: host == "mysql" and port == 3306)
    assert payload["api"] is True
    assert payload["dependencies"]["mysql"] is True
    assert payload["dependencies"]["redis"] is False
    assert payload["ready"] is False
