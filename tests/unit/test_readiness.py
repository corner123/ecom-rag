from trade_agent.server import readiness_payload


def test_readiness_requires_api_and_dependencies():
    payload = readiness_payload(lambda host, port: host == "mysql" and port == 3306)
    assert payload["api"] is True
    assert payload["dependencies"]["mysql"] is True
    assert payload["dependencies"]["redis"] is False
    assert payload["ready"] is False


def test_legacy_readiness_payload_is_pure_under_service_host_overrides(monkeypatch):
    monkeypatch.setenv("MYSQL__HOST", "127.0.0.1")
    observed = []

    def probe(host, port):
        observed.append((host, port))
        return host == "mysql" and port == 3306

    payload = readiness_payload(probe)

    assert observed[0] == ("mysql", 3306)
    assert payload["dependencies"]["mysql"] is True
