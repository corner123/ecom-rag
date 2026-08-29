def test_package_exposes_version():
    import trade_agent

    assert trade_agent.__version__ == "0.1.0"
