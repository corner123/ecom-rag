from decimal import Decimal

from trade_agent.db.seed import generate_trade_seed


def test_seed_is_deterministic_and_synthetic():
    left = generate_trade_seed(20260830)
    right = generate_trade_seed(20260830)

    assert left.model_dump(mode="json") == right.model_dump(mode="json")
    assert len(left.countries) >= 8
    assert len(left.companies) >= 60
    assert len(left.hs_codes) >= 12
    assert len(left.products) >= 30
    assert len(left.trade_records) >= 800
    assert all(company.is_synthetic for company in left.companies)
    assert all(source.is_synthetic for source in left.data_sources)
    assert {"growing", "declining", "dormant", "active"} <= set(left.lead_labels.values())


def test_seed_preserves_trade_contract_invariants():
    bundle = generate_trade_seed()

    assert len({(record.source_id, record.raw_record_id) for record in bundle.trade_records}) == len(
        bundle.trade_records
    )
    assert len({record.trade_date.strftime("%Y-%m") for record in bundle.trade_records}) == 18
    assert any(code.hs_code.startswith("0") for code in bundle.hs_codes)
    assert all(isinstance(record.trade_amount, Decimal) for record in bundle.trade_records)
    assert all(company.website.endswith(".example") for company in bundle.companies)
    assert all(source.source_url.split("/")[2].endswith(".example") for source in bundle.data_sources)
