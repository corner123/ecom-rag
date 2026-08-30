"""Deterministic, explicitly synthetic foreign-trade fixtures."""
from __future__ import annotations

import argparse
import random
from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel
from sqlalchemy import Engine, insert

from trade_agent.db.models import Company, CompanyProduct, Country, DataSource, HsCode, Product, TradeRecord

FIXED_TIMESTAMP = datetime(2026, 8, 30, 0, 0, 0)


class CountrySeed(BaseModel):
    id: int; country_code: str; country_name: str; region: str; created_at: datetime = FIXED_TIMESTAMP


class CompanySeed(BaseModel):
    id: int; company_name: str; normalized_name: str; country_id: int; company_type: str; website: str; website_domain: str; registration_id: str; address: str; industry: str; is_synthetic: bool = True; created_at: datetime = FIXED_TIMESTAMP


class HsCodeSeed(BaseModel):
    id: int; hs_code: str; description: str; category: str; parent_code: str | None = None; created_at: datetime = FIXED_TIMESTAMP


class ProductSeed(BaseModel):
    id: int; product_name: str; sku: str; hs_code_id: int; category: str; description: str; created_at: datetime = FIXED_TIMESTAMP


class DataSourceSeed(BaseModel):
    id: int; source_name: str; source_type: str; source_url: str; update_frequency: str; last_updated_at: datetime = FIXED_TIMESTAMP; is_synthetic: bool = True; created_at: datetime = FIXED_TIMESTAMP


class CompanyProductSeed(BaseModel):
    id: int; company_id: int; product_id: int; relation_type: str; created_at: datetime = FIXED_TIMESTAMP


class TradeRecordSeed(BaseModel):
    id: int; raw_record_id: str; source_id: int; importer_id: int; exporter_id: int; product_id: int; hs_code_id: int; import_country_id: int; export_country_id: int; trade_date: date; quantity: Decimal; unit: str; trade_amount: Decimal; currency: str; created_at: datetime = FIXED_TIMESTAMP


class TradeSeedBundle(BaseModel):
    countries: list[CountrySeed]
    companies: list[CompanySeed]
    hs_codes: list[HsCodeSeed]
    products: list[ProductSeed]
    data_sources: list[DataSourceSeed]
    company_products: list[CompanyProductSeed]
    trade_records: list[TradeRecordSeed]
    lead_labels: dict[int, str]


class SeedSummary(BaseModel):
    countries: int; companies: int; hs_codes: int; products: int; data_sources: int; company_products: int; trade_records: int; months: int


_COUNTRIES = [("CN", "China", "Asia"), ("US", "United States", "North America"), ("DE", "Germany", "Europe"), ("VN", "Vietnam", "Asia"), ("BR", "Brazil", "South America"), ("AE", "United Arab Emirates", "Middle East"), ("ZA", "South Africa", "Africa"), ("AU", "Australia", "Oceania")]
_HS = [("010121", "Pure-bred breeding horses", "Live animals"), ("020130", "Boneless bovine meat", "Meat"), ("040221", "Milk powder", "Dairy"), ("090111", "Coffee, not roasted", "Coffee"), ("100630", "Semi-milled rice", "Cereals"), ("271019", "Petroleum oils", "Mineral fuels"), ("390761", "Polyethylene terephthalate", "Plastics"), ("610910", "Cotton T-shirts", "Textiles"), ("730890", "Iron or steel structures", "Metals"), ("847130", "Portable computers", "Machinery"), ("850440", "Static converters", "Electronics"), ("940360", "Wooden furniture", "Furniture")]
_COMPANY_WORDS = [("Harbor", "Imports"), ("Summit", "Trading"), ("River", "Exports"), ("Atlas", "Supply"), ("Cedar", "Manufacturing"), ("Orchid", "Commerce"), ("Pioneer", "Logistics"), ("Meridian", "Industries")]


def _month(index: int) -> date:
    year, month = divmod(2 + index, 12)
    return date(2025 + year, month + 1, 1)


def generate_trade_seed(seed: int = 20260830) -> TradeSeedBundle:
    rng = random.Random(seed)
    countries = [CountrySeed(id=i + 1, country_code=code, country_name=name, region=region) for i, (code, name, region) in enumerate(_COUNTRIES)]
    companies = []
    labels: dict[int, str] = {}
    for index in range(60):
        country = countries[index % len(countries)]
        prefix, suffix = _COMPANY_WORDS[index % len(_COMPANY_WORDS)]
        company_name = f"{prefix} {country.country_code} {suffix} {index + 1:02d}"
        cohort = ("growing", "declining", "dormant", "active")[index // 15]
        labels[index + 1] = cohort
        domain = f"company-{index + 1:02d}.example"
        companies.append(CompanySeed(id=index + 1, company_name=company_name, normalized_name=company_name.lower(), country_id=country.id, company_type="importer" if index < 30 else "exporter", website=f"https://{domain}", website_domain=domain, registration_id=f"SYN-{country.country_code}-{index + 1:04d}", address=f"{index + 1} Synthetic Trade Way, {country.country_name}", industry="synthetic trade intelligence"))
    hs_codes = [HsCodeSeed(id=i + 1, hs_code=code, description=description, category=category, parent_code=code[:4]) for i, (code, description, category) in enumerate(_HS)]
    products = [ProductSeed(id=i + 1, product_name=f"Synthetic {hs_codes[i % 12].category} Product {i + 1:02d}", sku=f"SYN-SKU-{i + 1:03d}", hs_code_id=(i % 12) + 1, category=hs_codes[i % 12].category, description=f"Synthetic demonstration product for HS {hs_codes[i % 12].hs_code}") for i in range(30)]
    sources = [DataSourceSeed(id=1, source_name="Synthetic Customs Monthly", source_type="customs_profile", source_url="https://customs.synthetic.example/monthly", update_frequency="monthly"), DataSourceSeed(id=2, source_name="Synthetic Trade Ledger", source_type="trade_ledger", source_url="https://ledger.synthetic.example/records", update_frequency="monthly")]
    company_products = [CompanyProductSeed(id=i + 1, company_id=i + 1, product_id=(i % 30) + 1, relation_type="trades") for i in range(60)]
    records: list[TradeRecordSeed] = []
    record_id = 1
    for month_index in range(18):
        for company in companies:
            label = labels[company.id]
            if label == "dormant" and month_index >= 9:
                continue
            if label == "active" and month_index < 8:
                continue
            product_id = ((company.id + month_index - 1) % 30) + 1
            hs_code_id = ((product_id - 1) % 12) + 1
            multiplier = (month_index + 2) if label == "growing" else (19 - month_index) if label == "declining" else 10
            amount = Decimal(rng.randint(900, 2200) * multiplier).quantize(Decimal("0.01"))
            quantity = (amount / Decimal("12.5")).quantize(Decimal("0.001"))
            exporter_id = ((company.id + 29) % 60) + 1
            records.append(TradeRecordSeed(id=record_id, raw_record_id=f"SYN-{month_index + 1:02d}-{company.id:03d}", source_id=(record_id % 2) + 1, importer_id=company.id, exporter_id=exporter_id, product_id=product_id, hs_code_id=hs_code_id, import_country_id=company.country_id, export_country_id=companies[exporter_id - 1].country_id, trade_date=_month(month_index), quantity=quantity, unit="kg", trade_amount=amount, currency="USD"))
            record_id += 1
    return TradeSeedBundle(countries=countries, companies=companies, hs_codes=hs_codes, products=products, data_sources=sources, company_products=company_products, trade_records=records, lead_labels=labels)


def _rows(values: list[BaseModel]) -> list[dict[str, object]]:
    return [value.model_dump(mode="python") for value in values]


def seed_database(engine: Engine, bundle: TradeSeedBundle) -> SeedSummary:
    groups = [(Country, bundle.countries), (HsCode, bundle.hs_codes), (Company, bundle.companies), (Product, bundle.products), (DataSource, bundle.data_sources), (CompanyProduct, bundle.company_products), (TradeRecord, bundle.trade_records)]
    with engine.begin() as connection:
        for model, values in groups:
            connection.execute(insert(model).prefix_with("IGNORE"), _rows(values))
    return SeedSummary(countries=len(bundle.countries), companies=len(bundle.companies), hs_codes=len(bundle.hs_codes), products=len(bundle.products), data_sources=len(bundle.data_sources), company_products=len(bundle.company_products), trade_records=len(bundle.trade_records), months=len({record.trade_date.strftime("%Y-%m") for record in bundle.trade_records}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    from sqlalchemy import create_engine
    from trade_agent.db.session import database_url_from_environment
    seed_database(create_engine(database_url_from_environment(role="migration")), generate_trade_seed(args.seed))


if __name__ == "__main__":
    main()
