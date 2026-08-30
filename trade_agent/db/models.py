"""SQLAlchemy representation of the authoritative MySQL DDL."""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Index, Numeric, String, UniqueConstraint, func
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


MYSQL_ID = mysql.BIGINT(unsigned=True)


class Country(Base):
    __tablename__ = "countries"
    __table_args__ = (UniqueConstraint("country_code", name="uq_countries_country_code"),)
    id: Mapped[int] = mapped_column(MYSQL_ID, primary_key=True)
    country_code: Mapped[str] = mapped_column(mysql.CHAR(2), nullable=False)
    country_name: Mapped[str] = mapped_column(String(100), nullable=False)
    region: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.current_timestamp())


class Company(Base):
    __tablename__ = "companies"
    __table_args__ = (UniqueConstraint("registration_id", name="uq_companies_registration_id"), Index("ix_companies_country_id", "country_id"), Index("ix_companies_normalized_name", "normalized_name"))
    id: Mapped[int] = mapped_column(MYSQL_ID, primary_key=True)
    company_name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(255), nullable=False)
    country_id: Mapped[int] = mapped_column(ForeignKey("countries.id", name="fk_companies_country"), nullable=False)
    company_type: Mapped[str] = mapped_column(String(64), nullable=False)
    website: Mapped[str] = mapped_column(String(255), nullable=False)
    website_domain: Mapped[str] = mapped_column(String(255), nullable=False)
    registration_id: Mapped[str] = mapped_column(String(100), nullable=False)
    address: Mapped[str] = mapped_column(String(500), nullable=False)
    industry: Mapped[str] = mapped_column(String(128), nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.current_timestamp())


class HsCode(Base):
    __tablename__ = "hs_codes"
    __table_args__ = (UniqueConstraint("hs_code", name="uq_hs_codes_hs_code"),)
    id: Mapped[int] = mapped_column(MYSQL_ID, primary_key=True)
    hs_code: Mapped[str] = mapped_column(String(12), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=False)
    category: Mapped[str] = mapped_column(String(128), nullable=False)
    parent_code: Mapped[str | None] = mapped_column(String(12))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.current_timestamp())


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (UniqueConstraint("sku", name="uq_products_sku"), Index("ix_products_hs_code_id", "hs_code_id"))
    id: Mapped[int] = mapped_column(MYSQL_ID, primary_key=True)
    product_name: Mapped[str] = mapped_column(String(255), nullable=False)
    sku: Mapped[str] = mapped_column(String(100), nullable=False)
    hs_code_id: Mapped[int] = mapped_column(ForeignKey("hs_codes.id", name="fk_products_hs_code"), nullable=False)
    category: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.current_timestamp())


class DataSource(Base):
    __tablename__ = "data_sources"
    __table_args__ = (UniqueConstraint("source_url", name="uq_data_sources_source_url"),)
    id: Mapped[int] = mapped_column(MYSQL_ID, primary_key=True)
    source_name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_url: Mapped[str] = mapped_column(String(255), nullable=False)
    update_frequency: Mapped[str] = mapped_column(String(64), nullable=False)
    last_updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.current_timestamp())


class CompanyProduct(Base):
    __tablename__ = "company_products"
    __table_args__ = (UniqueConstraint("company_id", "product_id", name="uq_company_products_company_product"), Index("ix_company_products_product_id", "product_id"))
    id: Mapped[int] = mapped_column(MYSQL_ID, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", name="fk_company_products_company"), nullable=False)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id", name="fk_company_products_product"), nullable=False)
    relation_type: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.current_timestamp())


class TradeRecord(Base):
    __tablename__ = "trade_records"
    __table_args__ = (UniqueConstraint("source_id", "raw_record_id", name="uq_trade_records_source_raw_record"), Index("ix_trade_records_hs_date", "hs_code_id", "trade_date"), Index("ix_trade_records_importer_date", "importer_id", "trade_date"), Index("ix_trade_records_exporter_date", "exporter_id", "trade_date"), Index("ix_trade_records_product_id", "product_id"), Index("ix_trade_records_export_country_id", "export_country_id"), Index("ix_trade_records_country_hs_date", "import_country_id", "hs_code_id", "trade_date"))
    id: Mapped[int] = mapped_column(MYSQL_ID, primary_key=True)
    raw_record_id: Mapped[str] = mapped_column(String(100), nullable=False)
    source_id: Mapped[int] = mapped_column(ForeignKey("data_sources.id", name="fk_trade_records_source"), nullable=False)
    importer_id: Mapped[int] = mapped_column(ForeignKey("companies.id", name="fk_trade_records_importer"), nullable=False)
    exporter_id: Mapped[int] = mapped_column(ForeignKey("companies.id", name="fk_trade_records_exporter"), nullable=False)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id", name="fk_trade_records_product"), nullable=False)
    hs_code_id: Mapped[int] = mapped_column(ForeignKey("hs_codes.id", name="fk_trade_records_hs_code"), nullable=False)
    import_country_id: Mapped[int] = mapped_column(ForeignKey("countries.id", name="fk_trade_records_import_country"), nullable=False)
    export_country_id: Mapped[int] = mapped_column(ForeignKey("countries.id", name="fk_trade_records_export_country"), nullable=False)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 3), nullable=False)
    unit: Mapped[str] = mapped_column(String(32), nullable=False)
    trade_amount: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    currency: Mapped[str] = mapped_column(mysql.CHAR(3), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.current_timestamp())
