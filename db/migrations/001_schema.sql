CREATE TABLE IF NOT EXISTS countries (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  country_code CHAR(2) NOT NULL,
  country_name VARCHAR(100) NOT NULL,
  region VARCHAR(64) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  CONSTRAINT uq_countries_country_code UNIQUE (country_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS companies (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  company_name VARCHAR(255) NOT NULL,
  normalized_name VARCHAR(255) NOT NULL,
  country_id BIGINT UNSIGNED NOT NULL,
  company_type VARCHAR(64) NOT NULL,
  website VARCHAR(255) NOT NULL,
  website_domain VARCHAR(255) NOT NULL,
  registration_id VARCHAR(100) NOT NULL,
  address VARCHAR(500) NOT NULL,
  industry VARCHAR(128) NOT NULL,
  is_synthetic BOOLEAN NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  CONSTRAINT uq_companies_registration_id UNIQUE (registration_id),
  CONSTRAINT fk_companies_country FOREIGN KEY (country_id) REFERENCES countries(id),
  INDEX ix_companies_country_id (country_id),
  INDEX ix_companies_normalized_name (normalized_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS hs_codes (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  hs_code VARCHAR(12) NOT NULL,
  description VARCHAR(500) NOT NULL,
  category VARCHAR(128) NOT NULL,
  parent_code VARCHAR(12) NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  CONSTRAINT uq_hs_codes_hs_code UNIQUE (hs_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS products (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  product_name VARCHAR(255) NOT NULL,
  sku VARCHAR(100) NOT NULL,
  hs_code_id BIGINT UNSIGNED NOT NULL,
  category VARCHAR(128) NOT NULL,
  description VARCHAR(500) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  CONSTRAINT uq_products_sku UNIQUE (sku),
  CONSTRAINT fk_products_hs_code FOREIGN KEY (hs_code_id) REFERENCES hs_codes(id),
  INDEX ix_products_hs_code_id (hs_code_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS data_sources (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  source_name VARCHAR(255) NOT NULL,
  source_type VARCHAR(64) NOT NULL,
  source_url VARCHAR(255) NOT NULL,
  update_frequency VARCHAR(64) NOT NULL,
  last_updated_at DATETIME NOT NULL,
  is_synthetic BOOLEAN NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  CONSTRAINT uq_data_sources_source_url UNIQUE (source_url)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS company_products (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  company_id BIGINT UNSIGNED NOT NULL,
  product_id BIGINT UNSIGNED NOT NULL,
  relation_type VARCHAR(64) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  CONSTRAINT uq_company_products_company_product UNIQUE (company_id, product_id),
  CONSTRAINT fk_company_products_company FOREIGN KEY (company_id) REFERENCES companies(id),
  CONSTRAINT fk_company_products_product FOREIGN KEY (product_id) REFERENCES products(id),
  INDEX ix_company_products_product_id (product_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS trade_records (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  raw_record_id VARCHAR(100) NOT NULL,
  source_id BIGINT UNSIGNED NOT NULL,
  importer_id BIGINT UNSIGNED NOT NULL,
  exporter_id BIGINT UNSIGNED NOT NULL,
  product_id BIGINT UNSIGNED NOT NULL,
  hs_code_id BIGINT UNSIGNED NOT NULL,
  import_country_id BIGINT UNSIGNED NOT NULL,
  export_country_id BIGINT UNSIGNED NOT NULL,
  trade_date DATE NOT NULL,
  quantity DECIMAL(18,3) NOT NULL,
  unit VARCHAR(32) NOT NULL,
  trade_amount DECIMAL(18,2) NOT NULL,
  currency CHAR(3) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  CONSTRAINT uq_trade_records_source_raw_record UNIQUE (source_id, raw_record_id),
  CONSTRAINT fk_trade_records_source FOREIGN KEY (source_id) REFERENCES data_sources(id),
  CONSTRAINT fk_trade_records_importer FOREIGN KEY (importer_id) REFERENCES companies(id),
  CONSTRAINT fk_trade_records_exporter FOREIGN KEY (exporter_id) REFERENCES companies(id),
  CONSTRAINT fk_trade_records_product FOREIGN KEY (product_id) REFERENCES products(id),
  CONSTRAINT fk_trade_records_hs_code FOREIGN KEY (hs_code_id) REFERENCES hs_codes(id),
  CONSTRAINT fk_trade_records_import_country FOREIGN KEY (import_country_id) REFERENCES countries(id),
  CONSTRAINT fk_trade_records_export_country FOREIGN KEY (export_country_id) REFERENCES countries(id),
  INDEX ix_trade_records_hs_date (hs_code_id, trade_date),
  INDEX ix_trade_records_importer_date (importer_id, trade_date),
  INDEX ix_trade_records_exporter_date (exporter_id, trade_date),
  INDEX ix_trade_records_product_id (product_id),
  INDEX ix_trade_records_export_country_id (export_country_id),
  INDEX ix_trade_records_country_hs_date (import_country_id, hs_code_id, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
