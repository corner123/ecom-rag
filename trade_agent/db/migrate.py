"""Apply the authoritative SQL migration using the migration account."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine


def migrate_database(engine: Engine) -> None:
    sql = (Path(__file__).resolve().parents[2] / "db" / "migrations" / "001_schema.sql").read_text(encoding="utf-8")
    with engine.begin() as connection:
        for statement in sql.split(";"):
            if statement.strip():
                connection.exec_driver_sql(statement)


def main() -> None:
    from sqlalchemy import create_engine
    from trade_agent.db.session import database_url_from_environment
    migrate_database(create_engine(database_url_from_environment(role="migration")))


if __name__ == "__main__":
    main()
