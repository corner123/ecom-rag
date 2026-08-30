#!/bin/sh
set -eu

sql_escape() {
  if printf '%s' "$1" | LC_ALL=C od -An -tu1 | awk '{for (i = 1; i <= NF; i++) if ($i < 32 || $i == 127) exit 1}'; then :; else
    return 1
  fi
  printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e "s/'/''/g"
}

if [ "${1:-}" = "--escape" ]; then
  sql_escape "${2:?value required}"
  exit $?
fi

: "${MYSQL_MIGRATION_PASSWORD:?MYSQL_MIGRATION_PASSWORD is required}"
: "${MYSQL_QUERY_PASSWORD:?MYSQL_QUERY_PASSWORD is required}"

migration_password=$(sql_escape "$MYSQL_MIGRATION_PASSWORD") || { echo "invalid migration password" >&2; exit 1; }
query_password=$(sql_escape "$MYSQL_QUERY_PASSWORD") || { echo "invalid query password" >&2; exit 1; }

mysql -uroot -p"${MYSQL_ROOT_PASSWORD}" <<SQL
CREATE USER IF NOT EXISTS 'trade_migrator'@'%' IDENTIFIED BY '${migration_password}';
CREATE USER IF NOT EXISTS 'trade_query'@'%' IDENTIFIED BY '${query_password}';
GRANT ALL PRIVILEGES ON foreign_trade_db.* TO 'trade_migrator'@'%';
REVOKE ALL PRIVILEGES, GRANT OPTION FROM 'trade_query'@'%';
GRANT SELECT ON foreign_trade_db.* TO 'trade_query'@'%';
FLUSH PRIVILEGES;
SQL
