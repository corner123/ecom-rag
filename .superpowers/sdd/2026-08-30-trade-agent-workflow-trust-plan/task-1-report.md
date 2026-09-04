# Task 1 — Live Schema Registry report

## Delivered

- Added strict `QueryConstraints` for metrics, dimensions, and filters.
- Added `SchemaRegistry.refresh(connection)` backed only by live MySQL
  `INFORMATION_SCHEMA` reads and `SchemaRegistry.link(constraints)` for
  registered business fields.
- Added a packaged reviewed YAML contract for all seven tables, including
  column description/type/null/default, primary and unique keys, eleven
  foreign-key joins, role-aware aliases/dimensions/filters, aggregation unit
  and currency contracts, sensitive addresses, and a 50-row maximum.
- Refresh fingerprints live physical columns, keys, and joins. Any reviewed
  contract mismatch raises `SchemaDriftError` with its observed fingerprint and
  clears the cached snapshot.
- Added unit and live integration coverage. The live drift test uses the
  migration role to add one randomized owned column and drops that exact column
  in `finally`; baseline discovery is always done through the SELECT-only role.

## RED / GREEN evidence

RED:

```text
uv run pytest tests/unit/test_schema_registry.py -q
ERROR ... ModuleNotFoundError: No module named 'trade_agent.db.contracts'
```

Decimal parsing regression RED:

```text
uv run pytest tests/unit/test_schema_registry.py -q
FAILED ... expected 'decimal(18,3)', got 'decimal(18'
```

Final GREEN:

```text
python3 .superpowers/local-runtime/run.py --migration uv run pytest tests/unit/test_schema_registry.py tests/integration/test_live_schema_registry.py -q
6 passed in 0.25s
```

Packaging verification:

```text
uv build --wheel --out-dir <temporary directory>
unzip -l <wheel> | rg 'trade_agent/config/schema_registry.yaml'
trade_agent/config/schema_registry.yaml
```

## Self-review

- Discovery reads only metadata and never business rows or credentials.
- Query credentials remain runtime-only; the live baseline uses the existing
  query role, while the test-only DDL uses the migration role.
- Fail-closed comparison covers table/column sets, type, nullability, default,
  primary/unique key, and foreign-key join contracts.
- YAML decimal types are quoted so precision and scale survive YAML flow-map
  parsing.
- `pyproject.toml` packages the YAML for installed runtime loading.

## Concerns

The registry intentionally records business roles such as importer/exporter
and import/export country as reviewed names. The later SQL planner must assign
SQL table aliases when compiling those roles; this task deliberately does not
implement planning or execution.
