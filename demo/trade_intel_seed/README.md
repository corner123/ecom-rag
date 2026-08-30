# Synthetic trade-intelligence demo corpus

Every file in this directory is synthetic demonstration material only. All companies,
websites, contacts, transactions, reports, regulations, and social posts are fictional;
no private, customer, production, or real customs data is included.

Regenerate deterministically from the repository root:

```bash
python -m scripts.bootstrap_trade_intel_demo --output demo/trade_intel_seed --clean
```

The output marker is deliberately required before `--clean` can remove generated files.
The manifest uses relative POSIX paths and SHA-256 hashes. Website URLs use only reserved
`.example` domains. Customs profiles cover only fictional company IDs 1–3: every Task 3
record involving one of those companies is grouped by company, country, HS code, and calendar
month, with import/export/total amount and quantity aggregates. They are not a mirror of the
full synthetic trade ledger.
