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
`.example` domains. Customs profiles are a declared narrow deterministic subset of Task 3
monthly company/HS tuples, not a mirror of the full synthetic trade ledger.
