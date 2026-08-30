# Task 5 implementation report

## Scope delivered

- Added deterministic synthetic-only corpus generation in `trade_agent.data.demo_generator`.
- Added package initializers, the `python -m scripts.bootstrap_trade_intel_demo` CLI,
  source catalog, checked-in corpus README/ownership marker, and generated artifacts.
- Generated six and only six source types: official website, B2B, industry news,
  social, regulator, and customs profile.
- Included 12 website HTML sections plus Markdown, 18 B2B products, 16 news stories
  (four controlled syndicated mirrors), 12 JSONL social posts, two text-bearing PDFs,
  one image-only scanned PDF, and 12 narrow deterministic monthly company/HS customs
  profiles derived from `TradeSeedBundle` records.
- Manifest records use stable relative POSIX paths and include hashes, entity, fact
  type, fixed timestamps, claim IDs, source/file type, locators, and synthetic status.
- Hardened output cleanup: broad roots and nonempty unowned directories are refused;
  only a marked demo root can be cleaned, and its checked-in README is preserved.
- Copied scripts/data/demo into the API image and provided process-local non-secret
  Compose defaults for the four demo bootstrap variables.

## TDD evidence

RED:

```text
uv run pytest tests/unit/test_demo_corpus.py -q
ModuleNotFoundError: No module named 'trade_agent.data'
```

GREEN:

```text
uv run pytest tests/unit/test_demo_corpus.py -q
3 passed in 0.15s
```

Safety regression RED:

```text
test_clean_requires_a_owned_marker_and_rejects_broad_roots FAILED
Failed: DID NOT RAISE <class 'ValueError'>
```

Safety regression GREEN is included in the focused green run above.

## Verification evidence

- Two independently generated temporary corpora had byte-identical relative file hash
  maps; all generated manifest hashes rehashed successfully.
- Checked-in manifest rehash: `32 records`.
- `pypdf` extraction from the scanned fixture was empty, proving image-only content.
- `uv run pytest tests/unit -q`: `33 passed in 0.21s`.
- `docker compose config --quiet`: exit 0.
- `docker compose build api`: exit 0.
- In-image CLI generated `/tmp/trade-intel-image-demo`; in-image focused test:
  `3 passed in 0.57s`.
- `git diff --check`: exit 0.
- Targeted source scans found no non-example URLs, host paths, or credential-like
  assignments in the Task 5 corpus implementation/artifacts.
- `docker ps -aq` was empty after verification.

## Residual risks

The image-only scan fixture intentionally has no OCR text. Later PDF routing must either
OCR it or quarantine it explicitly, as specified by the next foundation task.
