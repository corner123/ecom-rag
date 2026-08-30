# Task 5 implementation report

## Scope delivered

- Added deterministic synthetic-only corpus generation in `trade_agent.data.demo_generator`.
- Added package initializers, the `python -m scripts.bootstrap_trade_intel_demo` CLI,
  source catalog, checked-in corpus README/ownership marker, and generated artifacts.
- Generated six and only six source types: official website, B2B, industry news,
  social, regulator, and customs profile.
- Included 12 website HTML sections plus Markdown, 18 B2B products, 16 news stories
  (four controlled syndicated mirrors), 12 JSONL social posts, two text-bearing PDFs,
  one image-only scanned PDF, and 54 narrow deterministic monthly company/country/HS
  customs profiles derived from `TradeSeedBundle` aggregates.
- Manifest records use stable relative POSIX paths and include hashes, entity, fact
  type, fixed timestamps, claim IDs, source/file type, locators, and synthetic status.
- Hardened output cleanup: broad roots and nonempty unowned directories are refused;
  only a marked demo root can be cleaned, and its checked-in README is preserved.
- Copied scripts/data/demo into the API image. Compose retains required secret
  interpolations; verification injects non-secret demo values only into each process.

## TDD evidence

RED:

```text
uv run pytest tests/unit/test_demo_corpus.py -q
ModuleNotFoundError: No module named 'trade_agent.data'
```

Initial GREEN (historical, before review-round regressions were added):

```text
uv run pytest tests/unit/test_demo_corpus.py -q
3 passed in 0.15s
```

Safety regression RED:

```text
test_clean_requires_a_owned_marker_and_rejects_broad_roots FAILED
Failed: DID NOT RAISE <class 'ValueError'>
```

Final safety regression GREEN is recorded in the final verification evidence below.

## Verification evidence

- Two independently generated temporary corpora had byte-identical relative file hash
  maps; all generated manifest hashes rehashed successfully.
- Checked-in manifest rehash: `74 records`.
- `pypdf` extraction from the scanned fixture was empty, proving image-only content.
- `uv run pytest tests/unit -q`: `38 passed in 1.51s`.
- `docker compose config --quiet`: exit 0.
- `docker compose build api`: exit 0.
- In-image CLI generated `/tmp/trade-intel-image-demo`; final in-image unit suite:
  `38 passed in 2.15s`.
- `git diff --check`: exit 0.
- Targeted source scans found no non-example URLs, host paths, or credential-like
  assignments in the Task 5 corpus implementation/artifacts.
- `docker ps -aq` was empty after verification.

## Residual risks

The image-only scan fixture intentionally has no OCR text. Later PDF routing must either
OCR it or quarantine it explicitly, as specified by the next foundation task.

## Review round 1 hardening

- Output handling now rejects a symlink root, marker, directory, or file anywhere in an
  existing owned tree before cleaning or writing. Generated relative paths are validated
  and each write repeats the tree check. Directory- and file-symlink regression probes
  both preserved an outside sentinel on regeneration and clean attempts.
- Customs profiles now aggregate every Task 3 record involving the declared fictional
  company-ID subset `{1, 2, 3}` by company/country/HS/calendar month. Each profile carries
  import/export/total USD and kg sums, roles, window/grain, source count, and compact raw-ID
  summary/hash; the corpus has 54 profiles, well below 825 ledger facts.
- The image-only scan is a deterministic 1200×1550 Pillow raster with the exact visible ASCII
  synthetic warning, fictional regulator notice, and table. A 150-DPI Poppler render was
  visually inspected: 1275×1650 pixels, 138501 black pixels, 256 colors; pypdf extraction
  remains empty and the embedded image is nontrivial.
- The catalog has explicit, non-overlapping paths for regulator PDFs and the industry-news
  bulletin. Controlled syndication now makes NEWS-013..016 content-equivalent mirrors of
  NEWS-001..004 with provenance, canonical IDs, dedupe clusters, and item-level claims.

Additional RED/GREEN evidence:

```text
RED symlink file regression: Failed: DID NOT RAISE <class 'ValueError'>
GREEN final focused corpus suite: 8 passed in 1.48s
GREEN host units: 38 passed in 1.51s; in-image units: 38 passed in 2.15s
```

The API image initially failed because the unpinned latest `uv` (0.12.7) produced an editable
requirements hash mismatch. Pinning the Docker build tool to `uv==0.9.21`, which matches the
project export/install workflow, rebuilt the image successfully.
