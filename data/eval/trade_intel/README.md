# Trade-intelligence development evaluation data

`dev_public.jsonl` contains product-facing development cases generated from
the frozen synthetic trade-intelligence corpus. Cases intentionally contain a
question, task route, time boundary, and opaque reference IDs only. They do
not contain answer text, source excerpts, or reference payloads.

`references_dev.jsonl` keeps the matching evidence and claim labels in a
separate artifact. It is needed to score development runs, but must never be
inserted into a retrieval index or exposed through product-facing evaluation
records.

Regenerate the development partition deterministically with:

```sh
.venv/bin/python -m scripts.generate_trade_eval development --output data/eval/trade_intel
```

The private holdout and its secret seed live under `data/eval/private/`, which
is ignored by Git. Generate it locally using a secret random seed, then gate
the split before use:

```sh
.venv/bin/python -m scripts.generate_trade_eval holdout --seed-file data/eval/private/holdout.seed --output data/eval/private/trade_intel
.venv/bin/python -m scripts.validate_trade_eval --dev data/eval/trade_intel --holdout data/eval/private/trade_intel
```

The validator rejects normalized or near-duplicate questions, exact and
near chunk identities, canonical URLs, source revisions, reference IDs,
entity/event templates, and indexed reference-label contamination.
