# Public trade development evidence

Synthetic, non-production local evaluation. Each cycle and run is immutable.

- [Paired development cycle](cycle-d2f1d18c65a549e2a191a989fd734236/report.md) ([machine-readable deltas](cycle-d2f1d18c65a549e2a191a989fd734236/paired.json))

## Development cycle — explicit calendar-month filtering

The only retrieval change is `LocalCorpusAdapter` metadata filtering of explicit valid `YYYY-MM` query months against source `calendar_month`, in addition to the existing HS predicate. Explicit multiple months use OR; questions without months keep the existing behavior. Query/reference labels were not indexed, and reference claims/Evidence/decision labels were not passed to a generation prompt. Only public development questions informed this change; the ignored private holdout was used for leakage validation, never retrieval, scoring, or tuning.

Baseline all-arm analysis measured 132 `filter_false_positive`, 80 `missed_required_evidence`, 61 `wrong_source_prior`, and 3 `keyword_miss` rows (buckets count arm/case observations, not unique questions). Development case 001 requested 2026-06 and HS 090111 but returned eight sources, including seven unrelated-month/undated sources. A failing regression first reproduced eight hits where one was required. The added month predicate returns the single correct source. Post-change all-arm analysis has 120 filter false positives; the other bucket counts are unchanged. These heuristic buckets do not imply a source-authority diagnosis for every noise row.

All seven arms ran over the same 44 development questions; recall and precision have 36 paired scoreable cases, while latency/status/refusal have 44. Eight unanswerable/unsafe cases have no positive reference evidence and are not silently counted as zero recall. Every retrieval-only arm completed all 44 queries with no degraded backend. `full_e2e` failed all 44 queries in both runs because local generation is unavailable (`generation:not_run`); the optimizer rejects it with `execution_constraint`. Its reported retrieval metrics describe retrieval only. Generation faithfulness, answer coverage, business decisions, and LLM Judge quality are unmeasured; the Judge is `judge_not_run` with null scores and recorded errors, never zero.

The accepted development candidate is `full_rerank`:

| Metric | Baseline full_rerank | Candidate full_rerank | Paired change |
|---|---:|---:|---:|
| Recall@10 | 0.472222 | 0.472222 | +0.000000 |
| Context precision | 0.075231 | 0.194907 | +0.119676 |
| Mean local query latency (ms) | 1.300039 | 0.963170 | -0.336869 |
| Refusal rate | 0.000000 | 0.000000 | +0.000000 |

There are zero paired recall/precision regressions across all arms. The month-filtered `wrrf_filter` precision also improves by +0.119676; dense, BM25, hybrid RRF, and WRRF retrieval metrics are unchanged. The cycle JSON records every paired metric/count, regression case list, same-arm decision, and rejection. `accepted_change=false` for unchanged arms means no metric gain; it does not mean their retrieval execution failed. Compared with candidate BM25, candidate full_rerank gains +0.013889 recall and +0.136574 precision, at +0.808362ms mean local query latency; the configured optimization limit allows at most +50ms and no refusal-rate increase. Missing faithfulness has zero scored pairs and does not override rule metrics.

These are **synthetic, non-production** seed measurements using local hash-token cosine, BM25, and a lexical reranker. They are not trained semantic/neural retrieval or live trade intelligence. Timings are one sequential local pass, include retrieval stages but exclude index construction, label first stage use cold and later uses warm without resetting shared caches, and have no confidence interval; the apparent speedup is not a production performance claim. Recall below 0.5 and 120 remaining filter-noise observations remain material limitations. Refusal metrics here describe retrieval-only execution statuses, not a validated answer-safety capability. No ecommerce metric/report is reused.

The baseline build is `local-build-6cfcbd8614b5462c02ed7f5a4d33f12c`; the candidate build is `local-build-a3137d12491fb54f1a058fefa47cc96e`. The corpus/index/model/prompt/evaluator/data/reference hashes are unchanged; code/build differ to record the filter and CLI/report-cycle work. Candidate code is committed as `e971ce5` (`perf: improve development retrieval failures`). Run `code_sha` is a SHA-256 digest of source contents, not a Git commit ID. The baseline was a historical working-tree run while CLI support was being implemented; its exact source digest is retained, but this publication does not claim exact baseline checkout or cross-machine latency reproduction.

To repeat the diagnostic procedure, use a fresh ignored output directory for each run and freeze before running. Do not overwrite this published cycle or reuse the same run IDs for publication:

```sh
.venv/bin/python -m scripts.validate_trade_eval --dev data/eval/trade_intel --holdout data/eval/private/trade_intel
.venv/bin/python -m scripts.run_trade_eval --freeze-only --dataset data/eval/trade_intel/dev_public.jsonl --output data/eval/private/dev-snapshot-next
.venv/bin/python -m scripts.run_trade_eval --dataset data/eval/trade_intel/dev_public.jsonl --arms dense,bm25,hybrid_rrf,wrrf,wrrf_filter,full_rerank,full_e2e --output data/eval/private/dev-baseline-next
.venv/bin/python -m scripts.analyze_trade_errors --runs data/eval/private/dev-baseline-next --output data/eval/private/dev-analysis-next.json
# After exactly one development-supported change and its failing/passing regression test:
.venv/bin/python -m scripts.run_trade_eval --dataset data/eval/trade_intel/dev_public.jsonl --arms dense,bm25,hybrid_rrf,wrrf,wrrf_filter,full_rerank,full_e2e --baseline data/eval/private/dev-baseline-next --output data/eval/private/dev-candidate-next --publish data/eval/trade_intel/reports_public
.venv/bin/python -m scripts.verify_trade_report --latest data/eval/trade_intel/reports_public --kind development
```

The verifier checks the latest development run plus every cycle checksum, every linked run bundle, recomputed paired deltas/decisions, and Markdown consistency. All 14 run bundles and the paired cycle are immutable. The root README is an index and commentary, outside bundle checksums.
