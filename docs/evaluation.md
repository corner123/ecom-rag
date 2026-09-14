# Evaluation protocol and measured evidence

## Scope

The committed trade evaluation is a **synthetic local retrieval evaluation**. Its adapter uses SHA-256 token cosine, BM25 Okapi, explicit metadata filters, weighted RRF, and a lexical reranker on local CPU. It does not use the production Milvus collection, BGE-M3 embeddings, or the BGE reranker. No online answer generator ran.

The repository also contains older ecommerce evaluation artifacts in separate legacy data/report paths. They are not loaded by the trade evaluator and are not cited as trade-agent evidence.

## Dataset control

The public development set contains 44 queries, exceeding the 36-query minimum. Its cases and reference records are separate files. The private holdout also contains 44 cases according to the immutable aggregate, exceeding the 15-query minimum; its questions and labels remain ignored.

Development is reusable for diagnosis and tuning. Holdout access follows this order:

1. validate public/private separation and index contamination;
2. freeze code, config, corpus, index, model, profile, prompt, evaluator, dataset, and reference hashes;
3. reproduce the selected development candidate without regressions;
4. atomically publish the private consumption lock before execution;
5. run the holdout once and publish only redacted aggregates;
6. preserve a failed result as failed and never tune against the same holdout.

The committed `holdout_snapshot.json` binds the first-run bundle and its frozen identities. The ignored local `holdout_consumption.json` proves that the one-shot gate has been consumed. Do not run the consume command again. A later final evaluation requires a new private partition and a new freeze.

## Metrics

Deterministic rule metrics are authoritative:

- retrieval: Recall@10, context precision, context recall, and reciprocal rank;
- fusion: conflict accuracy, macro-F1, escalation behavior, and source independence;
- generation: Evidence coverage, locator validity, deterministic faithfulness checks, and refusal behavior when generation executes;
- business: synthetic lead precision and decision-label metrics;
- operational diagnostics: stage status, degradation, and local latency distributions.

The optional LLM Judge is additive. A missing Judge is `judge_not_run`; an attempted failure is `judge_failed`. Scores stay null and errors remain recorded. Neither condition is converted to zero or allowed to override deterministic rule metrics.

## Development result

The immutable paired cycle is `cycle-d2f1d18c65a549e2a191a989fd734236`. All seven arms used the same 44 development queries and candidate budgets. For `full_rerank`:

| Metric | Baseline | Candidate | Paired delta |
| --- | ---: | ---: | ---: |
| Recall@10 | 0.472222 | 0.472222 | 0.000000 |
| Context precision | 0.075231 | 0.194907 | 0.119676 |
| Scored retrieval cases | 36 | 36 | n/a |

There were zero paired recall and precision regressions. The supported change was exact calendar-month filtering. Eight unanswerable/unsafe cases have null recall-style scores and do not become artificial zeros.

`full_e2e` failed all 44 development executions because generation was unavailable. Its retrieval fields remain diagnostics; it was rejected for the execution constraint. Faithfulness and business-decision quality remain unmeasured.

## First holdout result

The redacted first-run bundle reports:

| Metric | Value | Scored / total |
| --- | ---: | ---: |
| Recall@10 | 0.5833333333333334 | 36 / 44 |
| Context precision | 0.20601851851851852 | 36 / 44 |
| Context recall | 0.6111111111111112 | 36 / 44 |
| Reciprocal rank | 0.5729166666666666 | 36 / 44 |
| Completed retrieval cases | 44 | 44 / 44 |

Generation is `not_run`. Judge status is `judge_not_run`, coverage is `0.0`, faithfulness/relevance are null, and 44 rows are categorized as not judged. Passing the preregistered local retrieval floors does not imply answer safety, customer validity, live data coverage, production deployment, real Milvus/BGE acceptance, or business success.

## Verification

These read-only commands recalculate checksums and paired/holdout invariants:

```sh
.venv/bin/python -m scripts.verify_trade_report --latest data/eval/trade_intel/reports_public --kind development
.venv/bin/python -m scripts.verify_trade_report --latest data/eval/trade_intel/reports_public --kind holdout
.venv/bin/python -m scripts.validate_trade_eval --dev data/eval/trade_intel --holdout data/eval/private/trade_intel --require-zero-leakage
```

Use a new ignored output directory for any development rerun. Never overwrite a published report directory or the published first-run holdout snapshot.
