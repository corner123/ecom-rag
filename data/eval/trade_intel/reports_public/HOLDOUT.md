# First synthetic trade holdout

The one-shot private holdout passed the preregistered **local retrieval** gate. It does not establish production readiness, generated-answer quality, business-decision quality, or answer safety.

- [Immutable aggregate report](report-7cd261cf288521093c50d1083afec460ea2bb2c624cbea16e46abdfe8d70daf8-local-build-a3137d12491fb54f1a058fefa47cc96e-full_rerank-20260910T214832Z-7bb1aa43/report.md)
- [Frozen inputs and outcome](../holdout_snapshot.json)

The development-selected `full_rerank` candidate was frozen at clean Git commit `a42e56293ea66b8bee62e5bbb984226c52c0d7e4`. Holdout support was committed first; a fresh ignored development run then reproduced all 44 development cases and every deterministic retrieval field from the immutable Task 8 selected candidate. Config, corpus, index, model, and prompt hashes matched; source/evaluator hashes were refreshed to include holdout support. The existing public development cycle remains unchanged.

Before consumption, the frozen gate required Recall@10 >= 0.4722222222222222, context precision >= 0.19490740740740742, and completed execution without degradation. The private lock was published atomically and durably before any holdout query executed. A failed execution or process crash retains that lock and cannot be retried. A future evaluation requires a new private holdout and separately frozen evidence; this holdout must never inform tuning.

| Aggregate | Result | Scored / total |
|---|---:|---:|
| Recall@10 | 0.5833333333333334 | 36 / 44 |
| Context precision | 0.20601851851851852 | 36 / 44 |
| Context recall | 0.6111111111111112 | 36 / 44 |
| Reciprocal rank | 0.5729166666666666 | 36 / 44 |
| Completed retrieval execution | 44 | 44 / 44 |

Unanswerable cases have null recall-style scores and are excluded from their scored denominators. This is synthetic seed data evaluated with local hash-token cosine, BM25, and a lexical reranker. No online generation ran. Judge status is `judge_not_run`, faithfulness/relevance scores are null, and error count is 44; public error categories preserve that absence without private raw errors. The deterministic rule metrics are authoritative. No ecommerce evaluation result is reused.

The committed bundle contains aggregates and frozen identities only. Private inputs, labels, execution rows, preflight state, and the consumption lock remain Git-ignored locally. The snapshot binds the holdout hash, candidate hash, clean Git SHA, corpus manifest, dataset/reference hashes, corpus/index/model/prompt/profile/evaluator/code/config hashes, development selection evidence, and public report checksums.

Verification is read-only and may be repeated:

```sh
.venv/bin/python -m scripts.verify_trade_report --latest data/eval/trade_intel/reports_public --kind holdout
```

The original run used `.venv/bin/python`; Docker stayed closed. Do not repeat the consumption command.
