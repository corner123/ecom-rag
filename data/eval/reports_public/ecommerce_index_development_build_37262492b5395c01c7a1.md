# Engineering RAG evaluation

- Dataset: `ecommerce_development.jsonl`
- Questions: 12
- Top-K: 5
- Baseline: `bm25`
- Suite: `index`

| Strategy | Route acc. | Route macro-F1 | Primary Hit@K | Primary Recall@K | nDCG@K | Symbol Recall@K | MRR | Refusal F1 | False refusal | P50 ms | P95 ms | P99 ms* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | N/A | N/A | 0.583 | 0.583 | 0.378 | 0.929 | 0.333 | 0.000 | 0.000 | 0.56 | 1.28 | 1.38 |
| dense | N/A | N/A | 0.417 | 0.417 | 0.278 | 0.929 | 0.190 | 0.000 | 0.000 | 68.91 | 80.59 | 85.48 |
| hybrid | N/A | N/A | 0.583 | 0.583 | 0.361 | 0.929 | 0.278 | 0.000 | 0.000 | 82.23 | 110.09 | 114.02 |

## Interpretation boundaries

- Source metrics use explicit v2 primary/supporting labels and de-duplicate repeated chunks from the same source.
- Index ablation uses oracle routes, answerable questions only, and disables live rg/AST/Git plus answer generation.
- End-to-end evaluation fixes the hybrid index and evaluates routing, live verification, citations, and refusal separately.
- Symbol Recall@K uses explicit `relevant_symbols`; inspect the eligible sample count.
- Refusal is an explicit predictor decision. An empty retrieval result is not silently counted as a safe refusal.
- File/citation metrics are source-level, not passage entailment or answer correctness metrics.
- Latency is measured after one discarded warm-up call per strategy; model/index loading and report serialization are excluded.
- `P99 ms*` is exploratory for these small offline datasets; it is not a statistically stable production tail-latency SLA.
