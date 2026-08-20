# Engineering RAG response evaluation

- Baseline: `baseline-equal-legacy`
- Candidate: `candidate-split-query-aware-parent`
- Publishable: `false`

## Deterministic retrieval (answerable questions only)

| Metric | Baseline | Candidate | Delta | 95% paired bootstrap CI |
| --- | ---: | ---: | ---: | ---: |
| hit_at_k | 1.000 | 1.000 | 0.000 | [0.000, 0.000] |
| required_claim_recall_at_k | 0.854 | 0.910 | 0.056 | [0.000, 0.167] |
| mrr | 0.903 | 0.903 | 0.000 | [0.000, 0.000] |
| source_precision_at_k | 0.606 | 0.622 | 0.017 | [0.000, 0.050] |
| source_option_recall | 0.693 | 0.714 | 0.021 | [0.000, 0.062] |

## RAGAS response and context metrics

| Metric | Baseline | Candidate | Delta | 95% paired bootstrap CI |
| --- | ---: | ---: | ---: | ---: |
| context_precision | 0.837 | 0.782 | -0.027 | [-0.082, 0.000] |
| context_recall | 0.697 | 0.722 | 0.000 | [0.000, 0.000] |
| faithfulness | 0.920 | 0.858 | -0.042 | [-0.127, 0.027] |
| answer_relevancy | 0.703 | 0.854 | 0.170 | [0.029, 0.370] |
| answer_correctness | 0.477 | 0.506 | 0.016 | [-0.078, 0.121] |

## Safety and coverage

- `baseline-equal-legacy`: artifact complete=true, run execution coverage=1.000, judge execution coverage=1.000, judge score coverage=1.000, answerable response coverage=0.917, deterministic retrieval coverage=1.000, generation success=0.917, false refusal=0.083, refusal F1=0.889.
- `candidate-split-query-aware-parent`: artifact complete=true, run execution coverage=1.000, judge execution coverage=1.000, judge score coverage=0.917, answerable response coverage=1.000, deterministic retrieval coverage=1.000, generation success=1.000, false refusal=0.000, refusal F1=1.000.

## Generation token usage

- `baseline-equal-legacy`: attempted calls=11, usage reported=11 (coverage=1.000), prompt=20356, completion=7974, total=28330, cache hit=4864, cache miss=15492.
- `candidate-split-query-aware-parent`: attempted calls=12, usage reported=12 (coverage=1.000), prompt=23782, completion=9753, total=33535, cache hit=2688, cache miss=21094.

## Interpretation boundaries

- Deterministic retrieval metrics use reviewed evidence sources and include answerable questions only.
- source_option_recall is diagnostic coverage of alternative acceptable locators; required_claim_recall_at_k is the primary completeness metric.
- RAGAS scores are model-judged estimates, not absolute ground truth.
- Using a DeepSeek-family judge for DeepSeek-family answers can introduce self-evaluation bias.
- P99 is exploratory on small offline samples and is not a production SLA.
- A private holdout may be used only once after the candidate is frozen.
