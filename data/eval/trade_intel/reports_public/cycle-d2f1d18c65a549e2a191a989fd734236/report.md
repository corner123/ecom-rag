# Trade development cycle

Synthetic development seed; local CPU hash cosine/BM25/lexical reranker, not production or neural retrieval. No generation provider: full_e2e fails and is rejected; generation/faithfulness and judge scores remain null; judge_not_run. Latency is a single local sequential pass, excludes build time, and is not a production benchmark. No holdout tuning.

| Arm | Recall before → after (Δ) | Precision before → after (Δ) | Mean latency ms before → after (Δ) | Regressions recall / precision | Accepted |
|---|---|---|---|---|---|
| bm25 | 0.458333 → 0.458333 (+0.000000) | 0.058333 → 0.058333 (+0.000000) | 0.166163 → 0.154808 (-0.011355) | 0 / 0 | False |
| dense | 0.402778 → 0.402778 (+0.000000) | 0.047222 → 0.047222 (+0.000000) | 0.123094 → 0.106579 (-0.016514) | 0 / 0 | False |
| full_e2e | 0.472222 → 0.472222 (+0.000000) | 0.075231 → 0.194907 (+0.119676) | 1.304295 → 0.997283 (-0.307012) | 0 / 0 | False |
| full_rerank | 0.472222 → 0.472222 (+0.000000) | 0.075231 → 0.194907 (+0.119676) | 1.300039 → 0.963170 (-0.336869) | 0 / 0 | True |
| hybrid_rrf | 0.458333 → 0.458333 (+0.000000) | 0.058333 → 0.058333 (+0.000000) | 0.365366 → 0.315018 (-0.050348) | 0 / 0 | False |
| wrrf | 0.458333 → 0.458333 (+0.000000) | 0.058333 → 0.058333 (+0.000000) | 0.367652 → 0.311382 (-0.056270) | 0 / 0 | False |
| wrrf_filter | 0.458333 → 0.458333 (+0.000000) | 0.072454 → 0.192130 (+0.119676) | 0.372302 → 0.324903 (-0.047399) | 0 / 0 | True |

Rule metrics are authoritative. See paired.json for all paired counts, refusal deltas, null faithfulness, case regressions, and execution rejections.

## Immutable run bundles

- [run-bm25-522d547ca7c44385b75356021b28d5a8](../report-53e872b4842b30d726c09a6e3c7be6d37cb83978c17f5f0ac6972677b0781e56-local-build-6cfcbd8614b5462c02ed7f5a4d33f12c-bm25-20260909T150554Z-b301cd67/report.md)
- [run-bm25-983b1969d4fe4e25a20a154ca9b76155](../report-34e36618228876eac51b8593e4092d184e09b6f51fb1338d36f99e5d6d2d8a85-local-build-a3137d12491fb54f1a058fefa47cc96e-bm25-20260909T150554Z-e50059f0/report.md)
- [run-dense-1d0d2b050356466f81a28ba8ec2d1618](../report-53e872b4842b30d726c09a6e3c7be6d37cb83978c17f5f0ac6972677b0781e56-local-build-6cfcbd8614b5462c02ed7f5a4d33f12c-dense-20260909T150554Z-124d2c0b/report.md)
- [run-dense-2655d14041f1417ea8f567ba1da41052](../report-34e36618228876eac51b8593e4092d184e09b6f51fb1338d36f99e5d6d2d8a85-local-build-a3137d12491fb54f1a058fefa47cc96e-dense-20260909T150554Z-f1591c35/report.md)
- [run-full_e2e-75c318493f64450288272ba35a1c7ea7](../report-53e872b4842b30d726c09a6e3c7be6d37cb83978c17f5f0ac6972677b0781e56-local-build-6cfcbd8614b5462c02ed7f5a4d33f12c-full_e2e-20260909T150554Z-855f0979/report.md)
- [run-full_e2e-906dcc92229f4229bdd21b41bdfcaf72](../report-34e36618228876eac51b8593e4092d184e09b6f51fb1338d36f99e5d6d2d8a85-local-build-a3137d12491fb54f1a058fefa47cc96e-full_e2e-20260909T150554Z-295565f0/report.md)
- [run-full_rerank-70a5b63ecbef4625a81f927df1e45b7d](../report-53e872b4842b30d726c09a6e3c7be6d37cb83978c17f5f0ac6972677b0781e56-local-build-6cfcbd8614b5462c02ed7f5a4d33f12c-full_rerank-20260909T150554Z-5c71ac4a/report.md)
- [run-full_rerank-b0ac3de2602149f09f5177079bfcac4a](../report-34e36618228876eac51b8593e4092d184e09b6f51fb1338d36f99e5d6d2d8a85-local-build-a3137d12491fb54f1a058fefa47cc96e-full_rerank-20260909T150554Z-e6258cfa/report.md)
- [run-hybrid_rrf-4dc6e8e26ab94bfe8da94bcadd2d1547](../report-34e36618228876eac51b8593e4092d184e09b6f51fb1338d36f99e5d6d2d8a85-local-build-a3137d12491fb54f1a058fefa47cc96e-hybrid_rrf-20260909T150554Z-d4cb1222/report.md)
- [run-hybrid_rrf-c52e5bbc58794f7e940bfde49b0782d5](../report-53e872b4842b30d726c09a6e3c7be6d37cb83978c17f5f0ac6972677b0781e56-local-build-6cfcbd8614b5462c02ed7f5a4d33f12c-hybrid_rrf-20260909T150554Z-d441734f/report.md)
- [run-wrrf-ddcead27c19f4a778d2001d6e3489289](../report-34e36618228876eac51b8593e4092d184e09b6f51fb1338d36f99e5d6d2d8a85-local-build-a3137d12491fb54f1a058fefa47cc96e-wrrf-20260909T150554Z-9b6c8058/report.md)
- [run-wrrf-f75f69fee0b74478a8fd30f14a50978f](../report-53e872b4842b30d726c09a6e3c7be6d37cb83978c17f5f0ac6972677b0781e56-local-build-6cfcbd8614b5462c02ed7f5a4d33f12c-wrrf-20260909T150554Z-bc1ba884/report.md)
- [run-wrrf_filter-50fff16ca50d4153855c7c293ac84bb2](../report-34e36618228876eac51b8593e4092d184e09b6f51fb1338d36f99e5d6d2d8a85-local-build-a3137d12491fb54f1a058fefa47cc96e-wrrf_filter-20260909T150554Z-7f93ca36/report.md)
- [run-wrrf_filter-9b25004ffd3544bcad35368b7a890951](../report-53e872b4842b30d726c09a6e3c7be6d37cb83978c17f5f0ac6972677b0781e56-local-build-6cfcbd8614b5462c02ed7f5a4d33f12c-wrrf_filter-20260909T150554Z-3be064b7/report.md)
