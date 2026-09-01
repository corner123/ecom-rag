# Task 1 report — versioned BGE-M3 embeddings

Base: `c687a04`

## RED / GREEN

- RED: `tests/unit/test_embeddings.py` initially failed at collection because
  `trade_agent.index` did not exist. The new settings assertions then failed
  against the former `bge-small` defaults; the Compose cache contract failed
  because the API had no model-cache volume; and the smoke test failed because
  its module did not exist.
- GREEN: the focused suite passed `40` tests with `1` integration test
  deselected. It covers lazy batching, strict text and vector validation,
  explicit test-only fake injection, immutable SHA contracts, BGE-M3 1024-D
  enforcement, cache configuration, Compose cache isolation, and safe smoke
  output.

## Implementation evidence

- Default production embedding: `BAAI/bge-m3` at
  `5617a9f61b028005a4858fdac845db406aefb181`; expected and observed dense
  dimension: `1024`.
- Future reranker configuration is pinned to `BAAI/bge-reranker-v2-m3` at
  `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.
- `snapshot_download` excludes `onnx/**`, `imgs/**`, and `*.onnx*`; it retains
  the SentenceTransformer configuration/modules, tokenizer, and PyTorch
  weights. The downloaded model artifact set was `2,295,564,334` bytes
  (about `2.14 GiB`) and took about `13 minutes` over the available network.
- Host and Linux-container smokes ran offline after the initial download. The
  container used a read-only bind of the complete Hugging Face cache layout;
  no Compose named volume was created, changed, or removed.

## Real model smoke summary

```json
{"contract":{"dimension":1024,"dtype":"float32","library_version":"3.4.1","model_name":"BAAI/bge-m3","normalized":true,"provider":"sentence-transformers","requested_revision":"5617a9f61b028005a4858fdac845db406aefb181","resolved_revision":"5617a9f61b028005a4858fdac845db406aefb181"},"documents":{"finite":true,"normalized":true,"shape":[2,1024]},"query":{"finite":true,"normalized":true,"shape":[1024]},"snapshot_bytes":2295564334,"status":"ok"}
```

The host offline run completed in about `4.9s`; the Linux container run,
including model load and embedding, completed in about `10.4s`. Both use
Python `3.12.14`, `sentence-transformers 3.4.1`, `torch 2.6.0`, and
`transformers 4.48.3`.

No secrets or local absolute paths are recorded here.
