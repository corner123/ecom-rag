# Task 1 report — versioned BGE-M3 embeddings

Base: `c687a04`

## RED / GREEN

- Initial RED: `tests/unit/test_embeddings.py` failed at collection because
  `trade_agent.index` did not exist. Settings, Compose cache, and smoke tests
  then failed against the former unpinned implementation.
- Review RED: regression tests reproduced the unsafe test-provider path,
  forgeable production contracts, snapshot-directory-only provenance, public
  fake smoke success, and unsafe CLI output. New filesystem tests exercised
  missing/non-regular artifacts, symlink escape, wrong size, wrong SHA256,
  cache extras, and official metadata mismatches before the fixes were made.
- Re-review RED: a pinned-looking direct contract and Pydantic's non-validating
  `model_copy(update=...)` could still pass a provider-only production check;
  the package manifest also derived its trust value from its own current JSON.
  Regression tests covered direct construction, dump/validate round trips,
  every public-field mutation, unknown extra fields, altered canonical JSON,
  duplicate JSON keys, wrong schema/model/revision, unsafe paths, duplicate
  paths, invalid field types, and model-construction ordering.
- Final focused GREEN: `98 passed` for embedding contracts, artifact security,
  and smoke CLI behavior.
- Host non-model regression GREEN: `332 passed, 7 deselected`.

## Production identity and artifact provenance

- Production is only `BAAI/bge-m3` at immutable revision
  `5617a9f61b028005a4858fdac845db406aefb181`, dense dimension `1024`, normalized
  `float32`, provider `sentence-transformers`.
- The committed dense-runtime allowlist contains exactly ten required
  SentenceTransformer/config/tokenizer/PyTorch files. It excludes README and
  image assets, ONNX duplicates, `colbert_linear.pt`, and `sparse_linear.pt`.
- Each allowlisted file records byte size and SHA256. Large LFS files also
  record and verify the official Hugging Face LFS SHA256; small files record
  official Git blob IDs and SHA256 measured from the trusted pinned snapshot.
- The package-data manifest is authenticated against a hard-coded canonical
  JSON SHA256 both at module import and immediately before any production
  snapshot download or model construction. Only after that digest matches are
  schema version, exact model/revision/count, field types, lineage structure,
  unique safe relative POSIX paths, sizes, and hashes accepted. The trust
  anchor is never derived from the currently installed manifest.
- Online initialization checks `HfApi.model_info(..., files_metadata=True)`
  against the pinned revision and manifest. Every online or offline load then
  streams all ten local files through size and SHA256 verification before
  `SentenceTransformer` is constructed. Missing/non-regular files and symlinks
  escaping the repository blob store fail closed.
- Canonical manifest SHA256:
  `3a862f1d0a8543acc13e9faa5e6d6d1f916ee609b264960be8337e6ba509856b`.
  Verified dense runtime bytes: `2,293,315,801`. Extra cache files remain
  allowed but are never trusted, hashed into the manifest, or counted.
- No model weight is committed to Git.

## Test-provider and smoke boundaries

- `test_mode=True` and `test_encoder` are strictly paired and require the
  process environment value `TEST_EMBEDDING_PROVIDER=deterministic`.
- Test contracts use provider/model `deterministic-test`, zero revisions and
  manifest identity, and cannot satisfy `require_production()`.
- Production construction rejects every model/revision override.
- A production-looking set of serialized fields is only portable identity
  metadata. Live production authority is a process-local random-key HMAC over
  the complete validated public schema, issued by an internal factory after
  model bytes verify. `is_production` and `require_production()` revalidate the
  schema and compare this attestation in constant time. Direct construction,
  dump/validate round trips, changed/extra `model_copy` fields, and fake
  provider replacement all fail; an unchanged copy retains the attestation.
- Task 2 may persist the public identity fields for collection compatibility,
  but an index write must receive a live manager contract that passes
  `require_production()` in the current process.
- The public smoke calls `require_production()`; a deterministic encoder can
  no longer return `status=ok`.
- Invalid CLI arguments return exit `2`; runtime/serialization failures return
  exit `1`. Failure stdout is empty and stderr is exactly one safe JSON line
  without exception messages, arguments, secrets, or local paths.

## Real-model verification

- Host cached/offline model test: `1 passed in 5.02s` on the final implementation.
- Host cached/offline public smoke: production contract, finite normalized
  document shape `[2,1024]`, query shape `[1024]`, status `ok`.
- Official Hugging Face metadata verification: passed against the pinned
  revision and all ten manifest records.
- Rebuilt Linux image: `trade-agent-embedding-smoke:latest`.
- Linux public smoke ran with `--network none` and a read-only Hugging Face hub
  cache mounted at `/model-cache`: status `ok`, exact production contract,
  finite normalized vectors, and the same trusted byte count.
- Linux cached/offline model test: `1 passed in 13.30s` on the rebuilt image.
- Built wheel contains the committed artifact manifest as package data.

The host and image use `sentence-transformers 3.4.1`, `torch 2.6.0`, and
`transformers 4.48.3`. No secrets or local absolute paths are recorded here.
