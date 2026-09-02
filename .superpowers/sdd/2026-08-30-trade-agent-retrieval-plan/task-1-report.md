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
- Third review RED: the internal contract factory still accepted ordinary
  identity fields, so an in-process caller could sign a production-looking
  contract without proving which files were hashed. Runtime verification also
  discarded the freshly loaded manifest and continued to read rebindable
  module-level artifact dictionaries. Regression tests reproduced direct,
  replaced, copied, pickled, reconstructed, and field-mutated receipts; mutable
  allowlist rebinding with attacker bytes; deep manifest mutation; and failure
  to thread one manifest object through metadata, download, and file checks.
- Final adversarial RED: cache extras were correctly excluded from the
  manifest but the full cache snapshot was still passed to Transformers, which
  can prefer an unverified `model.safetensors` or adapter/index configuration.
  A strict-signature regression also showed the contract factory still
  accepted four caller-supplied output fields instead of only the receipt.
- Integrity-window RED: hard links and live file-descriptor links fixed file
  identity but not inode contents. A loader-construction regression overwrote
  a verified cache blob in place and proved the former pre-load-only hash still
  allowed manager activation. The new test requires post-load rejection,
  runtime-view cleanup, and no receipt, encoder, or contract state.
- Final focused GREEN: `97 passed` for embedding contracts, artifact security,
  and smoke CLI behavior.
- Host non-model regression GREEN: `331 passed, 7 deselected`.

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
- The loader materializes frozen `TrustedManifest` and `TrustedArtifact`
  values. Each encoder initialization keeps that one locally attested manifest
  object and explicitly passes it to official metadata verification,
  `snapshot_download(allow_patterns=...)`, and snapshot-content verification.
  Compatibility observation constants are never consulted for a security
  decision, so rebinding them cannot authorize alternate bytes.
- After snapshot verification, the manager creates a private temporary runtime
  view containing exactly the ten trusted paths. Same-filesystem files are
  hard-linked; cross-filesystem fallbacks use live file-descriptor links so an
  atomic cache-path replacement cannot retarget the view. The view is hashed
  before model construction and fully re-hashed immediately after construction;
  only the post-load check can issue the receipt and activate manager state.
  The view remains alive through model use. Cache extras such as
  README/images/ColBERT/sparse/ONNX remain allowed in the cache but are
  invisible to SentenceTransformer, including alternate safetensors, indexes,
  and adapter configuration. Constructor or post-load verification failures
  clean the view immediately.
- Online initialization checks `HfApi.model_info(..., files_metadata=True)`
  against the pinned revision and manifest. Every online or offline load then
  streams all ten local files through size and SHA256 verification before
  `SentenceTransformer` is constructed and streams the exact runtime view again
  before issuing authority. Missing/non-regular files, persistent in-place
  changes during construction, and symlinks escaping the repository blob store
  fail closed.
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
  metadata. Snapshot verification hashes the actual required paths, sizes, and
  SHA256 values, builds an exact-file loader view, and repeats the full view
  check after model construction before it registers a process-local,
  full-field HMAC receipt.
  Directly constructed, replaced, copied, deep-copied, pickled, reconstructed,
  or mutated receipts cannot be consumed. The internal contract factory has the
  exact `(*, receipt)` signature; it derives the model/revision/manifest/count,
  installed library version, and fixed 1024/normalized/float32 invariants
  internally. A separate random-key
  HMAC covers the complete contract schema. `is_production` and
  `require_production()` revalidate and compare it in constant time. Direct
  contract construction, dump/validate round trips, changed/extra `model_copy`
  fields, and fake provider replacement fail; an unchanged copy retains its
  live contract attestation.
- Task 2 may persist the public identity fields for collection compatibility,
  but an index write must receive a live manager contract that passes
  `require_production()` in the current process.
- The pre/post hashes are an integrity and production-authority boundary, not a
  model-deserialization sandbox. They reject a mutation that remains visible at
  the post-load check but cannot undo loader side effects or detect malicious
  same-process code that changes and restores bytes entirely between checks.
  `trust_remote_code=False` remains mandatory; a stronger same-process threat
  model requires sealed immutable backing plus a restricted loader process.
- The public smoke calls `require_production()`; a deterministic encoder can
  no longer return `status=ok`.
- Invalid CLI arguments return exit `2`; runtime/serialization failures return
  exit `1`. Failure stdout is empty and stderr is exactly one safe JSON line
  without exception messages, arguments, secrets, or local paths.

## Real-model verification

- Host cached/offline model test: `1 passed in 15.02s` on the final implementation.
- Host cached/offline public smoke: production contract, finite normalized
  document shape `[2,1024]`, query shape `[1024]`, status `ok`.
- Official Hugging Face metadata verification: passed against the pinned
  revision and all ten manifest records.
- Rebuilt Linux image: `trade-agent-embedding-smoke:latest`.
- Linux public smoke ran with `--network none` and a read-only Hugging Face hub
  cache mounted at `/model-cache`: status `ok`, exact production contract,
  finite normalized vectors, and the same trusted byte count.
- Linux cached/offline model test: `1 passed in 35.79s` on the rebuilt image.
- Built wheel contains the committed artifact manifest plus the contract,
  embedding-manager, and provenance modules.
- Independent final provenance threat review: PASS; seven directed checks
  passed in `6.70s`.

The host and image use `sentence-transformers 3.4.1`, `torch 2.6.0`, and
`transformers 4.48.3`. No secrets or local absolute paths are recorded here.
