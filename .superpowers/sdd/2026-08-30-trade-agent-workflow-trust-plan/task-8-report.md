# Task 8 — Redis checkpoint persistence and resume

Implemented an AsyncRedisSaver factory backed by Redis Stack, namespaced TTL configuration, and strict `thread_id` plus `run_id` resume lookup.
Checkpoint writes remove answer/claim text and raw-row/draft fields while retaining evidence references, status, typed errors, and identifiers.
Graph compilation now supports explicit interruption so a new saver, graph, and evidence repository instance can continue without rerunning completed SQL/RAG nodes.
Compose now pins `redis/redis-stack-server:7.4.0-v8`; the runtime dependency is `langgraph-checkpoint-redis==0.5.2` as resolved by uv.

Verified: real Redis Stack integration tests (5 passed), containerized API test run (5 passed), graph plus Compose non-integration tests (52 passed, 1 deselected), compileall, `uv lock --check`, wheel build, and `git diff --check`.
The direct `docker compose up -d redis --wait` attempt could not bind 6379 because another worktree's confirmed plain-Redis container owned it; validation instead used the same pinned Stack digest on isolated port 6380 and an attached Compose network alias, without stopping that service.

## Fix round 1

Run identity is now enforced by the saver itself: every read/write derives `checkpoint_ns=run:<run_id>`, even when LangGraph resets the top-level namespace.
`checkpoint_config` requires thread, run, and idempotency IDs; the idempotency key is persisted as safe state, while completed nodes resume from the exact run checkpoint without being reissued.
`resume_run(thread_id, run_id)` is now the public two-argument interface; saver injection remains private and the factory registers the active saver.
Checkpoint state now uses a whitelist projection. Raw question/filter/input channels, answer/claim text, source payloads, secret-like values, and host-path sentinels are excluded; structured intent omits its question and resume input remains caller-supplied config.
Fix-round verification: Redis Stack integration 7 passed; graph/Compose checks 52 passed (8 deselected); compileall, lock check, diff check, and sensitive scan passed.
The saver injects the config idempotency key into every checkpoint and rejects any conflicting state/write key before persistence.

## Fix round 2

Checkpoint reads now compare the caller configuration idempotency key with the persisted safe key before LangGraph receives state; mismatch fails as `checkpoint_unavailable` before any SQL, RAG, or generator call.
The file evidence repository now stores the original request as a content-addressed `RequestRef` outside Redis. The checkpoint holds only this reference and a question-free structured intent; a fresh graph/repository resolves it after resume.
The whitelist retains only `branch:to:*` scheduling markers required by LangGraph pending work, alongside reviewed safe state, so answer/guard/finalizer can complete without replaying completed retrieval nodes.
Fix-round verification: Redis Stack integration 8 passed; graph/Compose checks 52 passed (1 deselected); compileall, lock check, diff check, and sensitive scan passed.

## Fix round 3

Fresh resumes that are paused before `query_rewrite` now resolve the current request from a content-addressed request reference and retain a separate original request reference for scope comparison. Rewrites advance only the current reference, never a raw query field in Redis.
Router output persists the validated `RetrievalFilter` projection required to reparse a rewrite. The checkpoint whitelist validates that projection before writing it, so caller-supplied raw filters remain excluded.
The Redis Stack regression discards the original configuration, recreates the saver, graph, and evidence repository at `query_rewrite`, and confirms no completed retrieval call is replayed while raw question, secret, host-path, and evidence-content sentinels remain absent from Redis.
Fix-round verification: Redis Stack integration 9 passed; graph/Compose checks 52 passed (1 deselected); compileall, lock check, diff check, and sensitive scan passed.
