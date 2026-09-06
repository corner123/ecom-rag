# Task 8 — Redis checkpoint persistence and resume

Implemented an AsyncRedisSaver factory backed by Redis Stack, namespaced TTL configuration, and strict `thread_id` plus `run_id` resume lookup.
Checkpoint writes remove answer/claim text and raw-row/draft fields while retaining evidence references, status, typed errors, and identifiers.
Graph compilation now supports explicit interruption so a new saver, graph, and evidence repository instance can continue without rerunning completed SQL/RAG nodes.
Compose now pins `redis/redis-stack-server:7.4.0-v8`; the runtime dependency is `langgraph-checkpoint-redis==0.5.2` as resolved by uv.

Verified: real Redis Stack integration tests (5 passed), containerized API test run (5 passed), graph plus Compose non-integration tests (52 passed, 1 deselected), compileall, `uv lock --check`, wheel build, and `git diff --check`.
The direct `docker compose up -d redis --wait` attempt could not bind 6379 because another worktree's confirmed plain-Redis container owned it; validation instead used the same pinned Stack digest on isolated port 6380 and an attached Compose network alias, without stopping that service.
