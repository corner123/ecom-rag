# Repository maintenance instructions

These instructions apply to the whole repository.

## Explain engineering work before acting

- Treat debugging and engineering decisions as teaching opportunities. Do not
  silently fix an error, change an environment, stop a process, alter an index,
  install a dependency, or make an architectural choice and only report the
  finished result.
- Before a state-changing engineering action, explain in commentary:
  1. the observed symptom and the likely cause;
  2. the read-only checks or commands used to confirm the cause;
  3. how to interpret the important output;
  4. the exact action about to be taken, its scope, and relevant risks or
     alternatives; and
  5. how the result will be verified.
- After the action, report what changed, the decisive evidence, and a reusable
  manual procedure so the user can repeat the diagnosis and fix independently.
  Redact secrets and sensitive values while preserving the useful method.
- For process or port conflicts, always teach and follow this sequence: inspect
  the listening socket, map it to a PID, verify the executable and command line,
  stop only the confirmed process, and then verify that the port is free. Never
  kill processes blindly by name or port.
- If an immediate containment action is required to prevent data loss or a
  security incident, take only the minimum safe action first, then explain the
  evidence and procedure immediately afterward.

- Keep this RAG repository independent from Mini-Nanobot. Integration remains
  a read-only HTTP boundary; do not merge the two codebases.
- After an authorized code, test, data-contract, or documentation change,
  run the relevant tests and release checks, commit the completed change, and
  push it to the configured `origin` remote unless the user explicitly asks
  not to push.
- Never commit `.env`, API keys, tokens, local vector indexes, raw fetched
  corpora, build manifests, model weights, logs, databases, or cache files.
- Before every push, verify the staged diff, run a secret/host-path scan, and
  confirm that the remote commit matches the local commit after pushing.
- Keep formal engineering evaluation deterministic. Do not silently include
  online answer generation in frozen retrieval metrics.
- Do not overwrite the published first-run holdout report. A new final
  evaluation requires a new private holdout, snapshot, build ID, and report.
- Treat the published `dirty=true` snapshot as a historical local run. Do not
  claim exact cross-machine reproducibility until the corresponding
  Mini-Nanobot worktree changes exist in its own remote history.
