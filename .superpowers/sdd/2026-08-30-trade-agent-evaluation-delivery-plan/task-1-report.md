# Task 1 — Evaluation Contract Report

Base commit: `31427effc7a7e10e6a2bb49a85263dc11f7374a0`

## TDD evidence

- RED: `docker compose run --rm --no-deps -v "$PWD:/app" api pytest tests/unit/test_evaluation_models.py -q` (with non-secret dummy Compose interpolation values) failed during collection with `ModuleNotFoundError: No module named 'trade_agent.evaluation.hashing'`. The bind mount is required because the Compose API image copies source at build time and otherwise does not see newly created worktree files.
- GREEN: the same command passed `8` focused contract tests after implementing the models, hashing helpers, and generated schema.

## Verification evidence

- `python -m json.tool data/eval/trade_intel/schemas/evaluation-v1.schema.json` parsed the generated schema successfully.
- An independent ephemeral Draft 2020-12 `jsonschema` validator checked the exported schema, accepted a valid `EvaluationCase` JSON fixture, and rejected the same fixture with invalid `visibility="internal"`.
- The contracts are strict, frozen, and extra-forbidden. Cases contain only label IDs; private label records are separate. Holdouts require private visibility. Snapshots and manifests bind dataset, reference, corpus, index, profile, model, prompt, evaluator, and code hashes, plus backend and degradation status.

## Recovery verification

- Reviewed the recovered Task 1 artifacts against the brief and found that the generated JSON Schema did not express the Pydantic holdout/private cross-field validation. Added a focused red-green test and a Draft 2020-12 `if`/`then` condition to the `EvaluationCase` schema metadata, then regenerated `evaluation-v1.schema.json` from the model.
- `docker compose run --rm --no-deps -v "$PWD:/app" api pytest tests/unit/test_evaluation_models.py -q` passed `8` tests with non-secret dummy Compose interpolation values.
- `docker compose run --rm --no-deps -v "$PWD:/app" api python -m compileall -q trade_agent/evaluation` and `... api python -m json.tool data/eval/trade_intel/schemas/evaluation-v1.schema.json` both exited `0`.
- The API image intentionally does not include the undeclared `jsonschema` package, so the repository test asserts the emitted Draft 2020-12 condition structurally; the earlier independent validator evidence above remains applicable.
- Removed generated `trade_agent/evaluation/__pycache__` artifacts from the worktree before staging.
- Commit: `87233d0bb3e9c5448873757b6e9cdc93e4dd48bc` (`feat: define trade evaluation contracts`).

## Fix round 1 — schema/model equivalence

- Root cause: Pydantic-only validators rejected duplicate ID arrays, empty answerable claim references, whitespace-only labels, and redundant manifest/degradation fields, while the exported Draft 2020-12 schema had no equivalent constraints.
- Ruling: Draft 2020-12 cannot express equality between arbitrary top-level hash fields and nested `snapshot` fields, nor link an array of component names to dynamic object keys and values. `RunManifest` therefore keeps the nine frozen hashes only in `snapshot`; `backend_statuses` is the sole backend/degradation representation. This retains every required hash and status while removing contradictory duplicate state.
- RED: after adding the real `jsonschema` Draft 2020-12 validator and paired fixtures, `.venv/bin/python -m pytest tests/unit/test_evaluation_models.py -q` showed Pydantic rejected duplicate case claim IDs while the JSON Schema accepted them. The desired simplified manifest/result fixtures also failed until redundant fields were removed.
- GREEN: emitted schemas now use `uniqueItems` for every Pydantic-unique ID array, `if`/`then` conditions for holdout privacy and answerable nonempty claims, and a shared non-whitespace pattern for question/label text. The paired fixture test validates both Pydantic and Draft 2020-12 against valid and invalid cases.
- Verification: controller reran `.venv/bin/python -m pytest tests/unit/test_evaluation_models.py -q` (`8 passed`), `.venv/bin/python -m compileall -q trade_agent/evaluation`, `uv lock --check`, and `git diff --check 87233d0..HEAD`; all passed.
