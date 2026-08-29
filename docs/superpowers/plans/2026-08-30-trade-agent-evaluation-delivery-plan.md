# Trade Agent Evaluation Loop and Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construct leakage-controlled trade-intelligence datasets, calculate all required deterministic and optional semantic metrics, run retrieval ablations, analyze development failures, apply one evidence-backed optimization cycle, execute a one-time private holdout, and deliver immutable reports and a verified branch.

**Architecture:** Evaluation artifacts bind dataset, reference Evidence, corpus manifest, index/model/profile/code versions, and per-query traces. Development data drives changes; a locally ignored private holdout is generated from a withheld seed and consumed once after the candidate is frozen. Reports are append-only run bundles with checksums and explicit synthetic/non-production boundaries.

**Tech Stack:** Python 3.12, Pydantic v2, NumPy/Pandas, pytest, existing retrieval/workflow services, optional RAGAS/DeepSeek Judge, SHA-256 report manifests, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-08-29-foreign-trade-agent-overhaul-design.md`

## Global Constraints

- Development has at least 36 queries; private holdout has at least 15 queries and is Git-ignored.
- No question, reference claim, reference Evidence, or decision label enters the retrieval index or generation prompt.
- The holdout is consumed only after code/config/corpus/index/model/prompt/evaluator hashes are frozen.
- A failed holdout is reported as failed; the same holdout is never used for tuning.
- Rule metrics remain authoritative; a missing or failed LLM Judge is reported as `judge_not_run`/`null + error`, never zero.
- Existing ecommerce metrics and reports are not reused or cited as trade-agent results.

---

### Task 1: Define Evaluation Dataset, Reference, Snapshot, and Run Schemas

**Files:**
- Create: `trade_agent/evaluation/models.py`
- Create: `trade_agent/evaluation/hashing.py`
- Create: `data/eval/trade_intel/schemas/evaluation-v1.schema.json`
- Create: `tests/unit/test_evaluation_models.py`

**Interfaces:**
- Produces: `EvaluationCase`, `ReferenceEvidence`, `ReferenceClaim`, `BusinessDecision`, `EvaluationSnapshot`, `RunManifest`, `PerQueryResult`.
- Produces: `hash_file(path: Path) -> str` and `canonical_hash(model: BaseModel) -> str`.

- [ ] **Step 1: Write schema and separation tests**

```python
def test_case_references_labels_by_id_not_embedded_answer():
    case = EvaluationCase.validated_fixture()
    assert case.reference_evidence_set_id
    assert case.key_claim_ids
    assert not hasattr(case, "ground_truth_context")

def test_holdout_requires_private_role():
    with pytest.raises(ValidationError):
        EvaluationCase.validated_fixture(dataset_role="holdout", visibility="public")
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/unit/test_evaluation_models.py -q`

Expected: FAIL because evaluation models do not exist.

- [ ] **Step 3: Implement strict schemas and canonical serialization**

```python
class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    question: str
    task_type: TaskType
    answerable: bool
    dataset_role: Literal["development", "holdout"]
    visibility: Literal["public", "private"]
    as_of_date: date
    reference_evidence_set_id: str
    key_claim_ids: list[str]
    business_decision_id: str | None = None
    label_version: Literal["trade-intel-eval/v1"]
```

Run manifests include dataset/reference/corpus/index/profile/model/prompt/evaluator/code hashes and backend/degradation status.

- [ ] **Step 4: Run tests and validate JSON Schema round-trip**

Run: `docker compose run --rm api pytest tests/unit/test_evaluation_models.py -q`

Expected: PASS; Pydantic and exported JSON Schema accept/reject identical fixtures.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/evaluation/models.py trade_agent/evaluation/hashing.py data/eval/trade_intel/schemas tests/unit/test_evaluation_models.py
git commit -m "feat: define trade evaluation contracts"
```

### Task 2: Generate Public Development and Local Private Holdout Data

**Files:**
- Create: `trade_agent/evaluation/generator.py`
- Create: `trade_agent/evaluation/leakage.py`
- Create: `scripts/generate_trade_eval.py`
- Create: `scripts/validate_trade_eval.py`
- Create: `data/eval/trade_intel/dev_public.jsonl`
- Create: `data/eval/trade_intel/references_dev.jsonl`
- Create: `data/eval/trade_intel/README.md`
- Modify: `.gitignore`
- Create: `tests/unit/test_eval_generator.py`
- Create: `tests/unit/test_eval_leakage.py`

**Interfaces:**
- Produces: `generate_development(manifest, seed=20260830) -> EvaluationBundle`.
- Produces: `generate_private_holdout(manifest, secret_seed: int, output: Path) -> EvaluationBundle`.
- Produces: `LeakageAuditor.audit(dev, holdout, corpus) -> LeakageReport`.

- [ ] **Step 1: Write coverage and leakage tests**

```python
def test_development_covers_all_task_families(bundle):
    assert len(bundle.cases) >= 36
    assert REQUIRED_TASK_TYPES <= {case.task_type for case in bundle.cases}
    assert any(not case.answerable for case in bundle.cases)

def test_near_duplicate_holdout_question_is_detected(auditor):
    report = auditor.audit(dev_fixture(), near_duplicate_holdout(), corpus_fixture())
    assert report.passed is False
    assert report.near_duplicate_questions
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/unit/test_eval_generator.py tests/unit/test_eval_leakage.py -q`

Expected: FAIL because generator/auditor are absent.

- [ ] **Step 3: Implement controlled query/reference generation and leakage gates**

Development includes exact company/HS, semantic lead discovery, operating status, product/competitor, SQL aggregate, mixed-source, temporal conflict, duplicate-source, insufficient evidence, and unsafe/out-of-scope cases. The private generator uses disjoint entities/documents/events and a secret seed supplied at runtime.

Leakage checks normalized questions, exact/near chunk hashes, canonical URLs, source revisions, reference IDs, entity/event templates, and reference-label contamination in indexed content.

- [ ] **Step 4: Generate dev, generate private locally, and validate**

Run: `docker compose run --rm api python -m scripts.generate_trade_eval development --output data/eval/trade_intel`

Run: `umask 077 && mkdir -p data/eval/private && openssl rand -hex 16 > data/eval/private/holdout.seed`

Run: `docker compose run --rm api python -m scripts.generate_trade_eval holdout --seed-file data/eval/private/holdout.seed --output data/eval/private/trade_intel`

Run: `docker compose run --rm api python -m scripts.validate_trade_eval --dev data/eval/trade_intel --holdout data/eval/private/trade_intel`

Run: `docker compose run --rm api pytest tests/unit/test_eval_generator.py tests/unit/test_eval_leakage.py -q`

Expected: development/reference files are committed; private files remain ignored; leakage report passes with zero unapproved overlap.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/evaluation/generator.py trade_agent/evaluation/leakage.py scripts/generate_trade_eval.py scripts/validate_trade_eval.py data/eval/trade_intel .gitignore tests/unit/test_eval_generator.py tests/unit/test_eval_leakage.py
git commit -m "feat: generate leakage-controlled trade evaluations"
```

### Task 3: Implement Retrieval Metrics and Per-Query Traces

**Files:**
- Create: `trade_agent/evaluation/retrieval_metrics.py`
- Create: `tests/unit/test_retrieval_metrics.py`

**Interfaces:**
- Produces: `evaluate_retrieval(case, references, retrieval_outcome, k=10) -> RetrievalMetrics`.
- `RetrievalMetrics` contains Recall@10, Context Precision, Context Recall, reciprocal rank, evidence ranks, filter counts, and latency.

- [ ] **Step 1: Write hand-calculated metric tests**

```python
def test_recall_precision_and_context_recall_are_hand_calculated():
    metrics = evaluate_retrieval(
        case=case(required_evidence=["E1", "E2"], required_claims=["C1", "C2"]),
        references=refs(mapping={"C1": ["E1"], "C2": ["E2"]}),
        retrieval_outcome=outcome(["E1", "N1", "E2", "N2"]),
        k=10,
    )
    assert metrics.recall_at_10 == 1.0
    assert metrics.context_precision == 0.5
    assert metrics.context_recall == 1.0
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/unit/test_retrieval_metrics.py -q`

Expected: FAIL because metrics are absent.

- [ ] **Step 3: Implement deterministic metrics with zero-denominator policies**

Answerable cases require nonempty references. Unanswerable cases are evaluated with retrieval-noise/refusal metrics, not artificial perfect recall. Per-query results retain IDs and ranks so aggregates are independently recomputable.

- [ ] **Step 4: Run tests**

Run: `docker compose run --rm api pytest tests/unit/test_retrieval_metrics.py -q`

Expected: PASS for complete, partial, empty, duplicate, and unanswerable cases.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/evaluation/retrieval_metrics.py tests/unit/test_retrieval_metrics.py
git commit -m "feat: measure trade retrieval quality"
```

### Task 4: Implement Fusion, Business, and Generation Metrics

**Files:**
- Create: `trade_agent/evaluation/fusion_metrics.py`
- Create: `trade_agent/evaluation/generation_metrics.py`
- Create: `trade_agent/evaluation/business_metrics.py`
- Create: `tests/unit/test_fusion_metrics.py`
- Create: `tests/unit/test_generation_metrics.py`
- Create: `tests/unit/test_business_metrics.py`

**Interfaces:**
- Produces: `conflict_metrics(predictions, references) -> ClassificationMetrics`.
- Produces: `lead_precision(recommendations, labels, top_n) -> LeadMetrics`.
- Produces: `faithfulness(answer, evidence, guard) -> FaithfulnessMetrics`.
- Produces: `evidence_coverage(answer) -> float`.

- [ ] **Step 1: Write deterministic metric tests**

```python
def test_evidence_coverage_counts_factual_claims_only():
    answer = answer_with_claims([
        claim("采购额增加", factual=True, evidence_ids=["E1"]),
        claim("建议人工跟进", factual=False, status="analysis", evidence_ids=[]),
        claim("公司扩产", factual=True, evidence_ids=[]),
    ])
    assert evidence_coverage(answer) == 0.5

def test_synthetic_lead_precision_is_not_human_review_metric():
    result = lead_precision(recommendations(["A", "B"]), labels({"A": True, "B": False}), 2)
    assert result.precision == 0.5
    assert result.label_source == "synthetic_reference"
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/unit/test_fusion_metrics.py tests/unit/test_generation_metrics.py tests/unit/test_business_metrics.py -q`

Expected: FAIL because metrics are absent.

- [ ] **Step 3: Implement metrics and provenance labels**

Conflict reports include accuracy, macro-F1, confusion matrix, and escalation correctness. Faithfulness is supported factual claims divided by checked factual claims; invalid citations count unsupported. Evidence Coverage is bound factual claims divided by all factual claims. Lead Precision is labeled synthetic unless a separate human-review artifact exists.

- [ ] **Step 4: Run tests**

Run: `docker compose run --rm api pytest tests/unit/test_fusion_metrics.py tests/unit/test_generation_metrics.py tests/unit/test_business_metrics.py -q`

Expected: PASS for conflict/abstain, top-N lead, citation mutation, and nonfactual analysis cases.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/evaluation/fusion_metrics.py trade_agent/evaluation/generation_metrics.py trade_agent/evaluation/business_metrics.py tests/unit/test_fusion_metrics.py tests/unit/test_generation_metrics.py tests/unit/test_business_metrics.py
git commit -m "feat: measure fusion and grounded answers"
```

### Task 5: Build the Ablation Runner and Optional LLM Judge

**Files:**
- Create: `trade_agent/evaluation/runner.py`
- Create: `trade_agent/evaluation/profiles.py`
- Create: `trade_agent/evaluation/judge.py`
- Create: `scripts/run_trade_eval.py`
- Create: `tests/unit/test_eval_runner.py`
- Create: `tests/unit/test_judge.py`

**Interfaces:**
- Produces: `EvaluationRunner.run(bundle, arm: EvaluationArm, output: Path) -> EvaluationRun`.
- Arms: `dense`, `bm25`, `hybrid_rrf`, `wrrf`, `wrrf_filter`, `full_rerank`, `full_e2e`.
- Produces: `OptionalJudge.evaluate(...) -> JudgeOutcome(status, scores, errors, coverage)`.

- [ ] **Step 1: Write arm isolation and judge-failure tests**

```python
def test_dense_arm_cannot_use_sparse_or_reranker(runner_spy, bundle):
    runner_spy.run(bundle, EvaluationArm.DENSE, tmp_path())
    assert runner_spy.calls == ["dense"]

def test_missing_judge_key_is_not_zero_score(judge):
    outcome = judge.evaluate(answer_fixture())
    assert outcome.status == "judge_not_run"
    assert outcome.faithfulness is None
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/unit/test_eval_runner.py tests/unit/test_judge.py -q`

Expected: FAIL because runner/judge are absent.

- [ ] **Step 3: Implement fixed-budget arms and optional semantic residual evaluation**

Each arm records actual components, candidate counts, Top-K, backend, build/profile/model versions, cold/warm stage timings, and degradation. Judge uses temperature 0, strict JSON, prompt/model hashes, error preservation, and never changes deterministic metrics.

- [ ] **Step 4: Run tests and a two-case smoke**

Run: `docker compose run --rm api pytest tests/unit/test_eval_runner.py tests/unit/test_judge.py -q`

Run: `docker compose run --rm api python -m scripts.run_trade_eval --dataset data/eval/trade_intel/dev_public.jsonl --arms dense,bm25 --max-cases 2 --output /tmp/trade-eval-smoke`

Expected: both arm manifests show identical build/query budgets and different enabled components.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/evaluation/runner.py trade_agent/evaluation/profiles.py trade_agent/evaluation/judge.py scripts/run_trade_eval.py tests/unit/test_eval_runner.py tests/unit/test_judge.py
git commit -m "feat: run reproducible trade RAG ablations"
```

### Task 6: Implement Error Analysis and Development-Only Optimization Selection

**Files:**
- Create: `trade_agent/evaluation/error_analysis.py`
- Create: `trade_agent/evaluation/optimizer.py`
- Create: `scripts/analyze_trade_errors.py`
- Create: `tests/unit/test_error_analysis.py`
- Create: `tests/unit/test_optimizer.py`

**Interfaces:**
- Produces: `ErrorAnalyzer.analyze(runs: Sequence[EvaluationRun]) -> ErrorAnalysis`.
- Produces: `DevelopmentOptimizer.select(candidates, objective) -> OptimizationDecision`.

- [ ] **Step 1: Write failure-bucket and holdout-protection tests**

```python
def test_error_analyzer_assigns_actionable_bucket():
    analysis = ErrorAnalyzer().analyze([missed_hs_case()])
    assert analysis.cases[0].bucket == "keyword_miss"
    assert analysis.cases[0].suggested_component == "tokenizer_or_bm25"

def test_optimizer_refuses_holdout_input():
    with pytest.raises(HoldoutPolicyError):
        DevelopmentOptimizer().select([holdout_run()], objective_fixture())
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/unit/test_error_analysis.py tests/unit/test_optimizer.py -q`

Expected: FAIL because analyzer/optimizer are absent.

- [ ] **Step 3: Implement error taxonomy and preregistered objective**

Buckets are `missed_required_evidence`, `keyword_miss`, `filter_false_negative`, `filter_false_positive`, `stale_source`, `wrong_source_prior`, `dedupe_error`, `conflict_missed`, `reranker_regression`, `unsupported_claim`, `unsafe_decision`, and `backend_degraded`. The objective prioritizes Recall@10, then Context Precision/Faithfulness, subject to latency and refusal constraints.

- [ ] **Step 4: Run tests**

Run: `docker compose run --rm api pytest tests/unit/test_error_analysis.py tests/unit/test_optimizer.py -q`

Expected: PASS with paired-case deltas and holdout isolation.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/evaluation/error_analysis.py trade_agent/evaluation/optimizer.py scripts/analyze_trade_errors.py tests/unit/test_error_analysis.py tests/unit/test_optimizer.py
git commit -m "feat: turn development errors into retrieval changes"
```

### Task 7: Create Immutable Report Bundles and Verification

**Files:**
- Create: `trade_agent/evaluation/report.py`
- Create: `scripts/verify_trade_report.py`
- Create: `tests/contract/test_report_bundle.py`

**Interfaces:**
- Produces: `ReportWriter.write(run: EvaluationRun, root: Path) -> ReportBundle`.
- Produces: `verify_report_bundle(path: Path) -> VerificationResult`.

- [ ] **Step 1: Write no-overwrite and checksum tests**

```python
def test_report_writer_refuses_existing_run_directory(writer, run, tmp_path):
    writer.write(run, tmp_path)
    with pytest.raises(FileExistsError):
        writer.write(run, tmp_path)

def test_modified_report_fails_checksum(bundle):
    bundle.report_json.write_text("{}")
    assert verify_report_bundle(bundle.root).valid is False
```

- [ ] **Step 2: Verify failure**

Run: `docker compose run --rm api pytest tests/contract/test_report_bundle.py -q`

Expected: FAIL because report writer/verifier are absent.

- [ ] **Step 3: Implement append-only run IDs, manifests, JSON/Markdown, and checksums**

Run IDs include code SHA, build ID, profile, UTC timestamp, and random nonce. Bundle includes per-query JSONL, aggregates, environment/model/config manifests, error analysis, report JSON/Markdown, and `checksums.sha256`. Public Markdown never contains private holdout questions or labels.

- [ ] **Step 4: Run tests**

Run: `docker compose run --rm api pytest tests/contract/test_report_bundle.py -q`

Expected: PASS for valid, modified, missing, duplicate, and private-redaction bundles.

- [ ] **Step 5: Commit**

```bash
git add trade_agent/evaluation/report.py scripts/verify_trade_report.py tests/contract/test_report_bundle.py
git commit -m "feat: publish immutable evaluation evidence"
```

### Task 8: Execute the Development Baseline and One Optimization Cycle

**Files:**
- Create: `data/eval/trade_intel/reports_public/README.md`
- Create: immutable development run directory under `data/eval/trade_intel/reports_public/`
- Modify: only the component/config files identified by development error analysis

**Interfaces:**
- Consumes all evaluation arms and produces a paired before/after development decision.

- [ ] **Step 1: Freeze the development inputs**

Run: `docker compose run --rm api python -m scripts.validate_trade_eval --dev data/eval/trade_intel`

Run: `docker compose run --rm api python -m scripts.run_trade_eval --freeze-only --dataset data/eval/trade_intel/dev_public.jsonl --output data/eval/private/dev-snapshot`

Expected: snapshot hashes corpus, manifest, index, models, profiles, prompts, evaluator, and code.

- [ ] **Step 2: Run all retrieval and E2E development arms**

Run: `docker compose run --rm api python -m scripts.run_trade_eval --dataset data/eval/trade_intel/dev_public.jsonl --arms dense,bm25,hybrid_rrf,wrrf,wrrf_filter,full_rerank,full_e2e --output data/eval/private/dev-baseline`

Expected: every arm is complete, not degraded, and includes per-query traces.

- [ ] **Step 3: Analyze failures and select one bounded change set**

Run: `docker compose run --rm api python -m scripts.analyze_trade_errors --runs data/eval/private/dev-baseline --output data/eval/private/dev-analysis.json`

Expected: report identifies concrete dominant buckets and names only the components that may change.

- [ ] **Step 4: Use TDD to implement the selected changes and rebuild**

For every selected change, first add a regression test using the failing development pattern, verify it fails, implement the minimal query/chunk/filter/profile/rerank correction, run focused tests, create a new build ID, and commit with message `perf: improve development retrieval failures`.

- [ ] **Step 5: Rerun development and publish the paired delta**

Run: `docker compose run --rm api python -m scripts.run_trade_eval --dataset data/eval/trade_intel/dev_public.jsonl --arms dense,bm25,hybrid_rrf,wrrf,wrrf_filter,full_rerank,full_e2e --baseline data/eval/private/dev-baseline --output data/eval/private/dev-candidate --publish data/eval/trade_intel/reports_public`

Run: `docker compose run --rm api python -m scripts.verify_trade_report --latest data/eval/trade_intel/reports_public --kind development`

Expected: published report contains actual measured values, paired changes, regressions, latency costs, and synthetic qualification.

- [ ] **Step 6: Commit**

```bash
git add data/eval/trade_intel/reports_public trade_agent tests
git commit -m "perf: close the trade retrieval evaluation loop"
```

### Task 9: Freeze the Candidate and Consume the Private Holdout Once

**Files:**
- Create locally ignored: `data/eval/private/trade_intel/holdout_consumption.json`
- Create: `data/eval/trade_intel/holdout_snapshot.json`
- Create: immutable holdout run directory under `data/eval/trade_intel/reports_public/`

**Interfaces:**
- Produces one immutable holdout report and a local consumption lock keyed by holdout hash and candidate hash.

- [ ] **Step 1: Verify candidate and holdout preconditions**

Run: `docker compose run --rm api python -m scripts.validate_trade_eval --dev data/eval/trade_intel --holdout data/eval/private/trade_intel --require-zero-leakage`

Run: `docker compose run --rm api python -m scripts.run_trade_eval --preflight-holdout --dataset data/eval/private/trade_intel/holdout_private.jsonl --candidate data/eval/private/dev-candidate`

Expected: development thresholds pass, repository is clean at a committed SHA, no input mismatch exists, and no consumption lock exists for this holdout.

- [ ] **Step 2: Run the holdout exactly once**

Run: `docker compose run --rm api python -m scripts.run_trade_eval --consume-holdout --dataset data/eval/private/trade_intel/holdout_private.jsonl --output data/eval/private/holdout-run --publish data/eval/trade_intel/reports_public`

Expected: command atomically writes the consumption lock before evaluation, refuses a second invocation, and emits aggregate public results without private cases.

- [ ] **Step 3: Verify the immutable bundle and record the snapshot**

Run: `docker compose run --rm api python -m scripts.verify_trade_report --latest data/eval/trade_intel/reports_public --kind holdout`

Expected: checksums pass and snapshot matches candidate/corpus/index/model/profile/evaluator hashes.

- [ ] **Step 4: Commit without private inputs**

```bash
git status --short
git add data/eval/trade_intel/holdout_snapshot.json data/eval/trade_intel/reports_public
git commit -m "docs: publish first synthetic trade holdout"
```

### Task 10: Run Completion Audit, Document Truth Boundaries, and Deliver the Branch

**Files:**
- Replace: `README.md`
- Create: `docs/architecture.md`
- Create: `docs/data-contract.md`
- Create: `docs/evaluation.md`
- Create: `docs/operations.md`
- Create: `docs/completion-audit.md`
- Create: `scripts/verify_repository.py`
- Create: `tests/security/test_repository_hygiene.py`

**Interfaces:**
- Produces: `python -m scripts.verify_repository` as the final machine-readable requirement audit.

- [ ] **Step 1: Write repository hygiene and requirement-matrix tests**

```python
def test_repository_has_no_legacy_ecommerce_runtime(repo_root):
    forbidden = ["rag_core/engineering", "demo/ecommerce_seed", "engineering_api.py"]
    assert not [path for path in forbidden if (repo_root / path).exists()]

def test_readme_never_claims_production_or_real_customer_data(readme):
    forbidden = ["生产已部署", "真实客户数据", "线上转化提升"]
    assert not any(term in readme for term in forbidden)
```

- [ ] **Step 2: Verify failure before final docs/auditor**

Run: `docker compose run --rm api pytest tests/security/test_repository_hygiene.py -q`

Expected: FAIL until final docs and auditor exist.

- [ ] **Step 3: Implement documentation and requirement-by-requirement verifier**

The completion audit maps every user requirement and every spec acceptance item to exact code, test, runtime artifact, report, and status. Any missing or indirect evidence returns nonzero. README reports only current measured synthetic results and distinguishes development from first holdout.

- [ ] **Step 4: Run the complete verification stack**

Run: `docker compose up -d --wait`

Run: `docker compose exec api python -m pip check`

Run: `docker compose exec api pytest -q`

Run: `docker compose exec api python -m scripts.smoke_foundation`

Run: `docker compose exec api python -m scripts.smoke_milvus_roundtrip`

Run: `docker compose exec api python -m scripts.verify_repository`

Run: `git diff --check origin/main...HEAD`

Run: secret and absolute-host-path scan over tracked files.

Expected: all checks pass; completion audit has no missing requirement; private files and service volumes are untracked.

- [ ] **Step 5: Commit and attempt branch push**

```bash
git add README.md docs scripts/verify_repository.py tests/security/test_repository_hygiene.py
git commit -m "docs: deliver the synthetic trade intelligence agent"
git push -u origin codex/foreign-trade-agent-overhaul
```

After push, verify `git rev-parse HEAD` equals `git ls-remote origin refs/heads/codex/foreign-trade-agent-overhaul`; also verify `origin/main` still equals the recorded baseline unless the user changed it independently.
