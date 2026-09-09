import json
from dataclasses import replace
from pathlib import Path

import pytest

from trade_agent.evaluation.generator import EvaluationBundle
from trade_agent.evaluation.models import EvaluationCase, ReferenceEvidence, RunManifest, PerQueryResult
from trade_agent.evaluation.profiles import EvaluationArm, EvaluationBudget
from trade_agent.evaluation.runner import EvaluationRunner, LocalCorpusAdapter, StageOutcome


@pytest.fixture
def bundle():
    case = EvaluationCase.validated_fixture()
    reference = ReferenceEvidence(reference_evidence_id='ref-1', reference_evidence_set_id=case.reference_evidence_set_id,
                                  evidence_id='rag_' + 'a' * 64, required=True)
    return EvaluationBundle((case,), (reference,))


class Adapter:
    identity = {'backend': 'test-adapter', 'build_id': 'test-build', 'corpus_hash': 'a' * 64,
                'index_hash': 'b' * 64, 'model_id': 'test-model', 'prompt_id': 'test-prompt'}

    def __init__(self):
        self.calls = []

    def execute(self, component, query, hits, *, limit):
        self.calls.append((component, limit))
        assert isinstance(query, str)  # Labels/case answerability cannot enter the adapter.
        return StageOutcome(hits=({'evidence_id': 'rag_' + 'a' * 64},))


@pytest.mark.parametrize(('arm', 'calls'), [
    ('dense', ['dense']), ('bm25', ['bm25']),
    ('hybrid_rrf', ['dense', 'bm25']), ('wrrf', ['dense', 'bm25']),
    ('wrrf_filter', ['dense', 'bm25', 'filter']),
    ('full_rerank', ['dense', 'bm25', 'filter', 'reranker']),
    ('full_e2e', ['dense', 'bm25', 'filter', 'reranker', 'generation']),
])
def test_arms_isolate_components_and_write_recomputable_artifacts(tmp_path, bundle, arm, calls):
    adapter = Adapter()
    run = EvaluationRunner(adapter, budget=EvaluationBudget(candidate_limit=8, top_k=3)).run(bundle, EvaluationArm(arm), tmp_path)
    assert [name for name, _ in adapter.calls] == calls
    assert all(limit == 8 for name, limit in adapter.calls if name != 'generation')
    manifest = json.loads((run.path / 'manifest.json').read_text())
    RunManifest.model_validate_json(json.dumps(manifest['run']))
    assert manifest['budget'] == {'candidate_limit': 8, 'top_k': 3}
    row = json.loads((run.path / 'per_query.jsonl').read_text())
    PerQueryResult.model_validate_json(json.dumps(row['result']))
    assert row['metrics']['retrieval']['recall_at_10'] == 1.0
    assert row['trace']['candidate_counts']['selected'] == 1
    assert 'cold' in row['trace']['timings_ms']


def test_runs_are_append_only_and_share_build_query_budget(tmp_path, bundle):
    runner = EvaluationRunner(Adapter())
    runs = [runner.run(bundle, arm, tmp_path) for arm in (EvaluationArm.DENSE, EvaluationArm.BM25, EvaluationArm.DENSE)]
    assert len({run.path for run in runs}) == 3
    manifests = [json.loads((run.path / 'manifest.json').read_text()) for run in runs]
    assert manifests[0]['budget'] == manifests[1]['budget']
    assert manifests[0]['run']['snapshot']['index_hash'] == manifests[1]['run']['snapshot']['index_hash']
    assert manifests[0]['enabled_components'] == ['dense']
    assert manifests[1]['enabled_components'] == ['bm25']


def test_backend_failure_preserves_error_and_missing_metric(tmp_path, bundle):
    class Failing(Adapter):
        def execute(self, *args, **kwargs):
            raise RuntimeError('transport unavailable')
    run = EvaluationRunner(Failing()).run(bundle, EvaluationArm.DENSE, tmp_path)
    row = json.loads((run.path / 'per_query.jsonl').read_text())
    assert row['result']['status'] == 'failed'
    assert row['metrics']['retrieval'] is None
    assert 'transport unavailable' in row['errors'][0]
    assert row['trace']['backend_statuses']['dense'] == 'failed'


def test_development_runner_rejects_holdout_before_artifact_or_adapter(tmp_path, bundle):
    case = bundle.cases[0].model_copy(update={'dataset_role': 'holdout', 'visibility': 'private'})
    adapter = Adapter()
    with pytest.raises(ValueError, match='holdout'):
        EvaluationRunner(adapter).run(replace(bundle, cases=(case,)), EvaluationArm.DENSE, tmp_path)
    assert adapter.calls == []
    assert list(tmp_path.iterdir()) == []


def test_local_baselines_rank_actual_corpus_without_references():
    adapter = LocalCorpusAdapter(({'content': 'steel bolt fastener'}, {'content': 'cotton shirt garment'}, {'content': 'ceramic cup plate'}))
    for name in ('dense', 'bm25'):
        outcome = adapter.execute(name, 'steel bolt', (), limit=1)
        assert outcome.hits[0]['content'] == 'steel bolt fastener'
    assert adapter.execute('generation', 'steel bolt', (), limit=1).status == 'not_run'


@pytest.mark.parametrize('budget', [{'candidate_limit': 0}, {'top_k': 0}, {'candidate_limit': 2, 'top_k': 3}])
def test_invalid_budgets_rejected(budget):
    with pytest.raises(ValueError):
        EvaluationBudget(**budget)


def test_judge_does_not_override_deterministic_generation_metrics(tmp_path, bundle):
    from trade_agent.evaluation.judge import OptionalJudge
    class AnswerAdapter(Adapter):
        def execute(self, component, query, hits, *, limit):
            if component == 'generation':
                claim = {'claim_id': 'claim_' + 'b' * 64, 'status': 'supported',
                         'evidence_ids': ['rag_' + 'a' * 64], 'kind': 'fact'}
                return StageOutcome(answer={'claims': [claim]}, evidence=({'evidence_id': 'rag_' + 'a' * 64},),
                                    guard={'accepted': True, 'claims': [claim]})
            return super().execute(component, query, hits, limit=limit)
    judge = OptionalJudge(api_key='test', provider='test', model='test',
                          client=lambda **_: '{"faithfulness":0.0,"relevance":0.0}')
    run = EvaluationRunner(AnswerAdapter(), judge=judge).run(bundle, EvaluationArm.FULL_E2E, tmp_path)
    row = json.loads((run.path / 'per_query.jsonl').read_text())
    assert row['metrics']['generation']['faithfulness']['faithfulness'] == 1.0
    assert row['judge']['scores']['faithfulness'] == 0.0
    assert row['generation_outcome']['answer']['claims'][0]['claim_id'] == 'claim_' + 'b' * 64


def test_e2e_generation_failure_is_not_completed_and_retrieval_still_scored(tmp_path, bundle):
    class MissingGeneration(Adapter):
        def execute(self, component, query, hits, *, limit):
            if component == 'generation':
                return StageOutcome(status='not_run', errors=('provider missing',))
            return super().execute(component, query, hits, limit=limit)
    run = EvaluationRunner(MissingGeneration()).run(bundle, EvaluationArm.FULL_E2E, tmp_path)
    row = json.loads((run.path / 'per_query.jsonl').read_text())
    assert row['result']['status'] == 'failed'
    assert row['metrics']['retrieval']['recall_at_10'] == 1.0
    assert row['metrics']['generation'] is None


def test_failed_dense_branch_can_degrade_hybrid_without_losing_sparse_results(tmp_path, bundle):
    class Partial(Adapter):
        def execute(self, component, query, hits, *, limit):
            if component == 'dense':
                raise TimeoutError('dense deadline')
            return super().execute(component, query, hits, limit=limit)
    run = EvaluationRunner(Partial()).run(bundle, EvaluationArm.HYBRID_RRF, tmp_path)
    row = json.loads((run.path / 'per_query.jsonl').read_text())
    assert row['result']['status'] == 'completed'
    assert row['metrics']['retrieval']['recall_at_10'] == 1.0
    assert 'dense:failed' in row['trace']['degradation']


def test_failed_branch_cannot_contribute_stale_hits(tmp_path, bundle):
    class Stale(Adapter):
        def execute(self, component, query, hits, *, limit):
            if component == 'dense':
                return StageOutcome(hits=({'evidence_id': 'rag_' + 'a' * 64},), status='failed', errors=('stale cache',))
            return StageOutcome(hits=({'evidence_id': 'rag_' + 'c' * 64},))
    run = EvaluationRunner(Stale()).run(bundle, EvaluationArm.HYBRID_RRF, tmp_path)
    row = json.loads((run.path / 'per_query.jsonl').read_text())
    assert row['result']['retrieved_evidence_ids'] == ['rag_' + 'c' * 64]
    assert row['metrics']['retrieval']['recall_at_10'] == 0.0


def test_explicit_month_filter_removes_measured_development_001_noise():
    from trade_agent.evaluation.generator import indexed_corpus_content
    manifest = json.loads(Path('demo/trade_intel_seed/manifests/corpus_manifest.json').read_text())
    docs = tuple({'content': item['content'], 'metadata': json.loads(item['content'])}
                 for item in indexed_corpus_content(manifest))
    adapter = LocalCorpusAdapter(docs)
    query = 'What company is named in the 2026-06 profile for HS 090111?'
    hits = adapter.execute('filter', query, adapter.documents, limit=100).hits
    assert len(hits) == 1  # Baseline includes seven unrelated-month/undated sources.
    assert hits[0]['metadata']['calendar_month'] == '2026-06'
    assert hits[0]['metadata']['hs_code'] == '090111'
    assert hits[0]['metadata']['company'] == 'Harbor CN Imports 01'


def test_month_filter_preserves_explicit_comparisons_and_queries_without_month():
    docs = tuple({'content': month, 'metadata': {'calendar_month': month, 'hs_code': '090111'}}
                 for month in ('2026-06', '2026-07', '2025-06'))
    adapter = LocalCorpusAdapter(docs)
    hits = adapter.execute('filter', 'Compare 2026-06 and 2026-07 for HS 090111', adapter.documents, limit=100).hits
    assert {hit['metadata']['calendar_month'] for hit in hits} == {'2026-06', '2026-07'}
    assert adapter.execute('filter', 'Compare two dated records for HS 090111', adapter.documents, limit=100).hits == adapter.documents
