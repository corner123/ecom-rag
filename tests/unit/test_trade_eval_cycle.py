import json
from pathlib import Path

import pytest

from scripts.run_trade_eval import main
from trade_agent.evaluation.report import verify_report_bundle


DATA = Path('data/eval/trade_intel/dev_public.jsonl')


def test_freeze_only_writes_hashes_without_query_execution(tmp_path, monkeypatch):
    from trade_agent.evaluation.runner import LocalCorpusAdapter
    def forbidden(*args, **kwargs):
        pytest.fail('freeze must not execute retrieval or generation')
    monkeypatch.setattr(LocalCorpusAdapter, 'execute', forbidden)
    assert main(['--dataset', str(DATA), '--freeze-only', '--output', str(tmp_path)]) == 0
    freeze = json.loads((tmp_path / 'snapshot.json').read_text())
    assert len(freeze['corpus_manifest_hash']) == 64
    assert len(freeze['snapshots']) == 7
    for snapshot in freeze['snapshots'].values():
        for name in ('dataset', 'reference', 'corpus', 'index', 'profile', 'model', 'prompt', 'evaluator', 'code'):
            assert len(snapshot[name + '_hash']) == 64
    assert not list(tmp_path.rglob('per_query.jsonl'))
    with pytest.raises(FileExistsError):
        main(['--dataset', str(DATA), '--freeze-only', '--output', str(tmp_path)])


def test_cli_publishes_paired_all_arm_evidence_and_rejects_missing_generation(tmp_path):
    baseline, candidate, public = (tmp_path / name for name in ('baseline', 'candidate', 'public'))
    assert main(['--dataset', str(DATA), '--output', str(baseline)]) == 0
    assert main(['--dataset', str(DATA), '--baseline', str(baseline), '--output', str(candidate),
                 '--publish', str(public)]) == 0
    from trade_agent.evaluation.cycle import verify_cycle_bundle
    assert verify_cycle_bundle(next(public.glob('cycle-*'))).valid
    cycle = json.loads(next(public.glob('cycle-*/paired.json')).read_text())
    assert len(cycle['arms']) == 7
    assert cycle['arms']['bm25']['paired_deltas']['recall_at_10']['delta'] == 0
    assert cycle['arms']['full_e2e']['candidate_status_counts'] == {'failed': 44}
    assert cycle['arms']['full_e2e']['decision']['accepted_change'] is False
    assert cycle['arms']['full_e2e']['decision']['rejected_candidates']
    assert len(list(public.glob('report-*/report.json'))) == 14
    for path in public.glob('report-*'):
        assert verify_report_bundle(path).valid
    assert 'synthetic' in (public / 'README.md').read_text().lower()


def test_publish_requires_baseline_before_running(tmp_path):
    with pytest.raises(SystemExit):
        main(['--dataset', str(DATA), '--output', str(tmp_path / 'runs'), '--publish', str(tmp_path / 'public')])
    assert not (tmp_path / 'runs').exists()


def test_cycle_verifier_rejects_tampered_paired_evidence(tmp_path):
    from trade_agent.evaluation import cycle
    assert hasattr(cycle, 'verify_cycle_bundle'), 'cycle checksum and linked evidence verification is required'
    baseline, candidate, public = (tmp_path / name for name in ('baseline', 'candidate', 'public'))
    main(['--dataset', str(DATA), '--arms', 'bm25', '--output', str(baseline)])
    main(['--dataset', str(DATA), '--arms', 'bm25', '--baseline', str(baseline),
          '--output', str(candidate), '--publish', str(public)])
    path = next(public.glob('cycle-*'))
    assert cycle.verify_cycle_bundle(path).valid
    paired = path / 'paired.json'
    paired.write_text(paired.read_text() + ' ')
    assert not cycle.verify_cycle_bundle(path).valid


def test_paired_cycle_rejects_changed_reference_hash(tmp_path):
    from trade_agent.evaluation.cycle import compare_runs
    from trade_agent.evaluation.error_analysis import load_evaluation_runs
    from dataclasses import replace
    baseline, candidate = (tmp_path / name for name in ('baseline', 'candidate'))
    for path in (baseline, candidate):
        main(['--dataset', str(DATA), '--arms', 'bm25', '--output', str(path)])
    before, after = load_evaluation_runs((baseline,)), load_evaluation_runs((candidate,))
    manifest_path = after[0].path / 'manifest.json'
    metadata = json.loads(manifest_path.read_text())
    metadata['run']['snapshot']['reference_hash'] = 'f' * 64
    manifest_path.write_text(json.dumps(metadata))
    from trade_agent.evaluation.models import RunManifest
    after = (replace(after[0], manifest=RunManifest.model_validate(metadata['run'])),)
    with pytest.raises(ValueError, match='reference_hash'):
        compare_runs(before, after)


@pytest.mark.parametrize(('section', 'field', 'value', 'error'), [
    ('snapshot', 'index_hash', 'f' * 64, 'index_hash'),
    ('snapshot', 'model_hash', 'f' * 64, 'model_hash'),
    ('snapshot', 'profile_hash', 'f' * 64, 'profile_hash'),
    ('budget', 'candidate_limit', 101, 'budget'),
    ('budget', 'top_k', 9, 'budget'),
    ('profile', 'enabled_components', ['dense', 'bm25'], 'profile'),
    ('profile', 'dense_weight', 0.5, 'profile'),
    ('profile', 'sparse_weight', 0.5, 'profile'),
    ('profile', 'rrf_k', 61, 'profile'),
    ('profile', 'version', 'trade-eval-arms/v2', 'profile'),
])
@pytest.mark.parametrize('operation', ['compare', 'publish'])
def test_paired_cycle_rejects_changed_retrieval_inputs_before_publication(
        tmp_path, section, field, value, error, operation):
    from trade_agent.evaluation.cycle import compare_runs, publish_cycle
    from trade_agent.evaluation.error_analysis import load_evaluation_runs
    baseline, candidate, public = (tmp_path / name for name in ('baseline', 'candidate', 'public'))
    for path in (baseline, candidate):
        main(['--dataset', str(DATA), '--arms', 'bm25', '--output', str(path)])
    before, after = load_evaluation_runs((baseline,)), load_evaluation_runs((candidate,))
    cycle = compare_runs(before, after)
    manifest_path = after[0].path / 'manifest.json'
    metadata = json.loads(manifest_path.read_text())
    target = metadata['run']['snapshot'] if section == 'snapshot' else metadata[section]
    target[field] = value
    manifest_path.write_text(json.dumps(metadata))
    after = load_evaluation_runs((candidate,))
    with pytest.raises(ValueError, match=error):
        if operation == 'compare':
            compare_runs(before, after)
        else:
            publish_cycle(cycle, (*before, *after), public)
    assert not public.exists()
