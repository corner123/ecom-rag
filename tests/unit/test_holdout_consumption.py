import importlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts.run_trade_eval import main
from trade_agent.evaluation.generator import generate_private_holdout

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / 'demo/trade_intel_seed/manifests/corpus_manifest.json'
DEV = ROOT / 'data/eval/trade_intel'


def api():
    from trade_agent.evaluation import runner
    assert hasattr(runner.EvaluationRunner, '_run_frozen'), 'frozen execution path is missing'
    return importlib.import_module('trade_agent.evaluation.holdout')


@pytest.fixture
def prepared(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / '.gitignore').write_text('private/\n')
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.gitignore'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Test', '-c', 'user.email=test@example.test',
                    'commit', '-qm', 'fixture'], check=True)
    private = tmp_path / 'private'
    generate_private_holdout(json.loads(CORPUS.read_text()), 194725, private / 'trade_intel')
    candidate = private / 'dev-candidate'
    main(['--dataset', str(DEV / 'dev_public.jsonl'), '--arms', 'full_rerank', '--output', str(candidate)])
    return dict(dataset=private / 'trade_intel/holdout_private.jsonl', candidate=candidate,
                corpus_manifest=CORPUS, development=DEV, repo=tmp_path)


def test_preflight_freezes_without_execution_and_rejects_dirty(prepared, monkeypatch):
    h = api()
    from trade_agent.evaluation.runner import LocalCorpusAdapter
    monkeypatch.setattr(LocalCorpusAdapter, 'execute', lambda *a, **kw: pytest.fail('preflight executed a query'))
    frozen = h.preflight(**prepared)
    assert frozen['arm'] == 'full_rerank'
    assert len(frozen['candidate_hash']) == 64
    assert len(frozen['git_sha']) == 40
    assert not prepared['dataset'].with_name('holdout_consumption.json').exists()
    (prepared['repo'] / 'dirty.txt').write_text('dirty')
    with pytest.raises(ValueError, match='clean'):
        h.preflight(**prepared)


def test_preflight_rejects_stale_candidate(prepared):
    h = api()
    path = next(prepared['candidate'].glob('*/manifest.json'))
    value = json.loads(path.read_text())
    value['run']['snapshot']['code_hash'] = 'f' * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='code_hash'):
        h.preflight(**prepared)


def test_consume_locks_before_execution_and_refuses_retry_even_after_failure(prepared, monkeypatch):
    h = api()
    frozen = h.preflight(**prepared)
    from trade_agent.evaluation.runner import EvaluationRunner
    def broken(*a, **kw):
        assert prepared['dataset'].with_name('holdout_consumption.json').is_file()
        raise RuntimeError('simulated failure')
    monkeypatch.setattr(EvaluationRunner, '_run_frozen', broken)
    kwargs = dict(**prepared, output=prepared['repo'] / 'private/run', publish=prepared['repo'] / 'public', frozen=frozen)
    with pytest.raises(RuntimeError, match='simulated'):
        h.consume(**kwargs)
    with pytest.raises(FileExistsError, match='consum'):
        h.consume(**kwargs)


def test_consumption_public_projection_and_snapshot_verification(prepared):
    h = api()
    frozen = h.preflight(**prepared)
    report = h.consume(**prepared, output=prepared['repo'] / 'private/run',
                       publish=prepared['repo'] / 'public', frozen=frozen)
    assert h.verify_snapshot(report).valid
    assert not (report / 'per_query.jsonl').exists()
    aggregate = json.loads((report / 'aggregate.json').read_text())
    assert aggregate['case_count'] >= 15
    assert aggregate['judge']['status'] == 'judge_not_run'
    assert all(v is None for v in aggregate['judge']['scores'].values())
    assert aggregate['judge']['errors'] == []
    assert aggregate['judge']['error_count'] > 0
    for row in prepared['dataset'].read_text().splitlines():
        case = json.loads(row)
        assert case['question'] not in ''.join(p.read_text() for p in report.iterdir())
    snapshot = report.parent.parent / 'holdout_snapshot.json'
    value = json.loads(snapshot.read_text())
    value['candidate_hash'] = 'e' * 64
    snapshot.write_text(json.dumps(value))
    assert not h.verify_snapshot(report).valid


def test_validate_cli_accepts_explicit_zero_leakage(prepared):
    from scripts.validate_trade_eval import main as validate
    assert validate(['--dev', str(DEV), '--holdout', str(prepared['dataset'].parent),
                     '--require-zero-leakage']) == 0


def test_preflight_rejects_development_regression(prepared):
    h = api()
    path = next(prepared['candidate'].glob('*/aggregate.json'))
    value = json.loads(path.read_text())
    value['retrieval']['recall_at_10']['value'] = 0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='aggregate|threshold|regression'):
        h.preflight(**prepared)


def test_consume_revalidates_frozen_inputs_before_lock(prepared):
    h = api()
    frozen = h.preflight(**prepared)
    frozen['candidate_hash'] = '0' * 64
    with pytest.raises(ValueError, match='frozen'):
        h.consume(**prepared, output=prepared['repo'] / 'private/run',
                  publish=prepared['repo'] / 'public', frozen=frozen)
    assert not prepared['dataset'].with_name('holdout_consumption.json').exists()


def test_preflight_rejects_unignored_private_inputs(prepared):
    h = api()
    path = prepared['repo'] / 'private/trade_intel/holdout_private.jsonl'
    subprocess.run(['git', '-C', str(prepared['repo']), 'add', '-f', str(path)], check=True)
    subprocess.run(['git', '-C', str(prepared['repo']), '-c', 'user.name=Test', '-c', 'user.email=test@example.test',
                    'commit', '-qm', 'tracked private fixture'], check=True)
    with pytest.raises(ValueError, match='ignored'):
        h.preflight(**prepared)


def test_atomic_lock_has_one_winner_and_complete_payload(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    h = api()
    lock = tmp_path / 'consumption.json'
    def attempt(index):
        try:
            h._atomic_new(lock, {'index': index, 'payload': 'x' * 10000})
            return index
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=8) as workers:
        winners = [value for value in workers.map(attempt, range(8)) if value is not None]
    assert len(winners) == 1
    assert json.loads(lock.read_text()) == {'index': winners[0], 'payload': 'x' * 10000}
    assert len(list(tmp_path.iterdir())) == 1


def test_snapshot_verifier_rejects_extra_private_fields(prepared):
    h = api()
    report = h.consume(**prepared, output=prepared['repo'] / 'private/run',
                       publish=prepared['repo'] / 'public', frozen=h.preflight(**prepared))
    path = report.parent.parent / 'holdout_snapshot.json'
    value = json.loads(path.read_text())
    value['private_question'] = 'must never appear in a verified public artifact'
    path.write_text(json.dumps(value))
    assert not h.verify_snapshot(report).valid


def test_holdout_cli_roundtrip_and_retry_refusal(prepared, monkeypatch):
    h = api()
    original_preflight, original_consume = h.preflight, h.consume
    monkeypatch.setattr(h, 'preflight', lambda **kw: original_preflight(**{**kw, 'repo': prepared['repo']}))
    monkeypatch.setattr(h, 'consume', lambda **kw: original_consume(**{**kw, 'repo': prepared['repo']}))
    common = ['--dataset', str(prepared['dataset']), '--development', str(DEV), '--corpus-manifest', str(CORPUS)]
    assert main(['--preflight-holdout', '--candidate', str(prepared['candidate']), *common]) == 0
    consume = ['--consume-holdout', '--output', str(prepared['repo'] / 'private/run'),
               '--publish', str(prepared['repo'] / 'public'), *common]
    exit_code = main(consume)
    snapshot = json.loads((prepared['repo'] / 'holdout_snapshot.json').read_text())
    assert exit_code == (0 if snapshot['result']['status'] == 'passed' else 1)
    from scripts.verify_trade_report import main as verify
    assert verify(['--latest', str(prepared['repo'] / 'public'), '--kind', 'holdout']) == 0
    with pytest.raises(FileExistsError, match='consum'):
        main(consume)


def test_failed_holdout_is_reported_and_remains_consumed(prepared, monkeypatch):
    h = api()
    frozen = h.preflight(**prepared)
    from trade_agent.evaluation.runner import LocalCorpusAdapter, StageOutcome
    monkeypatch.setattr(LocalCorpusAdapter, 'execute', lambda *a, **kw: StageOutcome(status='failed', errors=('fixture backend failure',)))
    report = h.consume(**prepared, output=prepared['repo'] / 'private/run',
                       publish=prepared['repo'] / 'public', frozen=frozen)
    value = json.loads((report.parent.parent / 'holdout_snapshot.json').read_text())
    assert value['result']['status'] == 'failed'
    assert 'execution' in value['result']['failed_thresholds']
    assert h.verify_snapshot(report).valid
    assert prepared['dataset'].with_name('holdout_consumption.json').exists()
