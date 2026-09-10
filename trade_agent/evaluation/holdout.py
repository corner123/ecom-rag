"""One-shot private holdout execution after a clean, development-validated freeze."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
import re
from pathlib import Path
import subprocess
from uuid import uuid4

from trade_agent.data.manifest import canonical_hash, canonical_json
from trade_agent.evaluation.generator import indexed_corpus_content, read_bundle
from trade_agent.evaluation.leakage import LeakageAuditor
from trade_agent.evaluation.models import EvaluationSnapshot
from trade_agent.evaluation.profiles import EvaluationArm, EvaluationBudget, profile_for
from trade_agent.evaluation.report import ReportWriter, VerificationResult, verify_report_bundle
from trade_agent.evaluation.runner import EvaluationRunner, LocalCorpusAdapter

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / 'demo/trade_intel_seed/manifests/corpus_manifest.json'
DEFAULT_DEV = ROOT / 'data/eval/trade_intel'
ARM = EvaluationArm.FULL_RERANK


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def _ignored(repo, path):
    path = Path(path).resolve()
    try:
        relative = path.relative_to(Path(repo).resolve())
    except ValueError as exc:
        raise ValueError('private artifacts must be Git-ignored inside the repository') from exc
    if subprocess.run(['git', '-C', str(repo), 'check-ignore', '-q', str(relative)],
                      capture_output=True).returncode != 0:
        raise ValueError('private artifacts must be Git-ignored and untracked')


def _atomic_new(path, value):
    """Publish complete durable JSON with atomic no-replace hard-link semantics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            stream.write(canonical_json(value) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _runner(corpus_manifest, budget):
    corpus = indexed_corpus_content(_json(corpus_manifest))
    documents = tuple({'content': item['content'], 'metadata': json.loads(item['content'])} for item in corpus)
    return EvaluationRunner(LocalCorpusAdapter(documents), budget=budget), corpus


def _selected_development(development):
    """Use the immutable published development decision, never holdout scores."""
    from trade_agent.evaluation.cycle import verify_cycle_bundle
    cycles = sorted((development / 'reports_public').glob('cycle-*/paired.json'))
    if len(cycles) != 1:
        raise ValueError('requires one published development selection cycle')
    verified = verify_cycle_bundle(cycles[0].parent)
    if not verified.valid:
        raise ValueError('published development selection verification failed')
    selected = _json(cycles[0])['arms'][ARM.value]
    if not selected['decision']['accepted_change']:
        raise ValueError('development selection threshold did not pass')
    matches = [p.parent for p in (development / 'reports_public').glob('*/manifest.json')
               if _json(p)['run']['run_id'] == selected['candidate_run_id']]
    if len(matches) != 1 or not verify_report_bundle(matches[0]).valid:
        raise ValueError('selected development report is missing or invalid')
    return matches[0], _digest(cycles[0])


def preflight(*, dataset, candidate, corpus_manifest=DEFAULT_CORPUS, development=DEFAULT_DEV, repo=ROOT):
    dataset, candidate, development = Path(dataset), Path(candidate), Path(development)
    lock = dataset.with_name('holdout_consumption.json')
    if lock.exists():
        raise FileExistsError('holdout consumption already recorded; never rerun this holdout')
    if _git(repo, 'status', '--porcelain', '--untracked-files=all'):
        raise ValueError('holdout preflight requires a clean committed repository')
    git_sha = _git(repo, 'rev-parse', 'HEAD')
    references = dataset.with_name('references_private.jsonl')
    for path in (dataset, references, candidate, lock):
        _ignored(repo, path)
    dev = read_bundle(development / 'dev_public.jsonl', development / 'references_dev.jsonl')
    holdout = read_bundle(dataset, references)
    if len(dev.cases) < 36 or any(c.dataset_role != 'development' for c in dev.cases):
        raise ValueError('requires at least 36 development cases')
    if len(holdout.cases) < 15 or any(c.dataset_role != 'holdout' for c in holdout.cases):
        raise ValueError('requires at least 15 private holdout cases')
    paths = [p for p in candidate.glob('*/manifest.json') if _json(p).get('arm') == ARM.value]
    if len(paths) != 1:
        raise ValueError('candidate must contain exactly one full_rerank development run')
    metadata = _json(paths[0])
    if metadata['dataset_role'] != 'development':
        raise ValueError('candidate must be development evidence')
    budget = EvaluationBudget(**metadata['budget'])
    runner, corpus = _runner(corpus_manifest, budget)
    if not LeakageAuditor().audit(dev, holdout, corpus).passed:
        raise ValueError('zero-leakage precondition failed')
    current = runner.freeze(dev, ARM).model_dump(mode='json')
    recorded = metadata['run']['snapshot']
    for key, value in current.items():
        if key != 'snapshot_id' and value != recorded[key]:
            raise ValueError(f'candidate input mismatch: {key}')
    if metadata['adapter'] != runner.adapter.identity or metadata['profile'] != json.loads(canonical_json(asdict(profile_for(ARM)))):
        raise ValueError('candidate adapter/profile input mismatch')
    rows = [json.loads(line) for line in (paths[0].parent / 'per_query.jsonl').read_text().splitlines()]
    if {r['result']['case_id'] for r in rows} != {c.case_id for c in dev.cases} or len(rows) != len(dev.cases):
        raise ValueError('candidate development case set mismatch')
    if any(r['result']['status'] != 'completed' or r['trace']['degradation'] for r in rows):
        raise ValueError('candidate development execution threshold failed')
    aggregate = _json(paths[0].parent / 'aggregate.json')
    if aggregate != runner._aggregate(rows):
        raise ValueError('candidate aggregate does not match development rows')
    selected_path, selection_hash = _selected_development(development)
    selected_meta = _json(selected_path / 'manifest.json')
    for field in ('dataset_hash', 'reference_hash', 'corpus_hash', 'index_hash', 'model_hash', 'prompt_hash', 'profile_hash'):
        if recorded[field] != selected_meta['run']['snapshot'][field]:
            raise ValueError(f'development-selected candidate mismatch: {field}')
    if metadata['budget'] != selected_meta['budget'] or metadata['profile'] != selected_meta['profile']:
        raise ValueError('development-selected config mismatch')
    selected_rows = [json.loads(line) for line in (selected_path / 'per_query.jsonl').read_text().splitlines()]
    expected = {r['result']['case_id']: r for r in selected_rows}
    for row in rows:
        before = expected[row['result']['case_id']]['metrics']['retrieval']
        after = row['metrics']['retrieval']
        if {k: v for k, v in after.items() if k != 'latency_ms'} != {k: v for k, v in before.items() if k != 'latency_ms'}:
            raise ValueError('refreshed development retrieval regression or input mismatch')
    latency = sum(r['result']['latency_ms'] for r in rows) / len(rows)
    baseline_latency = sum(r['result']['latency_ms'] for r in selected_rows) / len(selected_rows)
    if not math.isfinite(latency) or latency > baseline_latency + 50:
        raise ValueError('development latency threshold failed')
    thresholds = {name: aggregate['retrieval'][name]['value'] for name in ('recall_at_10', 'context_precision')}
    identity = {'git_sha': git_sha, 'snapshot': {k: v for k, v in current.items() if k != 'snapshot_id'},
                'config_hash': canonical_hash({'profile': metadata['profile'], 'budget': metadata['budget']}),
                'adapter_hash': canonical_hash(metadata['adapter']), 'selection_hash': selection_hash,
                'candidate_artifact_hash': canonical_hash({name: _digest(paths[0].parent / name)
                    for name in ('manifest.json', 'aggregate.json', 'per_query.jsonl')})}
    candidate_hash = canonical_hash(identity)
    holdout_hash = canonical_hash({'dataset': _digest(dataset), 'references': _digest(references)})
    current.update(snapshot_id='holdout-' + canonical_hash([candidate_hash, holdout_hash])[:32],
                   dataset_hash=canonical_hash([c.model_dump(mode='json') for c in holdout.cases]),
                   reference_hash=canonical_hash([x.model_dump(mode='json') for group in
                       (holdout.evidence, holdout.claims, holdout.decisions, holdout.matches) for x in group]))
    return {'schema_version': 'trade-holdout-freeze/v1', 'arm': ARM.value, 'git_sha': git_sha,
            'candidate_hash': candidate_hash, 'candidate_identity': identity, 'holdout_hash': holdout_hash,
            'corpus_manifest_hash': _digest(corpus_manifest), 'snapshot': current,
            'budget': metadata['budget'], 'thresholds': thresholds,
            'evaluation_scope': 'synthetic_local_retrieval_only', 'generation_status': 'not_run'}


def consume(*, dataset, candidate, output, publish, frozen, corpus_manifest=DEFAULT_CORPUS,
            development=DEFAULT_DEV, repo=ROOT):
    dataset, output, publish = Path(dataset), Path(output), Path(publish)
    lock = dataset.with_name('holdout_consumption.json')
    if lock.exists():
        raise FileExistsError('holdout consumption already recorded; never rerun this holdout')
    snapshot_path = publish.parent / 'holdout_snapshot.json'
    if snapshot_path.exists() or any(_json(p).get('dataset_role') == 'holdout' for p in publish.glob('*/report.json')):
        raise FileExistsError('first holdout report/snapshot already exists; never overwrite')
    _ignored(repo, output)
    if output.exists():
        raise FileExistsError('holdout output must be a new private directory')
    actual = preflight(dataset=dataset, candidate=candidate, corpus_manifest=corpus_manifest,
                       development=development, repo=repo)
    if actual != frozen:
        raise ValueError('frozen inputs changed after preflight')
    runner, _ = _runner(corpus_manifest, EvaluationBudget(**frozen['budget']))
    bundle = read_bundle(dataset, dataset.with_name('references_private.jsonl'))
    # A durable lock is retained on every failure, including a crash before row 1.
    _atomic_new(lock, {'schema_version': 'trade-holdout-consumption/v1', 'status': 'consumed',
                      'consumed_at': datetime.now(timezone.utc).isoformat(), 'frozen': frozen})
    run = runner._run_frozen(bundle, ARM, output, EvaluationSnapshot.model_validate(frozen['snapshot']),
                             holdout_freeze=frozen)
    report = ReportWriter().write(run, publish)
    failures = [name for name, floor in frozen['thresholds'].items()
                if run.aggregate['retrieval'][name]['value'] is None or run.aggregate['retrieval'][name]['value'] < floor]
    if run.aggregate['status_counts'].get('failed', 0) or run.aggregate['degradation']:
        failures.append('execution')
    result = {'status': 'failed' if failures else 'passed', 'failed_thresholds': sorted(failures),
              'policy': 'retrieval metrics must meet frozen development floors; execution must complete',
              'generation_status': 'not_run'}
    _atomic_new(snapshot_path, {**frozen, 'schema_version': 'trade-holdout-snapshot/v1',
                              'report_bundle_id': report.root.name,
                              'report_checksums_hash': _digest(report.checksums), 'result': result})
    verified = verify_snapshot(report.root)
    if not verified.valid:
        raise ValueError('holdout snapshot/report verification failed')
    return report.root


def verify_snapshot(report):
    report = Path(report)
    base = verify_report_bundle(report)
    errors = list(base.errors)
    try:
        saved = _json(report.parent.parent / 'holdout_snapshot.json')
        metadata = _json(report / 'manifest.json')
        frozen = metadata['holdout_freeze']
        fields = {'schema_version', 'arm', 'git_sha', 'candidate_hash', 'candidate_identity',
                  'holdout_hash', 'corpus_manifest_hash', 'snapshot', 'budget', 'thresholds',
                  'evaluation_scope', 'generation_status'}
        if set(frozen) != fields or set(saved) != fields | {'report_bundle_id', 'report_checksums_hash', 'result'}:
            raise ValueError('invalid public freeze schema')
        identity = frozen['candidate_identity']
        if set(identity) != {'git_sha', 'snapshot', 'config_hash', 'adapter_hash', 'selection_hash', 'candidate_artifact_hash'}:
            raise ValueError('invalid public candidate schema')
        EvaluationSnapshot.model_validate(frozen['snapshot'])
        EvaluationSnapshot.model_validate({'snapshot_id': 'development', **identity['snapshot']})
        if 'snapshot_id' in identity['snapshot']:
            raise ValueError('invalid development snapshot schema')
        for digest in (frozen['candidate_hash'], frozen['holdout_hash'], frozen['corpus_manifest_hash'],
                       identity['config_hash'], identity['adapter_hash'], identity['selection_hash'], identity['candidate_artifact_hash']):
            if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                raise ValueError('invalid public digest')
        if not re.fullmatch('[0-9a-f]{40}', frozen['git_sha']) or identity['git_sha'] != frozen['git_sha']:
            raise ValueError('invalid frozen Git identity')
        if frozen['schema_version'] != 'trade-holdout-freeze/v1' or saved['schema_version'] != 'trade-holdout-snapshot/v1':
            raise ValueError('invalid schema version')
        if frozen['arm'] != ARM.value or frozen['evaluation_scope'] != 'synthetic_local_retrieval_only' or frozen['generation_status'] != 'not_run':
            raise ValueError('invalid evaluation scope')
        if set(frozen['thresholds']) != {'recall_at_10', 'context_precision'} or any(
            type(v) not in (int, float) or not 0 <= v <= 1 for v in frozen['thresholds'].values()):
            raise ValueError('invalid frozen thresholds')
        if frozen['budget'] != metadata['budget'] or identity['adapter_hash'] != canonical_hash(metadata['adapter']):
            raise ValueError('frozen adapter/budget mismatch')
        if any(saved.get(k) != v for k, v in frozen.items() if k != 'schema_version'):
            errors.append('holdout snapshot does not match frozen report inputs')
        if frozen['candidate_hash'] != canonical_hash(frozen['candidate_identity']):
            errors.append('candidate hash mismatch')
        if frozen['snapshot'] != metadata['run']['snapshot']:
            errors.append('frozen evaluation snapshot mismatch')
        for key, value in frozen['candidate_identity']['snapshot'].items():
            if key not in ('dataset_hash', 'reference_hash') and frozen['snapshot'][key] != value:
                errors.append(f'frozen candidate mismatch: {key}')
        if frozen['candidate_identity']['config_hash'] != canonical_hash({'profile': metadata['profile'], 'budget': metadata['budget']}):
            errors.append('frozen config mismatch')
        if saved['report_bundle_id'] != report.name or saved['report_checksums_hash'] != _digest(report / 'checksums.sha256'):
            errors.append('snapshot report checksum/identity mismatch')
        aggregate = _json(report / 'aggregate.json')
        failed = [name for name, floor in frozen['thresholds'].items()
                  if aggregate['retrieval'][name]['value'] is None or aggregate['retrieval'][name]['value'] < floor]
        if aggregate['status_counts'].get('failed', 0) or aggregate['degradation']:
            failed.append('execution')
        if saved['result'] != {'status': 'failed' if failed else 'passed', 'failed_thresholds': sorted(failed),
                              'policy': 'retrieval metrics must meet frozen development floors; execution must complete',
                              'generation_status': 'not_run'}:
            errors.append('holdout result does not match frozen thresholds')
    except (OSError, KeyError, TypeError, ValueError) as exc:
        errors.append(f'invalid holdout snapshot: {type(exc).__name__}')
    return VerificationResult(report, not errors, tuple(errors))
