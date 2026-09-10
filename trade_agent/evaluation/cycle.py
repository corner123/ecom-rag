"""Paired development evidence; retrieval labels never enter runtime adapters."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

from trade_agent.data.manifest import canonical_json
from trade_agent.evaluation.error_analysis import read_development_metadata, read_per_query_rows
from trade_agent.evaluation.optimizer import DevelopmentOptimizer, OptimizationObjective, _paired_deltas, _value
from trade_agent.evaluation.report import ReportWriter, verify_report_bundle


def index_arms(runs):
    indexed = {}
    for run in runs:
        arm = read_development_metadata(run)['arm']
        if arm in indexed:
            raise ValueError(f'baseline requires exactly one run per arm: {arm}')
        indexed[arm] = run
    return indexed


def compare_runs(baselines, candidates):
    before, after = index_arms(baselines), index_arms(candidates)
    if before.keys() != after.keys():
        raise ValueError('baseline and candidate arm sets must match')
    arms = {}
    for arm, candidate in after.items():
        baseline = before[arm]
        for field in ('dataset_hash', 'reference_hash', 'corpus_hash', 'index_hash',
                      'model_hash', 'profile_hash', 'prompt_hash', 'evaluator_hash'):
            if getattr(baseline.manifest.snapshot, field) != getattr(candidate.manifest.snapshot, field):
                raise ValueError(f'paired development cycle requires matching {field}')
        baseline_metadata = read_development_metadata(baseline)
        candidate_metadata = read_development_metadata(candidate)
        for field in ('budget', 'profile'):
            if (not isinstance(baseline_metadata.get(field), dict)
                    or baseline_metadata[field] != candidate_metadata.get(field)):
                raise ValueError(f'paired development cycle requires matching {field}')
        rows = []
        for run in (baseline, candidate):
            rows.append({row['result']['case_id']: row for row in read_per_query_rows(run, read_development_metadata(run))})
        if rows[0].keys() != rows[1].keys():
            raise ValueError('paired development cycle requires matching case sets')
        decision = DevelopmentOptimizer().select((baseline, candidate), OptimizationObjective(baseline.manifest.run_id))
        regressions = {metric: sorted(case for case in rows[0]
            if _value(rows[0][case], metric) is not None and _value(rows[1][case], metric) is not None
            and _value(rows[1][case], metric) < _value(rows[0][case], metric))
            for metric in DevelopmentOptimizer.priority}
        arms[arm] = {'baseline_run_id': baseline.manifest.run_id, 'candidate_run_id': candidate.manifest.run_id,
            'baseline_status_counts': baseline.aggregate['status_counts'],
            'candidate_status_counts': candidate.aggregate['status_counts'],
            'paired_deltas': {name: asdict(delta) for name, delta in _paired_deltas(*rows).items()},
            'regressions': regressions, 'decision': decision.to_dict()}
    return {'schema_version': 'trade-development-cycle/v1', 'dataset_role': 'development',
        'qualification': 'Synthetic development seed; local CPU hash cosine/BM25/lexical reranker, not production or neural retrieval. '
            'No generation provider: full_e2e fails and is rejected; generation/faithfulness and judge scores remain null; judge_not_run. '
            'Latency is a single local sequential pass, excludes build time, and is not a production benchmark. No holdout tuning.',
        'arms': arms}


def cycle_markdown(cycle):
    lines = ['# Trade development cycle', '', cycle['qualification'], '',
        '| Arm | Recall before → after (Δ) | Precision before → after (Δ) | Mean latency ms before → after (Δ) | Regressions recall / precision | Accepted |',
        '|---|---|---|---|---|---|']
    for arm, entry in sorted(cycle['arms'].items()):
        values = []
        for name in ('recall_at_10', 'context_precision', 'latency_ms'):
            d = entry['paired_deltas'][name]
            values.append('null' if d['delta'] is None else f"{d['baseline_mean']:.6f} → {d['candidate_mean']:.6f} ({d['delta']:+.6f})")
        r = entry['regressions']
        lines.append(f"| {arm} | {' | '.join(values)} | {len(r['recall_at_10'])} / {len(r['context_precision'])} | {entry['decision']['accepted_change']} |")
    lines += ['', 'Rule metrics are authoritative. See paired.json for all paired counts, refusal deltas, null faithfulness, case regressions, and execution rejections.', '', '## Immutable run bundles', '']
    for run_id, path in sorted(cycle.get('bundles', {}).items()):
        lines.append(f'- [{run_id}](../{path}/report.md)')
    return '\n'.join(lines) + '\n'


def publish_cycle(cycle, runs, root):
    runs = tuple(runs)
    by_id = {run.manifest.run_id: run for run in runs}
    if len(by_id) != len(runs):
        raise ValueError('publication requires unique run IDs')
    try:
        before = [by_id[item['baseline_run_id']] for item in cycle['arms'].values()]
        after = [by_id[item['candidate_run_id']] for item in cycle['arms'].values()]
    except KeyError as exc:
        raise ValueError('publication requires every paired run') from exc
    recomputed = compare_runs(before, after)
    if canonical_json(recomputed) != canonical_json(cycle):
        raise ValueError('paired evidence does not match publication runs')
    root = Path(root)
    writer = ReportWriter()
    bundles = {}
    for run in runs:
        bundle = writer.write(run, root)
        verified = verify_report_bundle(bundle.root)
        if not verified.valid:
            raise ValueError(verified.errors)
        bundles[run.manifest.run_id] = bundle.root.name
    cycle = {**cycle, 'bundles': bundles}
    destination = root / ('cycle-' + uuid4().hex)
    destination.mkdir(exist_ok=False)
    (destination / 'paired.json').write_text(canonical_json(cycle) + '\n')
    (destination / 'report.md').write_text(cycle_markdown(cycle))
    (destination / 'checksums.sha256').write_text(''.join(
        f'{sha256(path.read_bytes()).hexdigest()}  {path.name}\n' for path in sorted(destination.iterdir())))
    readme = root / 'README.md'
    with readme.open('a', encoding='utf-8') as stream:
        if readme.stat().st_size == 0:
            stream.write('# Public trade development evidence\n\nSynthetic, non-production local evaluation. Each cycle and run is immutable.\n\n')
        stream.write(f'- [Paired development cycle]({destination.name}/report.md) ([machine-readable deltas]({destination.name}/paired.json))\n')
    return destination


def verify_cycle_bundle(path):
    """Verify checksums and recompute paired deltas from the linked public runs."""
    from trade_agent.evaluation.error_analysis import load_evaluation_runs
    from trade_agent.evaluation.report import VerificationResult
    path = Path(path)
    errors = []
    try:
        names = {'paired.json', 'report.md', 'checksums.sha256'}
        if path.is_symlink() or {item.name for item in path.iterdir()} != names:
            raise ValueError('unsafe or unexpected cycle files')
        if any((path / name).is_symlink() for name in names):
            raise ValueError('unsafe cycle symlink')
        expected = ''.join(f'{sha256((path / name).read_bytes()).hexdigest()}  {name}\n'
                           for name in ('paired.json', 'report.md'))
        if (path / 'checksums.sha256').read_text() != expected:
            raise ValueError('cycle checksum mismatch')
        cycle = json.loads((path / 'paired.json').read_text())
        runs = {}
        for run_id, name in cycle['bundles'].items():
            if Path(name).name != name or not name.startswith('report-'):
                raise ValueError('unsafe linked bundle path')
            destination = path.parent / name
            result = verify_report_bundle(destination)
            if not result.valid:
                raise ValueError(f'linked bundle invalid: {result.errors}')
            run = load_evaluation_runs((destination,))[0]
            if run.manifest.run_id != run_id:
                raise ValueError('linked run identity mismatch')
            runs[run_id] = run
        before = [runs[item['baseline_run_id']] for item in cycle['arms'].values()]
        after = [runs[item['candidate_run_id']] for item in cycle['arms'].values()]
        recomputed = {**compare_runs(before, after), 'bundles': cycle['bundles']}
        if canonical_json(recomputed) != canonical_json(cycle):
            raise ValueError('paired evidence does not match linked runs')
        if (path / 'report.md').read_text() != cycle_markdown(cycle):
            raise ValueError('cycle Markdown mismatch')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
    return VerificationResult(path, not errors, tuple(errors))
