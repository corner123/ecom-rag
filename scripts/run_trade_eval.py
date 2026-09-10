"""Run reproducible local development ablations without service dependencies."""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from trade_agent.data.manifest import canonical_json
from trade_agent.evaluation.cycle import compare_runs, index_arms, publish_cycle
from trade_agent.evaluation.error_analysis import load_evaluation_runs
from trade_agent.evaluation.generator import indexed_corpus_content, read_bundle
from trade_agent.evaluation.profiles import EvaluationArm, EvaluationBudget
from trade_agent.evaluation.runner import EvaluationRunner, LocalCorpusAdapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--arms', default=','.join(arm.value for arm in EvaluationArm))
    parser.add_argument('--freeze-only', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight-holdout', action='store_true')
    mode.add_argument('--consume-holdout', action='store_true')
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--development', type=Path, default=Path('data/eval/trade_intel'))
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--publish', type=Path)
    parser.add_argument('--max-cases', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--candidate-limit', type=int, default=100)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--corpus-manifest', type=Path,
                        default=Path('demo/trade_intel_seed/manifests/corpus_manifest.json'))
    args = parser.parse_args(argv)
    if args.preflight_holdout or args.consume_holdout:
        from trade_agent.evaluation.holdout import _atomic_new, consume, preflight
        if args.freeze_only or args.baseline or args.max_cases is not None or args.references:
            parser.error('holdout cannot truncate, override references, or run development modes')
        if args.arms != ','.join(arm.value for arm in EvaluationArm) or args.candidate_limit != 100 or args.top_k != 10:
            parser.error('holdout uses the frozen development-selected arm and config')
        freeze_path = args.dataset.with_name('holdout_preflight.json')
        if args.preflight_holdout:
            if args.candidate is None:
                parser.error('--preflight-holdout requires --candidate')
            frozen = preflight(dataset=args.dataset, candidate=args.candidate,
                               corpus_manifest=args.corpus_manifest, development=args.development)
            payload = {'candidate': str(args.candidate.resolve()), 'frozen': frozen}
            if freeze_path.exists():
                if json.loads(freeze_path.read_text()) != payload:
                    parser.error('existing holdout preflight differs; do not overwrite')
            else:
                _atomic_new(freeze_path, payload)
            print(json.dumps({'preflight': 'passed', 'candidate_hash': frozen['candidate_hash'],
                              'git_sha': frozen['git_sha']}))
            return 0
        if args.output is None or args.publish is None or not freeze_path.is_file():
            parser.error('--consume-holdout requires saved preflight, --output, and --publish')
        saved = json.loads(freeze_path.read_text())
        report = consume(dataset=args.dataset, candidate=args.candidate or Path(saved['candidate']),
                         output=args.output, publish=args.publish, frozen=saved['frozen'],
                         corpus_manifest=args.corpus_manifest, development=args.development)
        result = json.loads((args.publish.parent / 'holdout_snapshot.json').read_text())['result']
        print(json.dumps({'published_holdout': str(report), 'result': result}, sort_keys=True))
        return 0 if result['status'] == 'passed' else 1
    if args.output is None:
        parser.error('--output is required')
    if args.max_cases is not None and args.max_cases < 1:
        parser.error('--max-cases must be positive')
    if args.publish and not args.baseline:
        parser.error('--publish requires --baseline')
    if args.freeze_only and (args.baseline or args.publish):
        parser.error('--freeze-only cannot compare or publish')
    arms = tuple(EvaluationArm(name.strip()) for name in args.arms.split(','))
    bundle = read_bundle(args.dataset, args.references or args.dataset.with_name('references_dev.jsonl'))
    # Validate every case before truncation so max-cases cannot hide holdout rows.
    if any(case.dataset_role != 'development' for case in bundle.cases):
        parser.error('holdout requires separate frozen-snapshot preflight')
    bundle = replace(bundle, cases=bundle.cases[:args.max_cases])
    baselines = load_evaluation_runs((args.baseline,)) if args.baseline else ()
    if baselines and set(index_arms(baselines)) != {arm.value for arm in arms}:
        parser.error('baseline and candidate arm sets must match')
    manifest = json.loads(args.corpus_manifest.read_text(encoding='utf-8'))
    documents = tuple({'content': item['content'], 'metadata': json.loads(item['content'])}
                      for item in indexed_corpus_content(manifest))
    runner = EvaluationRunner(LocalCorpusAdapter(documents), budget=EvaluationBudget(args.candidate_limit, args.top_k))
    if args.freeze_only:
        args.output.mkdir(parents=True, exist_ok=True)
        payload = {'schema_version': 'trade-development-freeze/v1', 'dataset_role': 'development',
            'corpus_manifest_hash': sha256(args.corpus_manifest.read_bytes()).hexdigest(),
            'adapter': runner.adapter.identity,
            'snapshots': {arm.value: runner.freeze(bundle, arm).model_dump(mode='json') for arm in arms}}
        with (args.output / 'snapshot.json').open('x', encoding='utf-8') as stream:
            stream.write(canonical_json(payload) + '\n')
        print(json.dumps({'snapshot': str(args.output / 'snapshot.json')}))
        return 0
    runs = []
    for arm in arms:
        run = runner.run(bundle, arm, args.output)
        runs.append(run)
        print(json.dumps({'arm': arm.value, 'run_id': run.manifest.run_id, 'path': str(run.path),
                          'cases': run.aggregate['case_count'], 'backend': 'local-cpu-baseline'}, sort_keys=True))
    if baselines:
        cycle = compare_runs(baselines, runs)
        with (args.output / ('paired-' + runs[0].manifest.run_id + '.json')).open('x', encoding='utf-8') as stream:
            stream.write(canonical_json(cycle) + '\n')
        if args.publish:
            path = publish_cycle(cycle, (*baselines, *runs), args.publish)
            print(json.dumps({'published_cycle': str(path)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
