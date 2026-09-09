"""Run reproducible local development ablations without service dependencies."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from trade_agent.evaluation.generator import indexed_corpus_content, read_bundle
from trade_agent.evaluation.profiles import EvaluationArm, EvaluationBudget
from trade_agent.evaluation.runner import EvaluationRunner, LocalCorpusAdapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--arms', default=','.join(arm.value for arm in EvaluationArm))
    parser.add_argument('--max-cases', type=int)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-limit', type=int, default=100)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--corpus-manifest', type=Path,
                        default=Path('demo/trade_intel_seed/manifests/corpus_manifest.json'))
    args = parser.parse_args(argv)
    if args.max_cases is not None and args.max_cases < 1:
        parser.error('--max-cases must be positive')
    arms = tuple(EvaluationArm(name.strip()) for name in args.arms.split(','))
    bundle = read_bundle(args.dataset, args.references or args.dataset.with_name('references_dev.jsonl'))
    # Validate every case before truncation so max-cases cannot hide holdout rows.
    if any(case.dataset_role != 'development' for case in bundle.cases):
        parser.error('holdout requires separate frozen-snapshot preflight')
    bundle = replace(bundle, cases=bundle.cases[:args.max_cases])
    manifest = json.loads(args.corpus_manifest.read_text(encoding='utf-8'))
    documents = tuple({'content': item['content'], 'metadata': json.loads(item['content'])}
                      for item in indexed_corpus_content(manifest))
    runner = EvaluationRunner(LocalCorpusAdapter(documents), budget=EvaluationBudget(args.candidate_limit, args.top_k))
    for arm in arms:
        run = runner.run(bundle, arm, args.output)
        print(json.dumps({'arm': arm.value, 'run_id': run.manifest.run_id, 'path': str(run.path),
                          'cases': run.aggregate['case_count'], 'backend': 'local-cpu-baseline'}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
