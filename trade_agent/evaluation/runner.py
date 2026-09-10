"""Append-only development ablations with isolated, injectable runtime components.

The CLI local adapter is an explicitly identified CPU baseline, not a production
Milvus/neural embedding evaluation. Only questions reach runtime adapters; private
reference labels are used exclusively by deterministic scoring after retrieval.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import platform
import math
from pathlib import Path
import re
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Mapping, Protocol
from uuid import uuid4

from rank_bm25 import BM25Okapi

from trade_agent.data.manifest import canonical_hash, canonical_json
from trade_agent.evaluation.generator import EvaluationBundle
from trade_agent.evaluation.generation_metrics import evidence_coverage, faithfulness
from trade_agent.evaluation.judge import OptionalJudge
from trade_agent.evaluation.models import EvaluationSnapshot, PerQueryResult, RunManifest
from trade_agent.evaluation.profiles import COMPONENTS, EvaluationArm, EvaluationBudget, profile_for
from trade_agent.evaluation.retrieval_metrics import evaluate_retrieval


@dataclass(frozen=True)
class StageOutcome:
    hits: tuple[Mapping[str, Any], ...] = ()
    status: str = 'available'
    errors: tuple[str, ...] = ()
    degradation: tuple[str, ...] = ()
    answer: Mapping[str, Any] | None = None
    evidence: tuple[Mapping[str, Any], ...] = ()
    guard: Mapping[str, Any] | None = None


class ComponentAdapter(Protocol):
    identity: Mapping[str, Any]

    def execute(self, component: str, query: str, hits: tuple[Mapping[str, Any], ...],
                *, limit: int) -> StageOutcome: ...


@dataclass(frozen=True)
class EvaluationRun:
    path: Path
    manifest: RunManifest
    aggregate: Mapping[str, Any]


def _write_new(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        stream.write(canonical_json(value) + '\n')


def _code_hash() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / 'trade_agent').rglob('*.py'))
    paths.extend(sorted((root / 'scripts').glob('*trade*.py')))
    return canonical_hash({str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest() for path in paths})


def _fuse(dense, sparse, profile, limit):
    scores, records = {}, {}
    for hits, weight in ((dense, profile.dense_weight), (sparse, profile.sparse_weight)):
        seen = set()
        for rank, hit in enumerate(hits, 1):
            key = hit['evidence_id']
            if key in seen:
                continue
            seen.add(key)
            records[key] = hit
            scores[key] = scores.get(key, 0.0) + weight / (profile.rrf_k + rank)
    return tuple({**records[key], 'fusion_score': scores[key]} for key in
                 sorted(records, key=lambda key: (-scores[key], key))[:limit])


class EvaluationRunner:
    def __init__(self, adapter: ComponentAdapter, *, budget: EvaluationBudget | None = None,
                 judge: OptionalJudge | None = None):
        self.adapter = adapter
        self.budget = budget or EvaluationBudget()
        self.judge = judge or OptionalJudge()

    def run(self, bundle: EvaluationBundle, arm: EvaluationArm, output: Path) -> EvaluationRun:
        if any(case.dataset_role == 'holdout' for case in bundle.cases):
            raise ValueError('holdout requires separate frozen-snapshot preflight; this runner is development-only')
        return self._run_frozen(bundle, arm, output, self.freeze(bundle, arm))

    def _run_frozen(self, bundle, arm, output, snapshot, *, holdout_freeze=None):
        if not bundle.cases or len({case.case_id for case in bundle.cases}) != len(bundle.cases):
            raise ValueError('bundle requires nonempty unique cases')
        profile = profile_for(arm)
        identity = dict(self.adapter.identity)
        run_id = f'run-{profile.arm.value}-{uuid4().hex}'
        path = Path(output) / run_id
        path.mkdir(parents=True, exist_ok=False)
        rows = []
        seen_components = set()
        for case in bundle.cases:
            row = self._query(bundle, case, profile, run_id, seen_components)
            # Flush each row independently: a later failure cannot overwrite earlier evidence.
            with (path / 'per_query.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(canonical_json(row) + '\n')
            rows.append(row)
        statuses = {}
        priority = {'not_run': 0, 'available': 1, 'degraded': 2, 'unavailable': 3, 'failed': 4}
        for component in COMPONENTS:
            statuses[component] = max((row['trace']['backend_statuses'][component] for row in rows), key=priority.get)
        manifest = RunManifest(run_id=run_id, snapshot=snapshot, backend_statuses=statuses)
        aggregate = self._aggregate(rows)
        _write_new(path / 'aggregate.json', aggregate)
        _write_new(path / 'manifest.json', {
            'schema_version': 'trade-eval-run/v1', 'run': manifest.model_dump(mode='json'),
            'arm': profile.arm.value, 'created_at': datetime.now(timezone.utc).isoformat(),
            'dataset_role': bundle.cases[0].dataset_role, 'case_count': len(rows),
            **({'holdout_freeze': holdout_freeze} if holdout_freeze else {}),
            'enabled_components': list(profile.enabled_components),
            'disabled_components': [name for name in COMPONENTS if name not in profile.enabled_components],
            'actual_components': [name for name in COMPONENTS if statuses[name] != 'not_run'],
            'budget': asdict(self.budget), 'profile': asdict(profile), 'adapter': identity,
            'evaluator_id': 'trade-deterministic-metrics/v1',
            'timing_policy': 'first invocation within this run labeled cold; subsequent invocations warm; shared adapter/build caches not reset; build excluded',
            'artifacts': {'per_query': 'per_query.jsonl', 'aggregate': 'aggregate.json'},
        })
        return EvaluationRun(path, manifest, aggregate)

    def freeze(self, bundle: EvaluationBundle, arm: EvaluationArm) -> EvaluationSnapshot:
        if not bundle.cases or any(case.dataset_role != 'development' for case in bundle.cases):
            raise ValueError('freeze requires nonempty development cases')
        profile = profile_for(arm)
        identity = dict(self.adapter.identity)
        return EvaluationSnapshot(snapshot_id=f'snapshot-{uuid4().hex}',
            dataset_hash=canonical_hash([case.model_dump(mode='json') for case in bundle.cases]),
            reference_hash=canonical_hash([item.model_dump(mode='json') for collection in
                (bundle.evidence, bundle.claims, bundle.decisions, bundle.matches) for item in collection]),
            corpus_hash=identity['corpus_hash'], index_hash=identity['index_hash'],
            profile_hash=canonical_hash({'profile': asdict(profile), 'budget': asdict(self.budget)}),
            model_hash=canonical_hash(identity['model_id']), prompt_hash=canonical_hash(identity['prompt_id']),
            evaluator_hash=canonical_hash({path.name: sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__file__).parent.glob('*.py'))}), code_hash=_code_hash())

    def _query(self, bundle, case, profile, run_id, seen):
        start = perf_counter()
        statuses = dict.fromkeys(COMPONENTS, 'not_run')
        timings = {'cold': {}, 'warm': {}}
        counts, errors, degradation = {}, [], []
        dense = sparse = hits = ()
        generated = None
        def execute(name, candidates=()):
            stage_start = perf_counter()
            try:
                result = self.adapter.execute(name, case.question, candidates,
                    limit=self.budget.top_k if name == 'generation' else self.budget.candidate_limit)
                if result.status not in {'available', 'degraded', 'unavailable', 'not_run', 'failed'}:
                    raise ValueError(f'invalid stage status: {result.status}')
                if len(result.hits) > (self.budget.top_k if name == 'generation' else self.budget.candidate_limit):
                    raise ValueError(f'{name} exceeded candidate budget')
            except Exception as exc:
                result = StageOutcome(status='failed', errors=(f'{type(exc).__name__}: {exc}',))
            temperature = 'warm' if name in seen else 'cold'
            timings[temperature][name] = (perf_counter() - stage_start) * 1000
            seen.add(name)
            statuses[name] = result.status
            counts[name] = len(result.hits)
            errors.extend(f'{name}: {error}' for error in result.errors)
            degradation.extend(result.degradation)
            if result.status != 'available':
                degradation.append(f'{name}:{result.status}')
            return result
        if 'dense' in profile.enabled_components:
            stage = execute('dense')
            dense = stage.hits if stage.status in ('available', 'degraded') else ()
            hits = dense
        if 'bm25' in profile.enabled_components:
            stage = execute('bm25')
            sparse = stage.hits if stage.status in ('available', 'degraded') else ()
            hits = sparse
        fusion = next((name for name in ('rrf', 'wrrf') if name in profile.enabled_components), None)
        if fusion:
            fusion_start = perf_counter()
            hits = _fuse(dense, sparse, profile, self.budget.candidate_limit)
            statuses[fusion] = 'available'
            timings['warm' if fusion in seen else 'cold'][fusion] = (perf_counter() - fusion_start) * 1000
            seen.add(fusion)
            counts[fusion] = len(hits)
        for name in ('filter', 'reranker'):
            if name in profile.enabled_components:
                stage = execute(name, hits)
                if stage.status in ('available', 'degraded'):
                    hits = stage.hits
        hits = hits[:self.budget.top_k]
        counts['selected'] = len(hits)
        if 'generation' in profile.enabled_components:
            generated = execute('generation', hits)
        latency = (perf_counter() - start) * 1000
        recall_components = [name for name in ('dense', 'bm25') if name in profile.enabled_components]
        failed = all(statuses[name] in ('failed', 'not_run', 'unavailable') for name in recall_components)
        trace = {'backend_statuses': statuses, 'candidate_counts': counts, 'timings_ms': timings,
                 'degradation': degradation, 'hits': hits, 'budget': asdict(self.budget)}
        generation_failed = generated is not None and (generated.status not in ('available', 'degraded') or generated.answer is None)
        status = 'failed' if failed or generation_failed else 'completed'
        if not failed and not generation_failed and generated and generated.answer.get('refusal_reason'):
            status = 'refused'
        result = PerQueryResult(run_id=run_id, case_id=case.case_id, status=status,
            retrieved_evidence_ids=tuple(dict.fromkeys(hit['evidence_id'] for hit in hits)),
            produced_claim_ids=tuple(claim['claim_id'] for claim in
                (generated.answer.get('claims', ()) if generated and generated.answer else ())),
            backend_statuses=statuses, latency_ms=latency)
        retrieval = None if failed else asdict(evaluate_retrieval(case,
            (*bundle.evidence, *bundle.claims, *bundle.matches),
            SimpleNamespace(hits=hits, candidate_counts=counts, degradation=degradation,
                            status=result.status, latency_ms=latency), k=self.budget.top_k))
        generation_metrics = None
        if generated and generated.answer is not None and generated.status in ('available', 'degraded'):
            generation_metrics = {'evidence_coverage': evidence_coverage(generated.answer),
                'faithfulness': asdict(faithfulness(generated.answer, generated.evidence, generated.guard))}
            judge = self.judge.evaluate(generated.answer, evidence=generated.evidence, question=case.question)
        else:
            judge = OptionalJudge().evaluate({})
        return {'result': result.model_dump(mode='json'), 'trace': trace, 'errors': errors,
                'metrics': {'retrieval': retrieval, 'generation': generation_metrics,
                            'fusion': None, 'business': None},
                'metric_coverage': {'retrieval': retrieval is not None, 'generation': generation_metrics is not None,
                                    'fusion': False, 'business': False},
                'generation_outcome': asdict(generated) if generated is not None else None,
                'judge': judge.to_dict()}

    @staticmethod
    def _aggregate(rows):
        metrics = {}
        for name in ('recall_at_10', 'context_precision', 'context_recall', 'reciprocal_rank'):
            values = [row['metrics']['retrieval'][name] for row in rows
                      if row['metrics']['retrieval'] is not None and row['metrics']['retrieval'][name] is not None]
            metrics[name] = {'value': sum(values) / len(values) if values else None,
                             'scored_count': len(values), 'total_count': len(rows)}
        return {'case_count': len(rows), 'status_counts': dict(Counter(row['result']['status'] for row in rows)),
                'retrieval': metrics, 'judge_coverage': sum(row['judge']['coverage'] for row in rows) / len(rows),
                'degradation': sorted({item for row in rows for item in row['trace']['degradation']})}


class LocalCorpusAdapter:
    """Dependency-free-of-services hash cosine/BM25 smoke backend over source payloads.

    This is not a trained semantic embedding model. Reranking is lexical and
    generation is unavailable; both facts are carried in every run manifest.
    """
    def __init__(self, documents):
        documents = tuple(dict(document) for document in documents)
        if not documents:
            raise ValueError('local corpus must be nonempty')
        self.documents = tuple({**document, 'evidence_id': 'rag_' + canonical_hash(document),
                                'branch': 'rag'} for document in documents)
        self.tokens = tuple(self._tokens(item['content']) for item in self.documents)
        self.vectors = tuple(self._vector(tokens) for tokens in self.tokens)
        self.bm25 = BM25Okapi(self.tokens)
        corpus_hash = canonical_hash(documents)
        filter_id = 'explicit-hs-calendar-month/v2'
        build_hash = canonical_hash({'corpus': corpus_hash, 'filter': filter_id})
        self.identity = {'backend': 'local-cpu-baseline', 'build_id': f'local-build-{build_hash[:32]}',
            'filter_id': filter_id,
            'corpus_hash': corpus_hash, 'index_hash': canonical_hash({'corpus': corpus_hash, 'version': 'local-v1'}),
            'model_id': 'sha256-token-cosine-256/v1;bm25-okapi-k1-1.5-b-0.75;lexical-rerank/v1',
            'prompt_id': 'generation-unavailable/v1', 'generation_available': False,
            'runtime': {'python': platform.python_version(), 'rank_bm25': version('rank-bm25'),
                        'numpy': version('numpy'), 'platform': platform.system(), 'machine': platform.machine()}}

    @staticmethod
    def _tokens(text):
        return re.findall(r'\w+', text.lower())

    @staticmethod
    def _vector(tokens):
        vector = Counter(int(sha256(token.encode()).hexdigest()[:8], 16) % 256 for token in tokens)
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {key: value / norm for key, value in vector.items()}

    def execute(self, component, query, hits, *, limit):
        tokens = self._tokens(query)
        if component in ('dense', 'bm25'):
            if component == 'dense':
                vector = self._vector(tokens)
                scores = [sum(value * candidate.get(key, 0.0) for key, value in vector.items())
                          for candidate in self.vectors]
            else:
                scores = self.bm25.get_scores(tokens)
            positions = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), self.documents[index]['evidence_id']))
            return StageOutcome(hits=tuple({**self.documents[index], 'score': float(scores[index])}
                                           for index in positions[:limit] if scores[index] > 0))
        if component == 'filter':
            hs_codes = re.findall(r'\bHS\s+(\d{4,10})\b', query, re.I)
            months = re.findall(r'\b(\d{4}-(?:0[1-9]|1[0-2]))\b', query)
            filtered = tuple(hit for hit in hits
                if (not hs_codes or str(hit.get('metadata', {}).get('hs_code', '')) in hs_codes)
                and (not months or hit.get('metadata', {}).get('calendar_month') in months))
            return StageOutcome(hits=filtered[:limit])
        if component == 'reranker':
            ordered = sorted(hits, key=lambda hit: (-len(set(tokens) & set(self._tokens(hit['content']))), hit['evidence_id']))
            return StageOutcome(hits=tuple(ordered[:limit]))
        if component == 'generation':
            return StageOutcome(status='not_run', errors=('local baseline has no generation provider',))
        raise ValueError(f'unsupported local component: {component}')
