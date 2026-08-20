"""Auditable generation and RAGAS judging for response-eval/v3 datasets.

The experiment has two deliberately separable paid stages:

``generate``
    Retrieve evidence and ask the configured answer model.  Reference answers
    are never passed to the service or generator.

``judge``
    Replay persisted generation records, join them to the reviewed dataset by
    sample id, and then run explicit RAGAS judges.  Raw Top-K contexts are used
    for retrieval metrics; the actual model context is used for response
    metrics.

Artifacts are JSON, written through a same-directory temporary file followed
by ``os.replace``.  Existing files are never overwritten unless ``resume`` is
explicit and the frozen experiment identity still matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from rag_core.engineering import (
    EngineeringIndex,
    EngineeringRAGService,
    EvidenceSufficiencyGuard,
    SupportSelectionProfile,
    SufficiencyProfile,
)
from rag_core.engineering.deepseek_generation import (
    DeepSeekGenerationSettings,
    DeepSeekGroundedGenerator,
)
from rag_core.engineering.grounding import GroundedAnswerer
from rag_core.ingestion import BuildManifest
from rag_core.retrieval.engineering import (
    LiveASTRetriever,
    LiveCodeRetriever,
    LiveGitVerifier,
)

from .engineering_adapter import (
    _assert_live_repo_matches_manifest,
    _retriever_view,
)
from .ragas_eval import RAGASEvaluationReport, RAGASEvaluator, RAGASSample
from .response_dataset import (
    ResponseEvaluationSample,
    dataset_sha256,
    file_sha256,
    load_response_jsonl,
)
from .response_report import (
    DETERMINISTIC_RETRIEVAL_METRICS,
    RESPONSE_METRICS,
)
from .retrieval_profiles import RetrievalExperimentProfile


SNAPSHOT_SCHEMA = "engineering-response-snapshot/v1"
ARTIFACT_SCHEMA = "engineering-response-experiment/v1"
DETERMINISTIC_RETRIEVAL_METRIC_SCHEMA = (
    "engineering-response-deterministic-retrieval/v2"
)
RETRIEVAL_METRICS = ("context_precision", "context_recall")
ANSWER_METRICS = ("faithfulness", "answer_relevancy", "answer_correctness")
JUDGE_AUDIT_SCHEMA = "engineering-response-judge-audit/v1"
JUDGE_IDENTITY_FIELDS = (
    "provider",
    "base_url",
    "model",
    "temperature",
    "thinking_enabled",
    "max_tokens",
    "timeout_seconds",
    "max_retries",
    "ragas_version",
    "embedding_provider",
    "embedding_model",
    "embedding_device",
    "embedding_normalize",
    "prompt_sha256",
    "run_config_seed",
    "run_config_max_workers",
)
class ResponseExperimentValidationError(RuntimeError):
    """Raised when a frozen or replayed experiment cannot be trusted."""


@dataclass(frozen=True, slots=True)
class FrozenResponseInputs:
    samples: tuple[ResponseEvaluationSample, ...]
    manifest: BuildManifest
    snapshot: Mapping[str, Any]
    dataset_file_sha256: str
    dataset_canonical_sha256: str
    snapshot_file_sha256: str
    index_catalog: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RAGASJudgeBundle:
    """Two strict evaluators so each metric sees the correct context set."""

    retrieval: RAGASEvaluator
    response: RAGASEvaluator
    metadata: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResponseExperimentValidationError(f"{name} must be a JSON object")
    return value


def _load_json(path: str | Path, name: str) -> Mapping[str, Any]:
    candidate = Path(path)
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResponseExperimentValidationError(
            f"cannot read {name}: {candidate}: {exc}"
        ) from exc
    return _require_mapping(payload, name)


def build_response_snapshot(
    *,
    dataset_path: str | Path,
    manifest_path: str | Path,
    index_root: str | Path,
) -> dict[str, Any]:
    """Build (but do not persist) a frozen response-experiment declaration."""

    samples = load_response_jsonl(dataset_path, formal=True)
    manifest = BuildManifest.read(manifest_path)
    catalog_path = Path(index_root) / "partitions.json"
    catalog = _load_json(catalog_path, "engineering index catalog")
    if str(catalog.get("build_id") or "") != manifest.build_id:
        raise ResponseExperimentValidationError(
            "cannot freeze a stale index: "
            f"manifest={manifest.build_id}, index={catalog.get('build_id')}"
        )
    roles = sorted({sample.dataset_role for sample in samples})
    if len(roles) != 1:
        raise ResponseExperimentValidationError(
            "one frozen response experiment cannot mix development and holdout roles"
        )
    return {
        "schema_version": SNAPSHOT_SCHEMA,
        "created_at": _utc_now(),
        "manifest_build_id": manifest.build_id,
        "manifest_sha256": file_sha256(manifest_path),
        "index_catalog_sha256": file_sha256(catalog_path),
        "dataset": {
            "file_sha256": file_sha256(dataset_path),
            "canonical_sha256": dataset_sha256(samples),
            "question_count": len(samples),
            "dataset_roles": roles,
            "source_revisions": sorted({sample.source_revision for sample in samples}),
        },
    }


def validate_frozen_response_inputs(
    *,
    dataset_path: str | Path,
    snapshot_path: str | Path,
    manifest_path: str | Path,
    index_root: str | Path,
) -> FrozenResponseInputs:
    """Fail closed if any frozen dataset/build byte has changed."""

    samples = tuple(load_response_jsonl(dataset_path, formal=True))
    manifest = BuildManifest.read(manifest_path)
    snapshot = _load_json(snapshot_path, "response evaluation snapshot")
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA:
        raise ResponseExperimentValidationError(
            f"response snapshot must use {SNAPSHOT_SCHEMA}"
        )
    dataset_meta = _require_mapping(snapshot.get("dataset"), "snapshot.dataset")
    actual_dataset_file_sha = file_sha256(dataset_path)
    actual_dataset_sha = dataset_sha256(samples)
    actual_manifest_sha = file_sha256(manifest_path)
    catalog_path = Path(index_root) / "partitions.json"
    catalog = _load_json(catalog_path, "engineering index catalog")
    actual_catalog_sha = file_sha256(catalog_path)
    checks = {
        "dataset file SHA": (
            str(dataset_meta.get("file_sha256") or ""),
            actual_dataset_file_sha,
        ),
        "dataset canonical SHA": (
            str(dataset_meta.get("canonical_sha256") or ""),
            actual_dataset_sha,
        ),
        "dataset question count": (
            dataset_meta.get("question_count"),
            len(samples),
        ),
        "dataset roles": (
            sorted(dataset_meta.get("dataset_roles") or []),
            sorted({sample.dataset_role for sample in samples}),
        ),
        "dataset source revisions": (
            sorted(dataset_meta.get("source_revisions") or []),
            sorted({sample.source_revision for sample in samples}),
        ),
        "manifest SHA": (
            str(snapshot.get("manifest_sha256") or ""),
            actual_manifest_sha,
        ),
        "manifest build id": (
            str(snapshot.get("manifest_build_id") or ""),
            manifest.build_id,
        ),
        "index catalog SHA": (
            str(snapshot.get("index_catalog_sha256") or ""),
            actual_catalog_sha,
        ),
        "index build id": (
            str(catalog.get("build_id") or ""),
            manifest.build_id,
        ),
    }
    mismatches = {
        name: {"frozen": expected, "actual": actual}
        for name, (expected, actual) in checks.items()
        if expected != actual
    }
    if mismatches:
        raise ResponseExperimentValidationError(
            "frozen response inputs do not match: " + _canonical_json(mismatches)
        )
    return FrozenResponseInputs(
        samples=samples,
        manifest=manifest,
        snapshot=snapshot,
        dataset_file_sha256=actual_dataset_file_sha,
        dataset_canonical_sha256=actual_dataset_sha,
        snapshot_file_sha256=file_sha256(snapshot_path),
        index_catalog=catalog,
    )


def build_experiment_service(
    *,
    index_root: str | Path,
    manifest: BuildManifest,
    mini_nanobot_repo: str | Path,
    profile: RetrievalExperimentProfile,
    sufficiency_profile: SufficiencyProfile | str = (
        SufficiencyProfile.LEGACY_EXACT_SLASH
    ),
    support_selection_profile: SupportSelectionProfile | str = (
        SupportSelectionProfile.LEGACY_FIRST
    ),
    generator: Any | None = None,
) -> EngineeringRAGService:
    """Build the real FAISS/live-verification service for one named profile."""

    index = EngineeringIndex.load(index_root, runtime_backend="faiss")
    if index.build_id != manifest.build_id:
        raise ResponseExperimentValidationError(
            f"stale FAISS index: manifest={manifest.build_id}, index={index.build_id}"
        )
    repo = Path(mini_nanobot_repo).expanduser().resolve()
    if not repo.is_dir():
        raise ResponseExperimentValidationError(
            f"Mini-Nanobot repository does not exist: {repo}"
        )
    _assert_live_repo_matches_manifest(repo, manifest)
    if generator is None:
        settings = DeepSeekGenerationSettings.from_env()
        generator = DeepSeekGroundedGenerator(settings)
    answerer = GroundedAnswerer(
        generator,
        provider=str(getattr(generator, "provider", "deepseek")),
        model=getattr(generator, "model", None),
    )
    return EngineeringRAGService(
        _retriever_view(index, "hybrid", profile=profile),
        live_code=LiveCodeRetriever(
            repo,
            corpus="internal",
            authority="code",
            case_sensitive=True,
        ),
        live_ast=LiveASTRetriever(repo),
        live_git=LiveGitVerifier(repo),
        answerer=answerer,
        sufficiency_guard=EvidenceSufficiencyGuard(sufficiency_profile),
        support_selection_profile=support_selection_profile,
        index_stats=index.stats(),
    )


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()
    if hasattr(value, "dict"):
        try:
            return value.dict()
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): _model_dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_model_dump(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _metric_prompt_hash(metrics: Mapping[str, Any]) -> str:
    payload: dict[str, Any] = {}
    for name, metric in sorted(metrics.items()):
        prompts = {
            key: _model_dump(value)
            for key, value in vars(metric).items()
            if "prompt" in key.casefold()
        }
        payload[name] = {
            "class": f"{type(metric).__module__}.{type(metric).__qualname__}",
            "prompts": prompts,
        }
    return _sha256_text(_canonical_json(payload))


def build_deepseek_ragas_judges(
    environ: Mapping[str, str] | None = None,
) -> RAGASJudgeBundle:
    """Build an explicit DeepSeek judge and local BGE evaluator embeddings."""

    env = os.environ if environ is None else environ
    api_key = str(env.get("DEEPSEEK_API_KEY", "")).strip()
    if not api_key or api_key.casefold().startswith(("replace-", "your_")):
        raise ResponseExperimentValidationError(
            "DEEPSEEK_API_KEY is required for RAGAS judging"
        )
    base_url = str(env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).strip()
    judge_model = str(
        env.get("RAGAS_JUDGE_MODEL", env.get("DEEPSEEK_JUDGE_MODEL", "deepseek-v4-pro"))
    ).strip()
    embedding_model = str(
        env.get("RAGAS_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    ).strip()
    timeout = float(env.get("RAGAS_JUDGE_TIMEOUT_SECONDS", "90"))
    max_retries = int(env.get("RAGAS_JUDGE_MAX_RETRIES", "1"))
    max_tokens_raw = str(env.get("RAGAS_JUDGE_MAX_TOKENS", "")).strip()
    max_tokens = int(max_tokens_raw) if max_tokens_raw else None
    embedding_device = str(env.get("RAGAS_EMBEDDING_DEVICE", "cpu"))

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas import RunConfig
    from ragas.metrics import (
        AnswerCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=judge_model,
        temperature=0,
        timeout=timeout,
        max_retries=max_retries,
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )
    run_config = RunConfig(
        timeout=int(timeout),
        max_retries=max_retries,
        max_workers=1,
        seed=20260813,
    )
    wrappers = RAGASEvaluator.from_langchain(
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )
    retrieval_metrics = {
        "context_precision": LLMContextPrecisionWithReference(
            llm=wrappers.llm_wrapper
        ),
        "context_recall": LLMContextRecall(llm=wrappers.llm_wrapper),
    }
    response_metrics = {
        "faithfulness": Faithfulness(llm=wrappers.llm_wrapper),
        "answer_relevancy": ResponseRelevancy(
            llm=wrappers.llm_wrapper,
            embeddings=wrappers.embeddings_wrapper,
        ),
        "answer_correctness": AnswerCorrectness(
            llm=wrappers.llm_wrapper,
            embeddings=wrappers.embeddings_wrapper,
        ),
    }
    retrieval = RAGASEvaluator(
        llm_wrapper=wrappers.llm_wrapper,
        embeddings_wrapper=wrappers.embeddings_wrapper,
        metrics=retrieval_metrics,
        run_config=run_config,
    )
    response = RAGASEvaluator(
        llm_wrapper=wrappers.llm_wrapper,
        embeddings_wrapper=wrappers.embeddings_wrapper,
        metrics=response_metrics,
        run_config=run_config,
    )
    prompt_hash = _metric_prompt_hash({**retrieval_metrics, **response_metrics})
    import ragas

    return RAGASJudgeBundle(
        retrieval=retrieval,
        response=response,
        metadata={
            "provider": "deepseek",
            "base_url": base_url.rstrip("/"),
            "model": judge_model,
            "temperature": 0.0,
            "thinking_enabled": False,
            "max_tokens": max_tokens,
            "timeout_seconds": timeout,
            "max_retries": max_retries,
            "ragas_version": getattr(ragas, "__version__", "unknown"),
            "embedding_provider": "local_huggingface",
            "embedding_model": embedding_model,
            "embedding_device": embedding_device,
            "embedding_normalize": True,
            "prompt_sha256": prompt_hash,
            "run_config_seed": 20260813,
            "run_config_max_workers": 1,
            "self_judge_bias_warning": (
                "DeepSeek-family answers judged by a DeepSeek-family model may "
                "have correlated bias."
            ),
        },
    )


def _source_sha256(module_file: str | None) -> str | None:
    if not module_file:
        return None
    path = Path(module_file)
    return file_sha256(path) if path.is_file() else None


def _code_hashes() -> dict[str, str | None]:
    from rag_core.engineering import grounding, index, sufficiency, support_selection
    from rag_core.engineering import deepseek_generation
    from rag_core.evaluation import ragas_eval
    from rag_core.evaluation import retrieval_profiles
    from rag_core.retrieval.engineering import routing

    return {
        "response_experiment": _source_sha256(__file__),
        "grounding": _source_sha256(grounding.__file__),
        "deepseek_generation": _source_sha256(deepseek_generation.__file__),
        "ragas_eval": _source_sha256(ragas_eval.__file__),
        "sufficiency": _source_sha256(sufficiency.__file__),
        "support_selection": _source_sha256(support_selection.__file__),
        "routing": _source_sha256(routing.__file__),
        "index": _source_sha256(index.__file__),
        "retrieval_profiles": _source_sha256(retrieval_profiles.__file__),
    }


def _generator_metadata(service: EngineeringRAGService) -> dict[str, Any]:
    from rag_core.engineering.deepseek_generation import _SYSTEM_PROMPT

    answerer = service.answerer
    prompt_material = "\n".join(
        (
            inspect.getsource(type(answerer).build_prompt),
            inspect.getsource(type(answerer).prepare_evidence),
        )
    )
    generator = answerer.generator
    settings = getattr(generator, "settings", None)
    return {
        "provider": answerer.provider,
        "model": answerer.model,
        "temperature": getattr(settings, "temperature", None),
        "max_tokens": getattr(settings, "max_tokens", None),
        "thinking_enabled": getattr(settings, "thinking_enabled", None),
        "system_prompt_sha256": _sha256_text(_SYSTEM_PROMPT),
        "prompt_code_sha256": _sha256_text(prompt_material),
    }


def _experiment_metadata(
    *,
    frozen: FrozenResponseInputs,
    profile: RetrievalExperimentProfile,
    sufficiency_profile: SufficiencyProfile | str,
    support_selection_profile: SupportSelectionProfile | str,
    top_k: int,
    service: EngineeringRAGService | None,
) -> dict[str, Any]:
    selected_sufficiency_profile = SufficiencyProfile(sufficiency_profile)
    selected_support_selection_profile = SupportSelectionProfile(
        support_selection_profile
    )
    stable = {
        "dataset_file_sha256": frozen.dataset_file_sha256,
        "dataset_canonical_sha256": frozen.dataset_canonical_sha256,
        "snapshot_file_sha256": frozen.snapshot_file_sha256,
        "manifest_build_id": frozen.manifest.build_id,
        "index_catalog_sha256": str(
            frozen.snapshot.get("index_catalog_sha256") or ""
        ),
        "profile": profile.as_metadata(),
        "sufficiency_profile": selected_sufficiency_profile.value,
        "support_selection_profile": selected_support_selection_profile.value,
        "top_k": top_k,
        "code_sha256": _code_hashes(),
    }
    if service is not None:
        stable["generator"] = _generator_metadata(service)
    return {
        **stable,
        "experiment_id": _sha256_text(_canonical_json(stable)),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_response_snapshot(
    *,
    dataset_path: str | Path,
    manifest_path: str | Path,
    index_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze current response inputs once; never overwrite an earlier run."""

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(
            f"response evaluation snapshot already exists: {output}"
        )
    snapshot = build_response_snapshot(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        index_root=index_root,
    )
    _atomic_write_json(output, snapshot)
    return snapshot


def _new_artifact(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "stage": "initialized",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "metadata": dict(metadata),
        "records": [],
    }


def _load_resume_artifact(
    path: Path,
    *,
    expected_experiment_id: str,
    resume: bool,
) -> dict[str, Any]:
    if not path.exists():
        return _new_artifact({})
    if not resume:
        raise FileExistsError(
            f"response experiment output already exists: {path}; use --resume only "
            "to continue the exact same frozen experiment"
        )
    artifact = dict(_load_json(path, "response experiment artifact"))
    if artifact.get("schema_version") != ARTIFACT_SCHEMA:
        raise ResponseExperimentValidationError("unsupported replay artifact schema")
    metadata = _require_mapping(artifact.get("metadata"), "artifact.metadata")
    if metadata.get("experiment_id") != expected_experiment_id:
        raise ResponseExperimentValidationError(
            "resume refused: artifact experiment_id differs from frozen inputs"
        )
    records = artifact.get("records")
    if not isinstance(records, list):
        raise ResponseExperimentValidationError("artifact.records must be an array")
    ids = [str(record.get("id") or "") for record in records if isinstance(record, Mapping)]
    if len(ids) != len(records) or len(set(ids)) != len(ids):
        raise ResponseExperimentValidationError(
            "resume artifact requires unique non-empty record ids"
        )
    return artifact


def _failure(stage: str, exc: Exception) -> dict[str, Any]:
    return {
        "stage": stage,
        "code": f"{stage}_error",
        "error_type": type(exc).__name__,
        "message": (str(exc).strip() or repr(exc))[:500],
    }


def _judge_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return only score-affecting Judge fields with stable ordering."""

    return {
        field: metadata[field]
        for field in JUDGE_IDENTITY_FIELDS
        if field in metadata
    }


def _assert_judge_identity_compatible(
    existing: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    """Fail closed before resuming scores produced by another Judge bundle.

    Older v1 artifacts do not contain every newly-recorded field.  They remain
    resumable only when every identity field they *do* contain agrees with the
    current bundle.  New artifacts persist the full current identity.
    """

    mismatches: dict[str, dict[str, Any]] = {}
    for field in JUDGE_IDENTITY_FIELDS:
        if field not in existing:
            continue
        if field not in current or existing[field] != current[field]:
            mismatches[field] = {
                "existing": existing.get(field),
                "current": current.get(field),
            }
    if mismatches:
        raise ResponseExperimentValidationError(
            "judge resume refused: score-affecting Judge configuration differs: "
            + _canonical_json(mismatches)
        )


def _has_existing_judge_state(artifact: Mapping[str, Any]) -> bool:
    records = artifact.get("records")
    if not isinstance(records, list):
        return False
    return any(
        isinstance(record, Mapping)
        and (
            record.get("judge_status") not in {None, "not_run"}
            or any(
                value is not None
                for value in (record.get("metrics") or {}).values()
            )
            or bool(record.get("judge_attempts"))
        )
        for record in records
    )


def _empty_metrics() -> dict[str, None]:
    return {name: None for name in RESPONSE_METRICS}


def _empty_deterministic_retrieval_metrics() -> dict[str, None]:
    return {name: None for name in DETERMINISTIC_RETRIEVAL_METRICS}


def _generate_record(
    service: EngineeringRAGService,
    sample: ResponseEvaluationSample,
    *,
    top_k: int,
) -> dict[str, Any]:
    """Generate one record without reading or passing the reference answer."""

    record: dict[str, Any] = {
        "id": sample.id,
        "question": sample.question,
        "answerable": sample.answerable,
        "expected_route": sample.expected_route,
        "response": None,
        "refused": False,
        "generation_succeeded": False,
        "retrieval_latency_ms": None,
        "generation_latency_ms": None,
        "total_latency_ms": None,
        "retrieval": None,
        "retrieved_evidence": [],
        "generation_context": [],
        "answer_citations": [],
        "generation_status": None,
        "generation_usage": None,
        "metrics": _empty_metrics(),
        "deterministic_retrieval_metrics": (
            _empty_deterministic_retrieval_metrics()
        ),
        "deterministic_retrieval_metric_status": "not_run",
        "deterministic_retrieval_metric_error": None,
        "metric_errors": [],
        "judge_status": "not_run",
        "failure": None,
    }
    total_started = perf_counter()
    retrieval_started = perf_counter()
    try:
        retrieval = service.retrieve(sample.question, top_k=top_k)
    except Exception as exc:
        record["retrieval_latency_ms"] = round(
            (perf_counter() - retrieval_started) * 1000, 3
        )
        record["total_latency_ms"] = round(
            (perf_counter() - total_started) * 1000, 3
        )
        record["failure"] = _failure("retrieval", exc)
        return record
    record["retrieval_latency_ms"] = round(
        (perf_counter() - retrieval_started) * 1000, 3
    )
    record["retrieval"] = retrieval.to_dict(include_content=True)

    generation_started = perf_counter()
    generator = service.answerer.generator
    clear_last_usage = getattr(generator, "clear_last_usage", None)
    if callable(clear_last_usage):
        # A route-level refusal can return before the generator's ``__call__``.
        # Clear only generators that explicitly expose the safe reset contract;
        # do not mutate arbitrary custom/fake generator state.
        clear_last_usage()
    try:
        answer = service.answerer.answer(retrieval)
    except Exception as exc:
        record["generation_latency_ms"] = round(
            (perf_counter() - generation_started) * 1000, 3
        )
        record["total_latency_ms"] = round(
            (perf_counter() - total_started) * 1000, 3
        )
        record["failure"] = _failure("generation", exc)
        return record
    payload = answer.to_dict()
    generation_status = payload.get("generation") or {}
    generator_usage = getattr(generator, "last_usage", None)
    if (
        bool(generation_status.get("attempted"))
        and isinstance(generator_usage, Mapping)
        and generator_usage
    ):
        record["generation_usage"] = {
            str(name): int(value)
            for name, value in generator_usage.items()
            if type(value) is int and value >= 0
        } or None
    record["generation_latency_ms"] = round(
        (perf_counter() - generation_started) * 1000, 3
    )
    record["total_latency_ms"] = round(
        (perf_counter() - total_started) * 1000, 3
    )
    record.update(
        {
            "response": payload["answer"],
            "refused": bool(payload["refused"]),
            "predicted_route": payload["intent"],
            "generation_succeeded": bool(payload["generation"]["succeeded"]),
            "retrieved_evidence": payload["retrieved_evidence"],
            "generation_context": payload["generation_context"],
            "answer_citations": payload["answer_citations"],
            "generation_status": payload["generation"],
            "warnings": payload["warnings"],
            "refusal_reason": payload["refusal_reason"],
        }
    )
    return record


def generate_response_records(
    *,
    service: EngineeringRAGService,
    samples: Sequence[ResponseEvaluationSample],
    output_path: str | Path,
    metadata: Mapping[str, Any],
    top_k: int,
    resume: bool = False,
) -> dict[str, Any]:
    """Generate and checkpoint records atomically after every question."""

    path = Path(output_path)
    artifact = _load_resume_artifact(
        path,
        expected_experiment_id=str(metadata["experiment_id"]),
        resume=resume,
    )
    if artifact["metadata"] and artifact.get("stage") not in {
        "initialized",
        "generating",
        "generated",
    }:
        raise ResponseExperimentValidationError(
            "generation resume refuses an artifact that already entered judging"
        )
    if not artifact["metadata"]:
        artifact["metadata"] = dict(metadata)
    completed = {str(record["id"]) for record in artifact["records"]}
    known_ids = {sample.id for sample in samples}
    if not completed.issubset(known_ids):
        raise ResponseExperimentValidationError(
            "resume artifact contains ids outside the frozen dataset"
        )
    artifact["stage"] = "generating"
    _atomic_write_json(path, artifact)
    for sample in samples:
        if sample.id in completed:
            continue
        artifact["records"].append(
            _generate_record(service, sample, top_k=top_k)
        )
        artifact["updated_at"] = _utc_now()
        _atomic_write_json(path, artifact)
    artifact["stage"] = "generated"
    artifact["updated_at"] = _utc_now()
    _atomic_write_json(path, artifact)
    return artifact


def _contexts(record: Mapping[str, Any], field_name: str) -> list[str]:
    raw = record.get(field_name)
    if not isinstance(raw, list):
        raise ResponseExperimentValidationError(f"record.{field_name} must be an array")
    contexts: list[str] = []
    for item in raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("content"), str):
            raise ResponseExperimentValidationError(
                f"record.{field_name} entries require string content"
            )
        contexts.append(str(item["content"]))
    return contexts


def _observations(
    report: RAGASEvaluationReport,
) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    if len(report.samples) != 1:
        raise ResponseExperimentValidationError(
            "per-record RAGAS evaluation must return exactly one sample"
        )
    scores: dict[str, float | None] = {}
    errors: list[dict[str, Any]] = []
    for name, observation in report.samples[0].metrics.items():
        scores[name] = observation.score
        if observation.score is None:
            errors.append(
                {
                    "metric": name,
                    "code": "ragas_metric_error",
                    "error_type": observation.error_type,
                    "message": observation.error_message,
                }
            )
    return scores, errors


def _retrieved_sources(record: Mapping[str, Any]) -> list[str]:
    raw = record.get("retrieved_evidence")
    if not isinstance(raw, list):
        raise ResponseExperimentValidationError(
            "record.retrieved_evidence must be an array"
        )
    sources: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ResponseExperimentValidationError(
                "record.retrieved_evidence entries must be objects"
            )
        source = item.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ResponseExperimentValidationError(
                "record.retrieved_evidence entries require a non-empty source"
            )
        identity = _logical_source_identity(source)
        if identity in seen:
            continue
        seen.add(identity)
        sources.append(source)
    return sources


def _logical_source_identity(source: str) -> str:
    """Return the logical source used by deterministic response metrics.

    Repository evidence may carry a locator such as ``#symbol:Class.method``.
    That locator identifies a region in the same logical file, so it is removed
    for source-level matching and deduplication.  URL fragments are deliberately
    retained: an HTML anchor can identify a semantically distinct official
    specification section and must not be silently collapsed.
    """

    text = source.strip().replace("\\", "/")
    parsed = urlsplit(text)
    if parsed.scheme.casefold() in {"http", "https"}:
        path = parsed.path.rstrip("/") or "/"
        return urlunsplit(
            (
                parsed.scheme.casefold(),
                parsed.netloc.casefold(),
                path,
                parsed.query,
                parsed.fragment,
            )
        )
    file_path = text.partition("#")[0]
    while file_path.startswith("./"):
        file_path = file_path[2:]
    parts = [
        part for part in PurePosixPath(file_path).parts if part not in {"/", "."}
    ]
    return "/".join(parts).casefold()


def _logical_source_matches(expected: str, actual: str) -> bool:
    """Match one acceptable truth source to a retrieved logical source."""

    expected_identity = _logical_source_identity(expected)
    actual_identity = _logical_source_identity(actual)
    if expected_identity == actual_identity:
        return True
    if expected_identity.startswith(("http://", "https://")):
        return False
    return actual_identity.endswith("/" + expected_identity)


def _score_deterministic_retrieval(
    record: dict[str, Any], sample: ResponseEvaluationSample
) -> None:
    """Score source retrieval after references are joined in the judge stage.

    The generation stage never calls this function.  It therefore cannot leak
    reference evidence paths into model context or answer generation.
    """

    record["deterministic_retrieval_metrics"] = (
        _empty_deterministic_retrieval_metrics()
    )
    record["deterministic_retrieval_metric_error"] = None
    if not sample.answerable:
        record["deterministic_retrieval_metric_status"] = "not_applicable"
        return
    if (record.get("failure") or {}).get("stage") == "retrieval":
        record["deterministic_retrieval_metric_status"] = "failed"
        record["deterministic_retrieval_metric_error"] = {
            "stage": "deterministic_retrieval_metric",
            "code": "retrieval_failed",
            "error_type": "RetrievalFailure",
            "message": "retrieval failed before a ranked evidence list was produced",
        }
        return
    try:
        required_claims = [
            claim for claim in sample.reference_claims if claim.required
        ]
        if not required_claims:
            raise ResponseExperimentValidationError(
                "answerable sample requires at least one required reference claim"
            )
        if any(not claim.evidence_sources for claim in required_claims):
            raise ResponseExperimentValidationError(
                "every required claim requires at least one acceptable evidence source"
            )
        retrieved_sources = _retrieved_sources(record)
    except Exception as exc:
        record["deterministic_retrieval_metric_status"] = "failed"
        record["deterministic_retrieval_metric_error"] = _failure(
            "deterministic_retrieval_metric", exc
        )
        return

    required_source_options = [
        source
        for claim in required_claims
        for source in claim.evidence_sources
    ]
    option_identities = {
        _logical_source_identity(source) for source in required_source_options
    }
    matched_option_identities = {
        _logical_source_identity(expected)
        for expected in required_source_options
        if any(
            _logical_source_matches(expected, actual)
            for actual in retrieved_sources
        )
    }
    matched_required_claims = [
        claim
        for claim in required_claims
        if any(
            _logical_source_matches(expected, actual)
            for expected in claim.evidence_sources
            for actual in retrieved_sources
        )
    ]
    relevant_results = [
        actual
        for actual in retrieved_sources
        if any(
            _logical_source_matches(expected, actual)
            for expected in required_source_options
        )
    ]
    reciprocal_rank = 0.0
    for rank, actual in enumerate(retrieved_sources, start=1):
        if any(
            _logical_source_matches(expected, actual)
            for expected in required_source_options
        ):
            reciprocal_rank = 1.0 / rank
            break
    record["deterministic_retrieval_metrics"] = {
        "hit_at_k": float(bool(matched_required_claims)),
        "required_claim_recall_at_k": (
            len(matched_required_claims) / len(required_claims)
        ),
        "mrr": reciprocal_rank,
        "source_precision_at_k": (
            len(relevant_results) / len(retrieved_sources)
            if retrieved_sources
            else 0.0
        ),
        "source_option_recall": (
            len(matched_option_identities) / len(option_identities)
        ),
    }
    record["deterministic_retrieval_metric_status"] = "complete"


def _atomic_write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a complete JSON file without replacing a prior file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ResponseExperimentValidationError(
                f"deterministic rescore refuses to overwrite: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def recompute_deterministic_retrieval_metrics(
    *,
    artifact_path: str | Path,
    dataset_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Recompute only deterministic retrieval metrics on a judged artifact.

    This path intentionally has no service, generator, embedding, or RAGAS
    dependency.  References are joined only after validating the frozen formal
    dataset against artifact identity and per-record labels.
    """

    source_path = Path(artifact_path)
    output = Path(output_path)
    if output.exists():
        raise ResponseExperimentValidationError(
            f"deterministic rescore refuses to overwrite: {output}"
        )
    artifact = dict(_load_json(source_path, "judged response artifact"))
    if artifact.get("schema_version") != ARTIFACT_SCHEMA:
        raise ResponseExperimentValidationError(
            "unsupported response artifact schema for deterministic rescore"
        )
    if artifact.get("stage") != "judged":
        raise ResponseExperimentValidationError(
            "deterministic rescore requires a stage='judged' artifact"
        )
    metadata = dict(_require_mapping(artifact.get("metadata"), "artifact.metadata"))
    samples = load_response_jsonl(dataset_path, formal=True)
    exact_dataset_sha = file_sha256(dataset_path)
    canonical_dataset_sha = dataset_sha256(samples)
    if metadata.get("dataset_file_sha256") != exact_dataset_sha:
        raise ResponseExperimentValidationError(
            "formal dataset file SHA does not match artifact metadata"
        )
    if metadata.get("dataset_canonical_sha256") != canonical_dataset_sha:
        raise ResponseExperimentValidationError(
            "formal dataset canonical SHA does not match artifact metadata"
        )
    raw_records = artifact.get("records")
    if not isinstance(raw_records, list):
        raise ResponseExperimentValidationError("artifact.records must be an array")
    records_by_id: dict[str, dict[str, Any]] = {}
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            raise ResponseExperimentValidationError(
                "artifact.records entries must be objects"
            )
        sample_id = str(raw_record.get("id") or "")
        if not sample_id or sample_id in records_by_id:
            raise ResponseExperimentValidationError(
                "artifact requires unique non-empty record ids"
            )
        records_by_id[sample_id] = dict(raw_record)
    if set(records_by_id) != {sample.id for sample in samples}:
        raise ResponseExperimentValidationError(
            "artifact ids must exactly match the formal dataset ids"
        )
    rescored_records: list[dict[str, Any]] = []
    for sample in samples:
        record = records_by_id[sample.id]
        if record.get("question") != sample.question:
            raise ResponseExperimentValidationError(
                f"question text differs for sample {sample.id!r}"
            )
        if record.get("answerable") is not sample.answerable:
            raise ResponseExperimentValidationError(
                f"answerable label differs for sample {sample.id!r}"
            )
        _score_deterministic_retrieval(record, sample)
        rescored_records.append(record)

    rescored_at = _utc_now()
    metadata["deterministic_retrieval_rescore"] = {
        "schema_version": DETERMINISTIC_RETRIEVAL_METRIC_SCHEMA,
        "rescored_at": rescored_at,
        "source_artifact_sha256": file_sha256(source_path),
        "formal_dataset_file_sha256": exact_dataset_sha,
        "formal_dataset_canonical_sha256": canonical_dataset_sha,
        "network_or_model_calls": False,
    }
    artifact["metadata"] = metadata
    artifact["records"] = rescored_records
    artifact["updated_at"] = rescored_at
    _atomic_write_json_new(output, artifact)
    return artifact


def _normalise_judge_audit(record: dict[str, Any]) -> None:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        metrics = {}
    record["metrics"] = {
        name: metrics.get(name) for name in RESPONSE_METRICS
    }
    current_errors = record.get("metric_errors")
    if not isinstance(current_errors, list):
        current_errors = []
    record["metric_errors"] = [
        dict(error) for error in current_errors if isinstance(error, Mapping)
    ]
    if not isinstance(record.get("judge_attempts"), list):
        record["judge_attempts"] = []
    if not isinstance(record.get("error_history"), list):
        # Preserve legacy failures before any retry can resolve them.
        record["error_history"] = [
            {**dict(error), "historical": True}
            for error in record["metric_errors"]
        ]


def _unresolved_metric_errors(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    metrics = record.get("metrics") or {}
    unresolved: list[dict[str, Any]] = []
    for raw_error in record.get("metric_errors") or ():
        if not isinstance(raw_error, Mapping):
            continue
        error = dict(raw_error)
        metric = error.get("metric")
        if isinstance(metric, str):
            if metrics.get(metric) is None:
                unresolved.append(error)
            continue
        stage = str(error.get("stage") or "")
        if stage == "retrieval_judge" and any(
            metrics.get(name) is None for name in RETRIEVAL_METRICS
        ):
            unresolved.append(error)
        elif stage == "response_judge" and any(
            metrics.get(name) is None for name in ANSWER_METRICS
        ):
            unresolved.append(error)
        elif stage not in {"retrieval_judge", "response_judge"}:
            unresolved.append(error)
    return unresolved


def _evaluate_one_metric(
    evaluator: RAGASEvaluator,
    sample: RAGASSample,
    metric_name: str,
) -> tuple[float | None, dict[str, Any] | None]:
    report = evaluator.evaluate_samples([sample], metric_names=[metric_name])
    scores, errors = _observations(report)
    if set(scores) != {metric_name}:
        raise ResponseExperimentValidationError(
            f"single-metric Judge returned unexpected metrics for {metric_name!r}"
        )
    error = errors[0] if errors else None
    return scores[metric_name], error


def _judge_record(
    record: dict[str, Any],
    sample: ResponseEvaluationSample,
    judges: RAGASJudgeBundle,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> None:
    _normalise_judge_audit(record)
    _score_deterministic_retrieval(record, sample)
    if (
        not sample.answerable
        or record.get("refused")
        or not record.get("generation_succeeded")
        or not isinstance(record.get("response"), str)
        or not str(record.get("response")).strip()
    ):
        record["judge_status"] = "not_eligible"
        if checkpoint is not None:
            checkpoint()
        return
    try:
        raw_contexts = _contexts(record, "retrieved_evidence")
        generation_contexts = _contexts(record, "generation_context")
    except Exception as exc:
        error = _failure("judge_input", exc)
        record["metric_errors"] = [error]
        record["error_history"].append({**error, "observed_at": _utc_now()})
        record["judge_status"] = "failed"
        if checkpoint is not None:
            checkpoint()
        return

    retrieval_sample = RAGASSample(
        user_input=sample.question,
        retrieved_contexts=raw_contexts,
        response=str(record["response"]),
        reference=sample.reference_answer,
    )
    response_sample = RAGASSample(
        user_input=sample.question,
        retrieved_contexts=generation_contexts,
        response=str(record["response"]),
        reference=sample.reference_answer,
    )
    metric_jobs = (
        *((name, judges.retrieval, retrieval_sample) for name in RETRIEVAL_METRICS),
        *((name, judges.response, response_sample) for name in ANSWER_METRICS),
    )
    for metric_name, evaluator, ragas_sample in metric_jobs:
        # A persisted numeric score is immutable under resume.  Only null/error
        # metric slots are eligible for another paid call.
        if record["metrics"].get(metric_name) is not None:
            continue
        record["metric_errors"] = [
            error
            for error in record["metric_errors"]
            if error.get("metric") != metric_name
        ]
        attempt_number = 1 + sum(
            attempt.get("metric") == metric_name
            for attempt in record["judge_attempts"]
            if isinstance(attempt, Mapping)
        )
        attempted_at = _utc_now()
        score: float | None = None
        error: dict[str, Any] | None = None
        try:
            score, error = _evaluate_one_metric(
                evaluator, ragas_sample, metric_name
            )
        except Exception as exc:
            error = {
                **_failure("judge_metric", exc),
                "metric": metric_name,
            }
        if error is not None:
            error = {**error, "metric": metric_name}
            record["metric_errors"].append(error)
            record["error_history"].append(
                {
                    **error,
                    "attempt": attempt_number,
                    "observed_at": _utc_now(),
                }
            )
        else:
            record["metrics"][metric_name] = score
        record["judge_attempts"].append(
            {
                "metric": metric_name,
                "attempt": attempt_number,
                "attempted_at": attempted_at,
                "completed_at": _utc_now(),
                "status": "ok" if error is None else "error",
                "score": score,
                "error": error,
            }
        )
        record["metric_errors"] = _unresolved_metric_errors(record)
        record["judge_status"] = (
            "complete"
            if all(record["metrics"].get(name) is not None for name in RESPONSE_METRICS)
            else "partial_or_failed"
        )
        if checkpoint is not None:
            checkpoint()


_GENERATION_IDENTITY_FIELDS = (
    "id",
    "question",
    "answerable",
    "expected_route",
    "response",
    "refused",
    "generation_succeeded",
    "retrieval",
    "retrieved_evidence",
    "generation_context",
    "answer_citations",
    "generation_status",
    "failure",
)


def _generation_record_identity(record: Mapping[str, Any]) -> str:
    return _sha256_text(
        _canonical_json(
            {field: record.get(field) for field in _GENERATION_IDENTITY_FIELDS}
        )
    )


def _validate_retry_records(
    state_records: Mapping[str, Mapping[str, Any]],
    replay_records: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(state_records) != set(replay_records):
        raise ResponseExperimentValidationError(
            "judge retry state ids differ from generation replay ids"
        )
    changed = [
        sample_id
        for sample_id in state_records
        if _generation_record_identity(state_records[sample_id])
        != _generation_record_identity(replay_records[sample_id])
    ]
    if changed:
        raise ResponseExperimentValidationError(
            "judge retry state changes frozen generation records: "
            + ", ".join(changed[:5])
        )


def judge_response_records(
    *,
    replay_path: str | Path,
    output_path: str | Path,
    samples: Sequence[ResponseEvaluationSample],
    metadata: Mapping[str, Any],
    judges: RAGASJudgeBundle,
    resume: bool = False,
    retry_from_path: str | Path | None = None,
) -> dict[str, Any]:
    """Judge persisted answers without regenerating them."""

    replay = dict(_load_json(replay_path, "generation replay artifact"))
    if replay.get("schema_version") != ARTIFACT_SCHEMA:
        raise ResponseExperimentValidationError("unsupported replay artifact schema")
    replay_meta = _require_mapping(replay.get("metadata"), "replay.metadata")
    if replay_meta.get("experiment_id") != metadata.get("experiment_id"):
        raise ResponseExperimentValidationError(
            "judge replay does not match the frozen experiment identity"
        )
    raw_records = replay.get("records")
    if not isinstance(raw_records, list):
        raise ResponseExperimentValidationError("replay.records must be an array")
    records_by_id = {
        str(record.get("id") or ""): dict(record)
        for record in raw_records
        if isinstance(record, Mapping)
    }
    expected_ids = {sample.id for sample in samples}
    if set(records_by_id) != expected_ids:
        raise ResponseExperimentValidationError(
            "judge replay must contain exactly one record for every frozen sample"
        )

    output = Path(output_path)
    current_judge_metadata = dict(judges.metadata)
    current_judge_metadata["identity_sha256"] = _sha256_text(
        _canonical_json(_judge_identity(current_judge_metadata))
    )
    current_judge_metadata["audit_schema"] = JUDGE_AUDIT_SCHEMA
    judged_metadata = {**dict(metadata), "judge": current_judge_metadata}
    if retry_from_path is not None:
        retry_from = Path(retry_from_path)
        if output.exists():
            raise ResponseExperimentValidationError(
                "--retry-from requires a new, non-existing output artifact"
            )
        if retry_from.resolve() == output.resolve():
            raise ResponseExperimentValidationError(
                "--retry-from and output must be different paths"
            )
        artifact = dict(_load_json(retry_from, "judge retry artifact"))
        if artifact.get("schema_version") != ARTIFACT_SCHEMA:
            raise ResponseExperimentValidationError(
                "unsupported judge retry artifact schema"
            )
        retry_meta = _require_mapping(artifact.get("metadata"), "retry.metadata")
        if retry_meta.get("experiment_id") != metadata.get("experiment_id"):
            raise ResponseExperimentValidationError(
                "judge retry artifact does not match the frozen experiment identity"
            )
        existing_judge = retry_meta.get("judge")
        if not isinstance(existing_judge, Mapping):
            raise ResponseExperimentValidationError(
                "judge retry artifact is missing metadata.judge"
            )
        _assert_judge_identity_compatible(existing_judge, current_judge_metadata)
        raw_retry_records = artifact.get("records")
        if not isinstance(raw_retry_records, list):
            raise ResponseExperimentValidationError(
                "judge retry artifact records must be an array"
            )
        retry_records = {
            str(record.get("id") or ""): dict(record)
            for record in raw_retry_records
            if isinstance(record, Mapping)
        }
        _validate_retry_records(retry_records, records_by_id)
        records_by_id = retry_records
        judged_metadata["judge_retry"] = {
            "source_sha256": file_sha256(retry_from),
            "resumed_at": _utc_now(),
            "source_stage": artifact.get("stage"),
        }
    elif output.exists():
        artifact = _load_resume_artifact(
            output,
            expected_experiment_id=str(metadata["experiment_id"]),
            resume=resume,
        )
        existing_meta = _require_mapping(
            artifact.get("metadata"), "artifact.metadata"
        )
        existing_judge = existing_meta.get("judge")
        if _has_existing_judge_state(artifact):
            if not isinstance(existing_judge, Mapping):
                raise ResponseExperimentValidationError(
                    "judge resume artifact has scores but no metadata.judge"
                )
            _assert_judge_identity_compatible(
                existing_judge, current_judge_metadata
            )
        existing_by_id = {str(item["id"]): item for item in artifact["records"]}
        _validate_retry_records(existing_by_id, records_by_id)
        for sample in samples:
            if sample.id in existing_by_id:
                records_by_id[sample.id] = dict(existing_by_id[sample.id])
    else:
        artifact = _new_artifact(judged_metadata)
    artifact["metadata"] = {
        **dict(artifact.get("metadata") or {}),
        **judged_metadata,
    }
    artifact["records"] = [records_by_id[sample.id] for sample in samples]
    artifact["stage"] = "judging"
    _atomic_write_json(output, artifact)
    for sample, record in zip(samples, artifact["records"]):
        _normalise_judge_audit(record)
        metric_complete = all(
            record["metrics"].get(name) is not None for name in RESPONSE_METRICS
        )
        semantically_ineligible = (
            not sample.answerable
            or record.get("refused")
            or not record.get("generation_succeeded")
            or not isinstance(record.get("response"), str)
            or not str(record.get("response")).strip()
        )
        if metric_complete or semantically_ineligible:
            if semantically_ineligible:
                _score_deterministic_retrieval(record, sample)
                record["judge_status"] = "not_eligible"
            continue
        def checkpoint() -> None:
            artifact["updated_at"] = _utc_now()
            _atomic_write_json(output, artifact)

        _judge_record(record, sample, judges, checkpoint=checkpoint)
        checkpoint()
    artifact["stage"] = "judged"
    artifact["updated_at"] = _utc_now()
    _atomic_write_json(output, artifact)
    return artifact


def run_response_experiment(
    *,
    dataset_path: str | Path,
    snapshot_path: str | Path,
    manifest_path: str | Path,
    index_root: str | Path,
    mini_nanobot_repo: str | Path,
    output_path: str | Path,
    profile: RetrievalExperimentProfile,
    sufficiency_profile: SufficiencyProfile | str = (
        SufficiencyProfile.LEGACY_EXACT_SLASH
    ),
    support_selection_profile: SupportSelectionProfile | str = (
        SupportSelectionProfile.LEGACY_FIRST
    ),
    top_k: int = 5,
    generate_only: bool = False,
    judge_only: bool = False,
    replay_path: str | Path | None = None,
    resume: bool = False,
    retry_from_path: str | Path | None = None,
    service: EngineeringRAGService | None = None,
    judges: RAGASJudgeBundle | None = None,
) -> dict[str, Any]:
    """Run or replay one response experiment under frozen inputs."""

    if generate_only and judge_only:
        raise ValueError("generate_only and judge_only are mutually exclusive")
    if not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50")
    frozen = validate_frozen_response_inputs(
        dataset_path=dataset_path,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        index_root=index_root,
    )
    selected_sufficiency_profile = SufficiencyProfile(sufficiency_profile)
    selected_support_selection_profile = SupportSelectionProfile(
        support_selection_profile
    )
    if judge_only:
        replay_source = replay_path or retry_from_path
        if replay_source is None:
            raise ValueError("judge_only requires replay_path or retry_from_path")
        metadata = _experiment_metadata(
            frozen=frozen,
            profile=profile,
            sufficiency_profile=selected_sufficiency_profile,
            support_selection_profile=selected_support_selection_profile,
            top_k=top_k,
            service=None,
        )
        replay = _load_json(replay_source, "generation replay artifact")
        replay_metadata = _require_mapping(replay.get("metadata"), "replay.metadata")
        # The generation identity includes generator metadata. Preserve it while
        # still validating every frozen, profile and code field we can derive.
        expected_without_generator = {
            key: value for key, value in metadata.items() if key != "experiment_id"
        }
        for key, value in expected_without_generator.items():
            if replay_metadata.get(key) != value:
                raise ResponseExperimentValidationError(
                    f"judge replay metadata differs for {key}"
                )
        metadata = dict(replay_metadata)
        if judges is None:
            judges = build_deepseek_ragas_judges()
        return judge_response_records(
            replay_path=replay_source,
            output_path=output_path,
            samples=frozen.samples,
            metadata=metadata,
            judges=judges,
            resume=resume,
            retry_from_path=retry_from_path,
        )

    if service is None:
        service = build_experiment_service(
            index_root=index_root,
            manifest=frozen.manifest,
            mini_nanobot_repo=mini_nanobot_repo,
            profile=profile,
            sufficiency_profile=selected_sufficiency_profile,
            support_selection_profile=selected_support_selection_profile,
        )
    metadata = _experiment_metadata(
        frozen=frozen,
        profile=profile,
        sufficiency_profile=selected_sufficiency_profile,
        support_selection_profile=selected_support_selection_profile,
        top_k=top_k,
        service=service,
    )
    generated = generate_response_records(
        service=service,
        samples=frozen.samples,
        output_path=output_path,
        metadata=metadata,
        top_k=top_k,
        resume=resume,
    )
    if generate_only:
        return generated
    if judges is None:
        judges = build_deepseek_ragas_judges()
    # Full mode deliberately replays the just-written artifact in place.  It
    # cannot overwrite a user-supplied prior artifact because generation above
    # already enforced non-overwrite/resume identity.
    return judge_response_records(
        replay_path=output_path,
        output_path=output_path,
        samples=frozen.samples,
        metadata=metadata,
        judges=judges,
        resume=True,
        retry_from_path=retry_from_path,
    )
