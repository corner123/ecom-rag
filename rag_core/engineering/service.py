"""Source routing, hybrid retrieval and live implementation verification."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
from typing import Iterable

from rag_core.retrieval.engineering import (
    EngineeringSearchResult,
    EngineeringCrossEncoderReranker,
    FederatedRetriever,
    LiveASTRetriever,
    LiveCodeRetriever,
    LiveGitVerifier,
    SourceIntent,
    SourceIntentRouter,
    rrf_fusion,
)

from .grounding import GroundedAnswerer
from .index import EngineeringIndex
from .models import AnswerOutcome, EvidenceCitation, RetrievalOutcome
from .sufficiency import EvidenceSufficiencyGuard


_BACKTICK_RE = re.compile(r"`([^`\r\n]{1,128})`")
_IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|"
    r"[A-Z][A-Za-z0-9_]{2,}|[a-z][a-z0-9]*_[a-z0-9_]+)\b"
)
_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]{5,}\b")
_STOPWORDS = {
    "current", "source", "implementation", "implemented", "function", "method",
    "class", "module", "project", "nanobot", "official", "documentation",
    "where", "which", "compare", "difference", "between",
}


class EngineeringRAGService:
    """The application boundary used by CLI, API and ``knowledge.search``."""

    def __init__(
        self,
        retriever: FederatedRetriever,
        *,
        router: SourceIntentRouter | None = None,
        live_code: LiveCodeRetriever | None = None,
        live_ast: LiveASTRetriever | None = None,
        live_git: LiveGitVerifier | None = None,
        reranker: EngineeringCrossEncoderReranker | None = None,
        answerer: GroundedAnswerer | None = None,
        index_stats: dict | None = None,
        manifest_path: str | Path | None = None,
        sufficiency_guard: EvidenceSufficiencyGuard | None = None,
    ) -> None:
        self.retriever = retriever
        self.router = router or SourceIntentRouter()
        self.live_code = live_code
        self.live_ast = live_ast
        self.live_git = live_git
        self.reranker = reranker
        self.answerer = answerer or GroundedAnswerer()
        self.index_stats = dict(index_stats or {})
        self.manifest_path = Path(manifest_path).resolve() if manifest_path else None
        self.sufficiency_guard = sufficiency_guard or EvidenceSufficiencyGuard()

    @classmethod
    def from_index(
        cls,
        index_root: str | Path,
        *,
        mini_nanobot_repo: str | Path | None = None,
        manifest_path: str | Path | None = None,
        embedding_manager=None,
        answerer: GroundedAnswerer | None = None,
        runtime_backend: str | None = None,
        milvus_settings=None,
        milvus_client_factory=None,
    ) -> "EngineeringRAGService":
        if answerer is None:
            from .deepseek_generation import build_grounded_answerer_from_env

            answerer = build_grounded_answerer_from_env()
        engineering_index = EngineeringIndex.load(
            index_root,
            embedding_manager=embedding_manager,
            runtime_backend=runtime_backend,
            milvus_settings=milvus_settings,
            milvus_client_factory=milvus_client_factory,
        )
        repo = mini_nanobot_repo or os.getenv("MINI_NANOBOT_REPO")
        live = (
            LiveCodeRetriever(
                repo,
                corpus="internal",
                authority="code",
                case_sensitive=True,
            )
            if repo and Path(repo).expanduser().is_dir()
            else None
        )
        live_ast = LiveASTRetriever(repo) if live is not None else None
        live_git = LiveGitVerifier(repo) if live is not None else None
        reranker = None
        if os.getenv("ENGINEERING_RERANK_ENABLED", "").strip().casefold() in {
            "1", "true", "yes", "on"
        }:
            reranker = EngineeringCrossEncoderReranker(
                os.getenv("RERANKER_MODEL_PATH", "BAAI/bge-reranker-base")
            )
        index_stats = engineering_index.stats()
        if manifest_path is not None:
            manifest_build_id = _read_manifest_build_id(manifest_path)
            index_stats["manifest_build_id"] = manifest_build_id
            index_stats["fresh"] = bool(
                manifest_build_id
                and manifest_build_id == index_stats.get("build_id")
            )
        return cls(
            engineering_index.federated,
            live_code=live,
            live_ast=live_ast,
            live_git=live_git,
            reranker=reranker,
            answerer=answerer,
            index_stats=index_stats,
            manifest_path=manifest_path,
        )

    def health(self) -> dict:
        revision = self._live_revision() if self.live_git is not None else {}
        index_stats = dict(self.index_stats)
        if self.manifest_path is not None:
            manifest_build_id = _read_manifest_build_id(self.manifest_path)
            index_stats["manifest_build_id"] = manifest_build_id
            index_stats["fresh"] = bool(
                manifest_build_id
                and manifest_build_id == index_stats.get("build_id")
            )
        fresh = index_stats.get("fresh")
        public_index = {
            key: value for key, value in index_stats.items() if key != "index_root"
        }
        return {
            "status": "ok" if fresh is not False else "stale_index",
            "index_fresh": fresh,
            "index": public_index,
            "partition_count": len(self.retriever.partitions),
            "live_code_enabled": self.live_code is not None,
            "live_ast_enabled": self.live_ast is not None,
            "live_git_enabled": self.live_git is not None,
            "reranker_enabled": self.reranker is not None,
            "answer_provider": self.answerer.provider,
            "answer_model": self.answerer.model,
            "model_generation_enabled": self.answerer.generator is not None,
            "live_repo": self.live_code.repo_root.name if self.live_code else None,
            "live_revision": revision,
        }

    def retrieve(self, query: str, *, top_k: int = 5) -> RetrievalOutcome:
        query = self._validate_query(query)
        retrieval_query, normalization_warning = self.router.normalize_query(query)
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        route = self.router.route(retrieval_query)
        if route.intent is SourceIntent.OUT_OF_SCOPE:
            return RetrievalOutcome(
                query=query,
                intent=route.intent,
                results=[],
                citations=[],
                sufficient_evidence=False,
                refusal_reason=self.router.refusal_reason(query),
                warnings=[
                    *([normalization_warning] if normalization_warning else []),
                    "query is outside the configured engineering knowledge scope",
                ],
                matched_rule=route.matched_rule,
            )

        indexed = self.retriever.search(
            retrieval_query,
            top_k=max(top_k, top_k * 2),
            corpora=route.corpora,
            authorities=route.authorities,
        )
        indexed = [self._stamp_evidence_role(result) for result in indexed]
        supporting_candidates: list[EngineeringSearchResult] = []
        if route.intent is SourceIntent.DESIGN:
            supporting_candidates = self.retriever.search(
                retrieval_query,
                top_k=max(2, top_k),
                corpora=("internal",),
                authorities=("code", "test"),
            )
        elif route.intent is SourceIntent.IMPLEMENTATION:
            supporting_candidates = self.retriever.search(
                retrieval_query,
                top_k=max(2, top_k),
                corpora=("internal",),
                authorities=("design", "history"),
            )
        supporting_candidates = [
            self._stamp_evidence_role(result) for result in supporting_candidates
        ]
        if route.intent is SourceIntent.COMPARISON:
            # Cross-corpus RRF may otherwise fill a small candidate window
            # with higher-weight internal code before an official item is
            # visible. Always retrieve the normative side explicitly; the
            # final evidence contract still requires live implementation too.
            official_candidates = self.retriever.search(
                retrieval_query,
                top_k=max(2, top_k),
                corpora=("official",),
                authorities=("official",),
            )
            existing_ids = {result.result_id for result in indexed}
            indexed.extend(
                self._stamp_evidence_role(result)
                for result in official_candidates
                if result.result_id not in existing_ids
            )
        if self.reranker is not None and indexed:
            indexed = self.reranker.rerank(
                retrieval_query,
                indexed,
                top_k=max(top_k, top_k * 2),
            )

        live_attempted = route.intent in {SourceIntent.IMPLEMENTATION, SourceIntent.COMPARISON}
        live_revision = self._live_revision() if live_attempted else {}
        terms: tuple[str, ...] = ()
        live_results: list[EngineeringSearchResult] = []
        if live_attempted and (self.live_code is not None or self.live_ast is not None):
            terms = tuple(self._verification_terms(retrieval_query, indexed))
            live_results = self._search_live_terms(
                terms,
                top_k=max(top_k, 8),
                revision=live_revision,
            )
        elif live_attempted:
            terms = tuple(self._verification_terms(retrieval_query, indexed))

        if live_results:
            results = rrf_fusion(
                [live_results, indexed],
                top_k=top_k,
                weights=[2.0, 1.0],
            )
            # RRF keeps the representative metadata. Reassert roles after fusion.
            results = [self._stamp_evidence_role(result) for result in results]
        else:
            results = indexed[:top_k]

        # Comparison answers need at least one result from each evidence side.
        # Preserve an official item when a live-heavy fusion would otherwise
        # crowd it out of a small top_k response.
        if route.intent is SourceIntent.COMPARISON:
            results = self._ensure_comparison_sides(results, live_results, indexed, top_k)
        elif supporting_candidates:
            results = self._ensure_internal_support(
                results, supporting_candidates, top_k
            )

        sufficient, warnings, refusal_reason = self._evidence_status(
            route.intent, results, live_attempted
        )
        if normalization_warning:
            warnings.insert(0, normalization_warning)
        if sufficient:
            guarded, guard_warnings, guard_reason = self.sufficiency_guard.check(
                retrieval_query, route.intent, results
            )
            sufficient = guarded
            warnings.extend(guard_warnings)
            if not guarded:
                refusal_reason = guard_reason
        degraded = sorted(
            {
                str(component)
                for result in results
                for component in result.metadata.get(
                    "retrieval_degraded_components", []
                )
            }
        )
        if degraded:
            warnings.append(
                "retrieval degraded; unavailable components: " + ", ".join(degraded)
            )
        citations = self._build_citations(results)
        return RetrievalOutcome(
            query=query,
            intent=route.intent,
            results=results,
            citations=citations,
            sufficient_evidence=sufficient,
            refusal_reason=refusal_reason if not sufficient else None,
            live_verification_attempted=live_attempted,
            live_verification_terms=terms,
            live_revision=live_revision,
            warnings=warnings,
            matched_rule=route.matched_rule,
        )

    def answer(self, query: str, *, top_k: int = 5) -> AnswerOutcome:
        return self.answerer.answer(self.retrieve(query, top_k=top_k))

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        value = query.strip()
        if not value:
            raise ValueError("query cannot be empty")
        if len(value) > 1_000:
            raise ValueError("query is too long")
        if "\x00" in value:
            raise ValueError("query cannot contain NUL")
        return value

    @staticmethod
    def _stamp_evidence_role(result: EngineeringSearchResult) -> EngineeringSearchResult:
        metadata = dict(result.metadata)
        if metadata.get("live_verification"):
            role = "current_implementation"
            corpus = "internal"
            authority = "code"
        elif result.corpus == "official" or result.authority == "official":
            role = "external_normative"
            corpus = "official"
            authority = "official"
        elif result.authority == "design":
            role = "internal_design"
            corpus = result.corpus
            authority = result.authority
        elif result.authority == "history":
            role = "internal_history"
            corpus = result.corpus
            authority = result.authority
        else:
            role = "indexed_implementation"
            corpus = result.corpus
            authority = result.authority
        metadata["evidence_role"] = role
        return result.updated(corpus=corpus, authority=authority, metadata=metadata)

    def _verification_terms(
        self,
        query: str,
        indexed: Iterable[EngineeringSearchResult],
    ) -> list[str]:
        quoted = [match.group(1).strip() for match in _BACKTICK_RE.finditer(query)]
        identifiers = [match.group(0) for match in _IDENTIFIER_RE.finditer(query)]
        # An identifier explicitly supplied by the caller is a verification
        # assertion.  Falling back to a parent class or a symbol suggested by
        # the vector index would turn "missing_method" into a false positive.
        direct = [*quoted, *identifiers]
        candidates: list[str] = list(direct)
        if not direct:
            candidates.extend(
                word
                for word in _WORD_RE.findall(query)
                if word.casefold() not in _STOPWORDS
            )
            # Index-derived terms are discovery hints only when the query did
            # not contain an explicit symbol.
            for result in list(indexed)[:5]:
                if result.symbol:
                    candidates.append(result.symbol)
                relative = (
                    result.metadata.get("relative_path")
                    or result.metadata.get("document_path")
                )
                if relative:
                    stem = Path(str(relative).split("#", 1)[0]).stem
                    if len(stem) >= 4 and stem.casefold() not in _STOPWORDS:
                        candidates.append(stem)

        expanded: list[str] = []
        for candidate in candidates:
            value = candidate.strip().strip("'\"")
            if not value or len(value) > 128:
                continue
            expanded.append(value)
        seen: set[str] = set()
        ordered = []
        for value in expanded:
            folded = value.casefold()
            if folded in seen or folded in _STOPWORDS:
                continue
            seen.add(folded)
            ordered.append(value)
            if len(ordered) >= 6:
                break
        return ordered

    def _search_live_terms(
        self,
        terms: Iterable[str],
        *,
        top_k: int,
        revision: dict[str, object],
    ) -> list[EngineeringSearchResult]:
        if self.live_code is None and self.live_ast is None:
            return []
        ast_collected: list[EngineeringSearchResult] = []
        rg_collected: list[EngineeringSearchResult] = []
        seen: set[str] = set()
        for term in terms:
            retrievers = (
                ("ast", self.live_ast, ast_collected),
                ("rg", self.live_code, rg_collected),
            )
            for method, retriever, target in retrievers:
                if retriever is None:
                    continue
                if len(target) >= top_k:
                    continue
                try:
                    matches = retriever.search(term, top_k=top_k)
                except (OSError, RuntimeError, TypeError, ValueError):
                    continue
                for match in matches:
                    if match.result_id in seen:
                        continue
                    seen.add(match.result_id)
                    metadata = dict(match.metadata)
                    metadata.update(
                        {
                            "evidence_role": "current_implementation",
                            "verification_term": term,
                            "verification_method": f"live_{method}",
                            "current_revision": dict(revision),
                            "git_commit_sha": revision.get("commit_sha"),
                            "git_branch": revision.get("branch"),
                            "git_dirty": revision.get("dirty"),
                        }
                    )
                    target.append(
                        match.updated(
                            corpus="internal",
                            authority="code",
                            metadata=metadata,
                        )
                    )
                    if len(target) >= top_k:
                        break
            if len(ast_collected) >= top_k and len(rg_collected) >= top_k:
                break
        if ast_collected and rg_collected:
            selected = rrf_fusion(
                [ast_collected, rg_collected],
                top_k=top_k,
                weights=[1.5, 1.0],
            )
        else:
            selected = (ast_collected or rg_collected)[:top_k]
        if self.live_git is not None:
            enriched: list[EngineeringSearchResult] = []
            path_cache: dict[str, dict[str, object]] = {}
            for index, result in enumerate(selected):
                metadata = dict(result.metadata)
                relative_path = metadata.get("relative_path")
                if relative_path and index < 3:
                    key = str(relative_path)
                    try:
                        if key not in path_cache:
                            path_cache[key] = self.live_git.path_state(
                                key, include_worktree_state=False
                            )
                        metadata["path_state"] = path_cache[key]
                    except (OSError, RuntimeError, ValueError):
                        pass
                enriched.append(result.updated(metadata=metadata))
            selected = enriched
        return selected

    def _live_revision(self) -> dict[str, object]:
        if self.live_git is not None:
            try:
                return self.live_git.state().to_dict()
            except (OSError, RuntimeError, ValueError):
                pass
        root = None
        if self.live_code is not None:
            root = self.live_code.repo_root
        elif self.live_ast is not None:
            root = self.live_ast.repo_root
        return _git_revision(root) if root is not None else {}

    @staticmethod
    def _ensure_internal_support(
        results: list[EngineeringSearchResult],
        supporting: list[EngineeringSearchResult],
        top_k: int,
    ) -> list[EngineeringSearchResult]:
        selected = list(results[:top_k])
        if top_k < 2 or not supporting:
            return selected
        candidate = next(
            (
                item
                for item in supporting
                if all(item.result_id != existing.result_id for existing in selected)
            ),
            None,
        )
        if candidate is None:
            return selected
        roles = {str(item.metadata.get("evidence_role")) for item in selected}
        candidate_role = str(candidate.metadata.get("evidence_role"))
        if candidate_role in roles:
            return selected
        return [*selected[: top_k - 1], candidate]

    @staticmethod
    def _ensure_comparison_sides(
        results: list[EngineeringSearchResult],
        live: list[EngineeringSearchResult],
        indexed: list[EngineeringSearchResult],
        top_k: int,
    ) -> list[EngineeringSearchResult]:
        selected = list(results[:top_k])
        live_item = next(
            (item for item in selected if item.metadata.get("live_verification")),
            live[0] if live else None,
        )
        official_item = next(
            (
                item
                for item in [*selected, *indexed]
                if item.metadata.get("evidence_role") == "external_normative"
            ),
            None,
        )
        required = [item for item in (live_item, official_item) if item is not None]
        required = [
            item
            for index, item in enumerate(required)
            if all(previous.result_id != item.result_id for previous in required[:index])
        ]
        if required:
            required_ids = {item.result_id for item in required}
            base = [item for item in selected if item.result_id not in required_ids]
            keep = max(0, top_k - len(required))
            selected = [*required[:top_k], *base[:keep]]
        return [EngineeringRAGService._stamp_evidence_role(item) for item in selected]

    @staticmethod
    def _evidence_status(
        intent: SourceIntent,
        results: list[EngineeringSearchResult],
        live_attempted: bool,
    ) -> tuple[bool, list[str], str | None]:
        roles = {str(result.metadata.get("evidence_role")) for result in results}
        warnings: list[str] = []
        if intent is SourceIntent.IMPLEMENTATION:
            sufficient = "current_implementation" in roles
            if not sufficient:
                warnings.append("no live source match; indexed or official evidence cannot prove current implementation")
            return (
                sufficient,
                warnings,
                None if sufficient else "missing_current_implementation_evidence",
            )
        if intent is SourceIntent.DESIGN:
            sufficient = bool(roles & {"internal_design", "internal_history"})
            if not sufficient:
                warnings.append("no internal design evidence")
            return (
                sufficient,
                warnings,
                None if sufficient else "missing_internal_design_evidence",
            )
        if intent is SourceIntent.OFFICIAL:
            sufficient = "external_normative" in roles
            if not sufficient:
                warnings.append("no official normative evidence")
            return (
                sufficient,
                warnings,
                None if sufficient else "missing_official_evidence",
            )
        if intent is SourceIntent.COMPARISON:
            has_live = "current_implementation" in roles
            has_official = "external_normative" in roles
            if not has_live:
                warnings.append("comparison lacks live current-implementation evidence")
            if not has_official:
                warnings.append("comparison lacks official normative evidence")
            sufficient = has_live and has_official
            return (
                sufficient,
                warnings,
                None if sufficient else "missing_comparison_evidence",
            )
        return False, ["query is outside the configured scope"], "outside_configured_corpus"

    @staticmethod
    def _build_citations(results: list[EngineeringSearchResult]) -> list[EvidenceCitation]:
        citations: list[EvidenceCitation] = []
        for index, result in enumerate(results, start=1):
            metadata = result.metadata
            revision = (
                metadata.get("git_commit_sha")
                or metadata.get("source_commit_sha")
                or metadata.get("source_version")
            )
            current_revision = metadata.get("current_revision")
            if not isinstance(current_revision, dict):
                current_revision = {}
            public_source = result.source
            if metadata.get("live_verification") and metadata.get("relative_path"):
                public_source = str(metadata["relative_path"]).replace("\\", "/")
            url = (
                public_source
                if public_source.lower().startswith(("https://", "http://"))
                else None
            )
            citations.append(
                EvidenceCitation(
                    citation_id=f"E{index}",
                    source=public_source,
                    corpus=result.corpus,
                    authority=result.authority,
                    evidence_role=str(metadata.get("evidence_role") or "supporting"),
                    line_start=result.line_start,
                    line_end=result.line_end,
                    symbol=result.symbol,
                    revision=str(revision) if revision else None,
                    branch=_optional_string(
                        metadata.get("git_branch") or current_revision.get("branch")
                    ),
                    dirty=_optional_bool(
                        metadata.get(
                            "git_dirty",
                            current_revision.get(
                                "dirty", metadata.get("source_dirty", metadata.get("dirty"))
                            ),
                        )
                    ),
                    url=url,
                    live_verified=bool(metadata.get("live_verification")),
                    source_version=_optional_string(metadata.get("source_version")),
                    fetched_at=_optional_string(metadata.get("source_fetched_at")),
                    content_hash=_optional_string(
                        metadata.get("document_content_hash")
                        or metadata.get("source_content_hash")
                    ),
                )
            )
        return citations


def _git_revision(repo_root: Path) -> dict[str, object]:
    """Read Git revision metadata without changing repository state."""

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=3,
                check=False,
                shell=False,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=normal")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    return {
        "commit_sha": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def _read_manifest_build_id(path: str | Path) -> str | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    value = payload.get("build_id") if isinstance(payload, dict) else None
    return str(value).strip() or None if value is not None else None


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None
