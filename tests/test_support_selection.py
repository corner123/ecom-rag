from __future__ import annotations

from pathlib import Path

from rag_core.engineering.support_selection import (
    SupportSelectionProfile,
    ensure_internal_support,
    extract_latin_identifier_tokens,
    logical_file_key,
    select_supporting_candidate,
)
from rag_core.engineering.service import EngineeringRAGService
from rag_core.retrieval.engineering import EngineeringSearchResult


def _result(
    source: str,
    content: str,
    *,
    role: str,
    symbol: str | None = None,
    document_id: str | None = None,
    relative_path: str | None = None,
) -> EngineeringSearchResult:
    metadata = {
        "document_id": document_id or f"id:{source}:{symbol or ''}",
        "evidence_role": role,
    }
    if relative_path:
        metadata["relative_path"] = relative_path
    return EngineeringSearchResult(
        source=source,
        content=content,
        corpus="internal",
        authority="code" if role == "indexed_implementation" else "design",
        symbol=symbol,
        metadata=metadata,
    )


def test_identifier_token_extraction_is_casefolded_and_decomposes_compounds():
    tokens = extract_latin_identifier_tokens(
        "Why QueryEngine.submit_message uses model-facing state?"
    )

    assert {
        "queryengine.submit_message",
        "queryengine",
        "submit",
        "message",
        "model-facing",
        "model",
        "facing",
        "state",
    } <= tokens
    assert "why" not in tokens


def test_query_aware_selector_prefers_state_support_over_first_compressor():
    primary = [
        _result(
            "docs/context-memory-checkpoint.md",
            "The architecture separates canonical history and projection.",
            role="internal_design",
        )
    ]
    supporting = [
        _result(
            "mini_nanobot/core/context.py",
            "Context compression creates a model-facing projection.",
            role="indexed_implementation",
            symbol="ContextCompressor.compress",
        ),
        _result(
            "mini_nanobot/core/state.py",
            "AgentState preserves canonical history while active_messages exposes "
            "the model-facing projection.",
            role="indexed_implementation",
            symbol="AgentState.active_messages",
        ),
    ]

    selected = select_supporting_candidate(
        query=(
            "Why does the context/checkpoint architecture keep canonical history "
            "and a model-facing projection?"
        ),
        primary_results=primary,
        supporting_candidates=supporting,
        profile=SupportSelectionProfile.QUERY_AWARE_DIVERSE,
    )

    assert selected is supporting[1]
    assert selected.source == "mini_nanobot/core/state.py"


def test_query_aware_selector_deduplicates_logical_files():
    primary = [
        _result(
            "mini_nanobot/core/state.py#symbol:AgentState",
            "primary state chunk",
            role="internal_design",
            relative_path="mini_nanobot/core/state.py",
        )
    ]
    same_file = _result(
        "mini_nanobot/core/state.py:40-60",
        "AgentState.active_messages projection",
        role="indexed_implementation",
        symbol="AgentState.active_messages",
        document_id="different-chunk",
        relative_path="mini_nanobot/core/state.py",
    )
    other_file = _result(
        "mini_nanobot/core/checkpoint.py",
        "CheckpointStore projection",
        role="indexed_implementation",
        symbol="CheckpointStore",
    )

    selected = select_supporting_candidate(
        query="AgentState checkpoint projection",
        primary_results=primary,
        supporting_candidates=[same_file, other_file],
        profile=SupportSelectionProfile.QUERY_AWARE_DIVERSE,
    )

    assert logical_file_key(primary[0]) == logical_file_key(same_file)
    assert selected is other_file


def test_query_aware_selector_uses_original_rank_as_stable_tie_break():
    primary = [
        _result("docs/design.md", "design", role="internal_design")
    ]
    first = _result(
        "src/first.py",
        "Checkpoint state",
        role="indexed_implementation",
        symbol="State",
    )
    second = _result(
        "src/second.py",
        "Checkpoint state",
        role="indexed_implementation",
        symbol="State",
    )

    selected = select_supporting_candidate(
        query="Checkpoint state",
        primary_results=primary,
        supporting_candidates=[first, second],
        profile=SupportSelectionProfile.QUERY_AWARE_DIVERSE,
    )

    assert selected is first


def test_query_aware_selector_without_latin_tokens_falls_back_to_legacy():
    primary = [
        _result("docs/design.md", "design", role="internal_design")
    ]
    first = _result(
        "src/first.py", "first", role="indexed_implementation"
    )
    second = _result(
        "src/second.py",
        "AgentState active_messages canonical projection",
        role="indexed_implementation",
        symbol="AgentState.active_messages",
    )

    selected = select_supporting_candidate(
        query="为什么采用这样的设计？",
        primary_results=primary,
        supporting_candidates=[first, second],
        profile=SupportSelectionProfile.QUERY_AWARE_DIVERSE,
    )

    assert selected is first


def test_legacy_profile_preserves_first_candidate_and_one_slot_contract():
    primary = [
        _result("docs/a.md", "a", role="internal_design"),
        _result("docs/b.md", "b", role="internal_design"),
        _result("docs/c.md", "c", role="internal_design"),
    ]
    first = _result(
        "docs/a.md#another-chunk",
        "first support",
        role="indexed_implementation",
        document_id="support-first",
    )
    second = _result(
        "src/better.py",
        "AgentState canonical projection",
        role="indexed_implementation",
        symbol="AgentState.active_messages",
    )

    selected = ensure_internal_support(
        query="AgentState canonical projection",
        results=primary,
        supporting_candidates=[first, second],
        top_k=3,
        profile=SupportSelectionProfile.LEGACY_FIRST,
    )

    assert selected == [primary[0], primary[1], first]
    assert len(selected) == 3


def test_query_aware_parent_expansion_uses_live_enclosing_class(tmp_path: Path):
    method = _result(
        "mini_nanobot/core/state.py#symbol:AgentState.active_messages",
        "def active_messages(self): ...",
        role="indexed_implementation",
        symbol="AgentState.active_messages",
    )
    parent = EngineeringSearchResult(
        source="D:/repo/mini_nanobot/core/state.py",
        content="class AgentState:\n    def active_messages(self): ...\n    def to_dict(self): ...",
        corpus="internal",
        authority="code",
        symbol="AgentState",
        metadata={
            "relative_path": "mini_nanobot/core/state.py",
            "live_verification": True,
        },
    )

    class FakeAST:
        repo_root = tmp_path

        def search(self, query: str, top_k: int = 10):
            assert query == "AgentState"
            assert top_k == 8
            return [parent]

    service = object.__new__(EngineeringRAGService)
    service.live_ast = FakeAST()
    service.live_code = None
    service.live_git = None

    expanded = service._expand_selected_support_parent([method])

    assert expanded[0].symbol == "AgentState"
    assert expanded[0].metadata["evidence_role"] == "current_implementation"
    assert expanded[0].metadata["support_parent_expanded_from"] == (
        "AgentState.active_messages"
    )
    assert expanded[0].metadata["relative_path"] == "mini_nanobot/core/state.py"
