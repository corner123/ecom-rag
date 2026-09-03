"""Deterministic, trade-identifier-aware lexical tokenization."""

from __future__ import annotations

import re
import unicodedata


TOKENIZER_VERSION = "trade-bm25-tokenizer-v1"
_IDENTIFIER = re.compile(r"[A-Za-z]+(?:[-_.][A-Za-z0-9]+)+|[A-Za-z]*\d+[A-Za-z0-9-]*")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")


class TradeTokenizer:
    """Preserve exact identifiers while adding CJK bigram recall."""

    version = TOKENIZER_VERSION

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        normalized = unicodedata.normalize("NFKC", text).casefold()
        identifiers = [match.group(0) for match in _IDENTIFIER.finditer(normalized)]
        masked_characters = list(normalized)
        for match in _IDENTIFIER.finditer(normalized):
            for index in range(match.start(), match.end()):
                masked_characters[index] = " "
        masked = "".join(masked_characters)

        latin_words = [match.group(0) for match in _LATIN_WORD.finditer(masked)]
        cjk_runs = [match.group(0) for match in _CJK_RUN.finditer(masked)]
        cjk_bigrams: list[str] = []
        for run in cjk_runs:
            cjk_bigrams.extend(run[index : index + 2] for index in range(len(run) - 1))
        return identifiers + cjk_bigrams + latin_words
