from __future__ import annotations

import re
from collections.abc import Sequence

from mini_rag.adapters import Embedder

_TRAILING_QUESTION = re.compile(r"\?\s*$")


def normalize_question(question: str) -> str:
    """Trim, collapse whitespace, and drop one trailing question mark."""
    collapsed = " ".join(question.split())
    return _TRAILING_QUESTION.sub("", collapsed).rstrip()


class QueryVectorCache:
    def __init__(self) -> None:
        self._vectors: dict[tuple[str, str], list[float]] = {}

    def get(self, model_name: str, text: str) -> list[float] | None:
        stored = self._vectors.get((model_name, text))
        if stored is None:
            return None
        return list(stored)

    def put(self, model_name: str, text: str, vector: Sequence[float]) -> None:
        self._vectors[(model_name, text)] = list(vector)


class RerankScoreCache:
    def __init__(self) -> None:
        self._scores: dict[tuple[str, str, str, str], float] = {}

    def get(
        self, model_name: str, question: str, chunk_id: str, text: str
    ) -> float | None:
        return self._scores.get((model_name, question, chunk_id, text))

    def put(
        self,
        model_name: str,
        question: str,
        chunk_id: str,
        text: str,
        score: float,
    ) -> None:
        self._scores[(model_name, question, chunk_id, text)] = score


class AnswerCache:
    def __init__(self) -> None:
        self._items: dict[tuple[object, ...], object] = {}

    def get(self, key: tuple[object, ...]) -> object | None:
        return self._items.get(key)

    def put(self, key: tuple[object, ...], value: object) -> None:
        self._items[key] = value


class PolicyCaches:
    def __init__(self) -> None:
        self.query_vectors = QueryVectorCache()
        self.rerank_scores = RerankScoreCache()
        self.answers = AnswerCache()


class CachingEmbedder:
    """Caches each embedded string for one model. The caller normalizes questions."""

    def __init__(
        self,
        embedder: Embedder,
        cache: QueryVectorCache,
        model_name: str,
    ) -> None:
        self._embedder = embedder
        self._cache = cache
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float] | None] = []
        missing: list[str] = []
        missing_at: list[int] = []
        for index, text in enumerate(texts):
            stored = self._cache.get(self._model_name, text)
            vectors.append(stored)
            if stored is None:
                missing.append(text)
                missing_at.append(index)
        if missing:
            encoded = self._embedder.embed_texts(missing)
            for index, text, vector in zip(missing_at, missing, encoded, strict=True):
                self._cache.put(self._model_name, text, vector)
                vectors[index] = vector
        return [vector for vector in vectors if vector is not None]
