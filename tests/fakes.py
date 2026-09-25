from __future__ import annotations

from collections.abc import Sequence

from mini_rag.embedder import EMBEDDING_DIM, validate_embeddings


class FakeEmbedder:
    """Deterministic embedder for offline tests. Does not load a model."""

    def __init__(self, dimension: int = EMBEDDING_DIM) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for index, text in enumerate(texts):
            vector = [0.0] * self._dimension
            vector[0] = float(index + 1)
            vector[1] = float(len(text) + 1)
            embeddings.append(vector)
        validate_embeddings(embeddings, expected_dim=self._dimension)
        return embeddings
