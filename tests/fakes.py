from __future__ import annotations

from collections.abc import Sequence

from mini_rag.embeddings import EMBEDDING_DIM, validate_embeddings
from mini_rag.models import ChunkRecord


class FakeEmbedder:
    """Deterministic embedder for offline tests. Does not load MiniLM."""

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


class InMemoryDatabase:
    """Dict-backed store that mirrors upsert and stale-row deletion."""

    def __init__(self) -> None:
        self._rows: dict[str, ChunkRecord] = {}

    def upsert(self, records: Sequence[ChunkRecord]) -> None:
        if not records:
            return
        documents = {record.document for record in records}
        if len(documents) != 1:
            raise ValueError("upsert expects records from a single document")
        document = next(iter(documents))
        incoming_ids = {record.chunk_id for record in records}
        for record in records:
            self._rows[record.chunk_id] = record
        stale_ids = [
            chunk_id
            for chunk_id, stored in self._rows.items()
            if stored.document == document and chunk_id not in incoming_ids
        ]
        for chunk_id in stale_ids:
            del self._rows[chunk_id]

    def count(self) -> int:
        return len(self._rows)

    def get(self, chunk_id: str) -> ChunkRecord:
        return self._rows[chunk_id]
