from __future__ import annotations

import math
from collections.abc import Sequence

from mini_rag.expense.models import ChunkRecord, RetrievedChunk


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

    def search(
        self, query_vector: Sequence[float], k: int = 3
    ) -> list[RetrievedChunk]:
        ranked = [
            RetrievedChunk(
                chunk_id=record.chunk_id,
                document=record.document,
                version=record.version,
                section=record.section,
                section_title=record.section_title,
                text=record.text,
                distance=_cosine_distance(query_vector, record.embedding),
            )
            for record in self._rows.values()
        ]
        ranked.sort(key=lambda chunk: chunk.distance)
        return ranked[:k]

    def count(self) -> int:
        return len(self._rows)

    def get(self, chunk_id: str) -> ChunkRecord:
        return self._rows[chunk_id]


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError(
            f"Vector length mismatch: query={len(left)} stored={len(right)}"
        )
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("cosine distance is undefined for a zero vector")
    return 1.0 - (dot / (left_norm * right_norm))
