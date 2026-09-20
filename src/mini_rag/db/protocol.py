from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from mini_rag.models import ChunkRecord, RetrievedChunk


class DatabaseAdapter(Protocol):
    def upsert(self, records: Sequence[ChunkRecord]) -> None: ...

    def search(
        self, query_vector: Sequence[float], k: int = 3
    ) -> list[RetrievedChunk]: ...

    def count(self) -> int: ...
