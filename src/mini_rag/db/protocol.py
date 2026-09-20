from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from mini_rag.models import ChunkRecord


class DatabaseAdapter(Protocol):
    def upsert(self, records: Sequence[ChunkRecord]) -> None: ...

    def count(self) -> int: ...
