from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

RecordT = TypeVar("RecordT", contravariant=True)
HitT = TypeVar("HitT")
ChunkT = TypeVar("ChunkT", contravariant=True)


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


class DatabaseAdapter(Protocol[RecordT, HitT]):
    """Vector store used by both pipelines: write rows, dense search, count."""

    def upsert(self, records: Sequence[RecordT]) -> None: ...

    def search(
        self, query_vector: Sequence[float], k: int = 3
    ) -> list[HitT]: ...

    def count(self) -> int: ...


@dataclass(frozen=True)
class Citation:
    document: str
    version: str
    section: str


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    sufficient: bool
    section: str | None
    citations: tuple[Citation, ...] = ()


class Generator(Protocol[ChunkT]):
    def generate(
        self, question: str, chunks: Sequence[ChunkT]
    ) -> GenerationResult: ...


class Reranker(Protocol):
    """Scores one question against passages. Higher is a closer match."""

    def score(self, question: str, passages: Sequence[str]) -> list[float]: ...


@dataclass(frozen=True)
class Decision:
    choice: str
    probability: float


class Decider(Protocol):
    """Picks one label from a fixed menu. It does not write an answer."""

    def choose(
        self,
        name: str,
        state: str,
        instructions: str,
        choices: Sequence[str],
    ) -> Decision: ...
