from __future__ import annotations

from collections.abc import Sequence

from mini_rag.adapters import GenerationResult
from mini_rag.expense.models import RetrievedChunk


class FakeGenerator:
    """Scripted generator for offline tests. Does not call Ollama."""

    def __init__(
        self,
        result: GenerationResult | None = None,
        results: dict[str, GenerationResult] | None = None,
    ) -> None:
        self.results = results or {}
        self.result = result
        if self.result is None and not self.results:
            self.result = GenerationResult(
                answer="Employees may claim up to $65 per day for meals.",
                sufficient=True,
                section="1",
            )
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    def generate(
        self, question: str, chunks: Sequence[RetrievedChunk]
    ) -> GenerationResult:
        self.calls.append((question, list(chunks)))
        if question in self.results:
            return self.results[question]
        if self.result is not None:
            return self.result
        raise KeyError(question)


class ScriptedEmbedder:
    """Maps exact question text to a query vector for offline retrieval tests."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        if not mapping:
            raise ValueError("ScriptedEmbedder requires at least one vector")
        self._mapping = mapping
        self._dimension = len(next(iter(mapping.values())))

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(self._mapping[text]) for text in texts]

