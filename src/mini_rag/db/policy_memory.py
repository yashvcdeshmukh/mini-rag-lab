from __future__ import annotations

import math
import re
from collections.abc import Sequence

from mini_rag.policy.models import (
    KeywordHit,
    PolicyChunk,
    PolicyFact,
    RetrievedPolicyChunk,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "get",
        "had",
        "has",
        "have",
        "he",
        "her",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "many",
        "me",
        "more",
        "most",
        "my",
        "no",
        "not",
        "of",
        "on",
        "one",
        "only",
        "or",
        "our",
        "she",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)


class InMemoryPolicyStore:
    """Dict-backed store. Re-ingest replaces one edition and leaves the other."""

    def __init__(self) -> None:
        self.chunks: dict[str, PolicyChunk] = {}
        self.facts: dict[tuple[str, str, str], PolicyFact] = {}
        self._revision = 0

    def upsert(
        self,
        records: Sequence[PolicyChunk],
        facts: Sequence[PolicyFact] | None = None,
    ) -> None:
        if not records:
            raise ValueError("upsert requires at least one chunk")
        _require_embeddings(records)
        incoming_ids = {chunk.chunk_id for chunk in records}
        pairs = {(chunk.document_slug, chunk.version) for chunk in records}
        for chunk in records:
            self.chunks[chunk.chunk_id] = chunk
        stale_ids = [
            chunk_id
            for chunk_id, stored in self.chunks.items()
            if (stored.document_slug, stored.version) in pairs
            and chunk_id not in incoming_ids
        ]
        for chunk_id in stale_ids:
            del self.chunks[chunk_id]
        self._revision += 1
        if facts is None:
            return

        incoming_keys = {
            (fact.document_slug, fact.version, fact.fact_key) for fact in facts
        }
        for fact in facts:
            self.facts[(fact.document_slug, fact.version, fact.fact_key)] = fact
        stale_keys = [
            key
            for key, fact in self.facts.items()
            if (fact.document_slug, fact.version) in pairs and key not in incoming_keys
        ]
        for key in stale_keys:
            del self.facts[key]

    def search(
        self, query_vector: Sequence[float], k: int = 3
    ) -> list[RetrievedPolicyChunk]:
        ranked = [
            RetrievedPolicyChunk(
                chunk_id=chunk.chunk_id,
                document_slug=chunk.document_slug,
                document=chunk.document,
                version=chunk.version,
                section=chunk.section,
                section_title=chunk.section_title,
                parent_section=chunk.parent_section,
                parent_title=chunk.parent_title,
                text=chunk.text,
                distance=_cosine_distance(query_vector, chunk.embedding),
            )
            for chunk in self.chunks.values()
        ]
        ranked.sort(key=lambda hit: hit.distance)
        return ranked[:k]

    def search_keyword(self, question: str, k: int = 20) -> list[KeywordHit]:
        tokens = _query_tokens(question)
        if not tokens:
            return []
        ranked = [
            KeywordHit(
                chunk_id=chunk.chunk_id,
                document_slug=chunk.document_slug,
                document=chunk.document,
                version=chunk.version,
                section=chunk.section,
                section_title=chunk.section_title,
                parent_section=chunk.parent_section,
                parent_title=chunk.parent_title,
                text=chunk.text,
                rank=float(len(tokens & _text_tokens(chunk.search_text))),
            )
            for chunk in self.chunks.values()
        ]
        ranked = [hit for hit in ranked if hit.rank > 0]
        ranked.sort(key=lambda hit: (-hit.rank, hit.chunk_id))
        return ranked[:k]

    def count(self) -> int:
        return len(self.chunks)

    def corpus_stamp(self) -> tuple[int, str]:
        return (len(self.chunks), str(self._revision))

    def facts_for(self, fact_key: str) -> list[PolicyFact]:
        return [
            fact
            for fact in self.facts.values()
            if fact.fact_key == fact_key
        ]

    def chunks_for(self, document_slug: str, version: str) -> list[PolicyChunk]:
        return [
            chunk
            for chunk in self.chunks.values()
            if chunk.document_slug == document_slug and chunk.version == version
        ]


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


def _query_tokens(question: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(question.casefold())
        if token not in _STOPWORDS
    }


def _text_tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.casefold()))


def _require_embeddings(chunks: Sequence[PolicyChunk]) -> None:
    missing = [chunk.chunk_id for chunk in chunks if not chunk.embedding]
    if missing:
        raise ValueError(f"Chunks are missing embeddings: {missing}")
