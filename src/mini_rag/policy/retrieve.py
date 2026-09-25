from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from mini_rag.adapters import Embedder
from mini_rag.db import InMemoryPolicyStore, PgPolicyStore
from mini_rag.embedder import SentenceTransformerEmbedder
from mini_rag.policy.config import policy_database_url
from mini_rag.policy.models import (
    BGE_MODEL_NAME,
    KeywordHit,
    RetrievedPolicyChunk,
)

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SEARCH_K = 20
FUSED_K = 10
RRF_K = 60

_VERSION_RE = re.compile(
    r"\b(?:version\s+v?|v)(?P<major>[12])(?:\.0\b|\b(?!\.))",
    re.IGNORECASE,
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FusedHit:
    """A chunk in reciprocal-rank-fusion order."""

    chunk_id: str
    document_slug: str
    document: str
    version: str
    section: str
    section_title: str
    parent_section: str | None
    parent_title: str | None
    text: str
    rrf_score: float


@dataclass(frozen=True)
class RetrievalResult:
    dense: list[RetrievedPolicyChunk]
    keyword: list[KeywordHit]
    fused: list[FusedHit]


def retrieve(
    question: str,
    embedder: Embedder,
    store: InMemoryPolicyStore | PgPolicyStore,
    *,
    k: int = SEARCH_K,
    fused_k: int | None = FUSED_K,
) -> RetrievalResult:
    """Dense search plus keyword search, fused, then filtered by an explicit version."""
    logger.info("Retrieve %s", question)
    query_vector = embedder.embed_texts([f"{QUERY_PREFIX}{question}"])[0]
    dense = store.search(query_vector, k=k)
    logger.info(
        "Dense %s",
        _format_scores((hit.chunk_id, hit.distance) for hit in dense),
    )
    keyword = store.search_keyword(question, k=k)
    logger.info(
        "Keyword %s",
        _format_scores((hit.chunk_id, hit.rank) for hit in keyword),
    )
    catalog: dict[str, RetrievedPolicyChunk | KeywordHit] = {
        hit.chunk_id: hit for hit in keyword
    }
    catalog.update({hit.chunk_id: hit for hit in dense})
    ranked = reciprocal_rank_fusion(
        [[hit.chunk_id for hit in dense], [hit.chunk_id for hit in keyword]],
        k=RRF_K,
    )
    logger.info("RRF %s", _format_scores(ranked))
    version = explicit_version(question)
    if version is not None:
        ranked = [
            (chunk_id, score)
            for chunk_id, score in ranked
            if catalog[chunk_id].version == version
        ]
        logger.info("Version filter %s kept %s", version, _format_scores(ranked))
    else:
        logger.info("Version filter skipped")
    chosen = ranked if fused_k is None else ranked[:fused_k]
    fused = [_fused_hit(catalog[chunk_id], score) for chunk_id, score in chosen]
    logger.info(
        "Fused %s",
        _format_scores((hit.chunk_id, hit.rrf_score) for hit in fused),
    )
    return RetrievalResult(dense=dense, keyword=keyword, fused=fused)


def reciprocal_rank_fusion(
    ranked_ids: Sequence[Sequence[str]],
    *,
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Score each id by the sum of 1 / (k + rank). Rank starts at 1."""
    scores: dict[str, float] = {}
    for ids in ranked_ids:
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def chunk_rank(ids: Sequence[str], chunk_id: str) -> int | None:
    """Return the 1-based position of chunk_id, or None when it is absent."""
    for index, candidate in enumerate(ids, start=1):
        if candidate == chunk_id:
            return index
    return None


def hybrid_finds_what_dense_misses(
    result: RetrievalResult, chunk_id: str, *, k: int
) -> bool:
    """True when fused top-k contains the chunk and dense top-k does not."""
    dense_ids = [hit.chunk_id for hit in result.dense[:k]]
    fused_ids = [hit.chunk_id for hit in result.fused[:k]]
    return chunk_rank(dense_ids, chunk_id) is None and chunk_rank(
        fused_ids, chunk_id
    ) is not None


def explicit_version(question: str) -> str | None:
    """Return 1.0 or 2.0 when the question names exactly one of those editions."""
    majors = {match.group("major") for match in _VERSION_RE.finditer(question)}
    if len(majors) != 1:
        return None
    return f"{majors.pop()}.0"


def _format_scores(pairs: Iterable[tuple[str, float]]) -> str:
    scored = list(pairs)
    if not scored:
        return "(none)"
    return ", ".join(f"{chunk_id} {score:.4f}" for chunk_id, score in scored)


def _fused_hit(hit: RetrievedPolicyChunk | KeywordHit, score: float) -> FusedHit:
    return FusedHit(
        chunk_id=hit.chunk_id,
        document_slug=hit.document_slug,
        document=hit.document,
        version=hit.version,
        section=hit.section,
        section_title=hit.section_title,
        parent_section=hit.parent_section,
        parent_title=hit.parent_title,
        text=hit.text,
        rrf_score=score,
    )


class _Labeled(Protocol):
    @property
    def chunk_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def section(self) -> str: ...


def _label(hit: _Labeled) -> str:
    return f"{hit.chunk_id} version={hit.version} section={hit.section}"


def _print_group(title: str, hits: Sequence[_Labeled]) -> None:
    print(title)
    if not hits:
        print("  (none)")
        return
    for hit in hits:
        print(f"  {_label(hit)}")


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Retrieve policy chunks for a question."
    )
    parser.add_argument("question", help="Question to search the policy corpus with")
    args = parser.parse_args(argv)

    try:
        with PgPolicyStore(policy_database_url()) as store:
            result = retrieve(
                args.question,
                SentenceTransformerEmbedder(BGE_MODEL_NAME),
                store,
            )
            from mini_rag.policy.rerank import rerank
            from mini_rag.reranker import CrossEncoderReranker

            reranked = rerank(args.question, result.fused, CrossEncoderReranker())
    except Exception as exc:
        logger.error("Retrieve failed: %s", exc)
        return 1
    _print_group("dense", result.dense)
    _print_group("keyword", result.keyword)
    _print_group("fused", result.fused)
    _print_group("reranked", reranked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
