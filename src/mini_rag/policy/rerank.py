from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from mini_rag.adapters import Reranker
from mini_rag.policy.cache import RerankScoreCache
from mini_rag.policy.models import embed_prefix
from mini_rag.policy.retrieve import FusedHit

RERANK_K = 3
STRONG_SCORE = 1.0
WEAK_SCORE = -1.0
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankedHit:
    """A fused chunk rescored by the cross-encoder."""

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
    rerank_score: float


def triage_band(scores: Sequence[float]) -> str:
    """Classify the top cross-encoder score as strong, weak, or middle."""
    if not scores or all(score <= WEAK_SCORE for score in scores):
        return "weak"
    if max(scores) >= STRONG_SCORE:
        return "strong"
    return "middle"


def rerank(
    question: str,
    fused: Sequence[FusedHit],
    reranker: Reranker,
    *,
    k: int = RERANK_K,
    cache: RerankScoreCache | None = None,
    model_name: str = "",
) -> list[RerankedHit]:
    """Score the raw question against each fused passage and keep the top k."""
    if not fused:
        logger.info("Rerank skipped")
        return []
    scores = _scores(question, fused, reranker, cache, model_name)
    order = sorted(range(len(fused)), key=lambda index: (-scores[index], index))
    kept = [_reranked_hit(fused[index], scores[index]) for index in order[:k]]
    logger.info(
        "Rerank %s",
        ", ".join(f"{hit.chunk_id} {hit.rerank_score:.4f}" for hit in kept),
    )
    return kept


def _scores(
    question: str,
    fused: Sequence[FusedHit],
    reranker: Reranker,
    cache: RerankScoreCache | None,
    model_name: str,
) -> list[float]:
    scores: list[float | None] = [None] * len(fused)
    missing: list[int] = []
    for index, hit in enumerate(fused):
        stored = None
        if cache is not None:
            stored = cache.get(model_name, question, hit.chunk_id, hit.text)
        if stored is None:
            missing.append(index)
        else:
            scores[index] = stored
    if missing:
        passages = [_passage(fused[index]) for index in missing]
        fetched = reranker.score(question, passages)
        if len(fetched) != len(missing):
            raise ValueError(
                f"Expected {len(missing)} rerank scores, got {len(fetched)}"
            )
        for index, score in zip(missing, fetched, strict=True):
            scores[index] = score
            if cache is not None:
                hit = fused[index]
                cache.put(model_name, question, hit.chunk_id, hit.text, score)
    resolved: list[float] = []
    for filled in scores:
        if filled is None:
            raise RuntimeError("rerank score was not filled")
        resolved.append(filled)
    return resolved


def _passage(hit: FusedHit) -> str:
    prefix = embed_prefix(
        hit.document,
        hit.version,
        hit.section,
        hit.section_title,
        hit.parent_section,
        hit.parent_title,
    )
    return f"{prefix}\n{hit.text}"


def _reranked_hit(hit: FusedHit, score: float) -> RerankedHit:
    return RerankedHit(
        chunk_id=hit.chunk_id,
        document_slug=hit.document_slug,
        document=hit.document,
        version=hit.version,
        section=hit.section,
        section_title=hit.section_title,
        parent_section=hit.parent_section,
        parent_title=hit.parent_title,
        text=hit.text,
        rrf_score=hit.rrf_score,
        rerank_score=score,
    )
