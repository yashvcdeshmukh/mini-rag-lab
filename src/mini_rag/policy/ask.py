from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Protocol, TypeVar

from mini_rag.adapters import (
    Citation,
    Decider,
    Embedder,
    GenerationResult,
    Generator,
    Reranker,
)
from mini_rag.db import InMemoryPolicyStore, PgPolicyStore
from mini_rag.decider import JevDecider
from mini_rag.embedder import SentenceTransformerEmbedder
from mini_rag.generator import OllamaGenerator, OllamaRewriter
from mini_rag.policy.cache import CachingEmbedder, PolicyCaches, normalize_question
from mini_rag.policy.config import policy_database_url
from mini_rag.policy.facts import CHECKED_FACTS, CHOICE_CRITERIA
from mini_rag.policy.generation import (
    REFUSAL_ANSWER,
    SYSTEM_PROMPT,
    build_generation_prompt,
)
from mini_rag.policy.models import (
    BGE_MODEL_NAME,
    PolicyFact,
    document_title,
    section_covers,
)
from mini_rag.policy.rerank import RERANK_K, RerankedHit, rerank, triage_band
from mini_rag.policy.retrieve import (
    FUSED_K,
    FusedHit,
    RetrievalResult,
    explicit_version,
    retrieve,
)
from mini_rag.reranker import RERANKER_MODEL_NAME, CrossEncoderReranker

logger = logging.getLogger(__name__)
Store = InMemoryPolicyStore | PgPolicyStore
Rewriter = Callable[[str], str]
_NEED_MORE = (
    "Do these excerpts contain enough specific information to answer the "
    "question fully? Choose no when they only mention the topic and more "
    "sections are needed."
)
EXTRA_K = 3


class _Scoped(Protocol):
    @property
    def document_slug(self) -> str: ...

    @property
    def version(self) -> str: ...


ScopedT = TypeVar("ScopedT", bound=_Scoped)


@dataclass(frozen=True)
class DenseHitView:
    id: str
    distance: float


@dataclass(frozen=True)
class KeywordHitView:
    id: str
    rank: float


@dataclass(frozen=True)
class RerankView:
    id: str
    score: float


@dataclass(frozen=True)
class DecisionView:
    name: str
    choice: str
    probability: float


@dataclass(frozen=True)
class CorpusStampView:
    chunks: int
    revision: str


@dataclass(frozen=True)
class AskTrace:
    question: str
    embedding_model: str
    reranker_model: str
    generator_model: str
    jev_model: str
    corpus: CorpusStampView
    route: str
    fact_key: str | None
    edition: str
    decisions: tuple[DecisionView, ...]
    dense: tuple[DenseHitView, ...]
    keyword: tuple[KeywordHitView, ...]
    fused: tuple[str, ...]
    rerank_scores: tuple[RerankView, ...]
    triage: str
    sufficiency: str
    rewrite: str | None
    final_chunk_ids: tuple[str, ...]
    citations: tuple[Citation, ...]
    answer: str
    cache_hit: bool
    timings_ms: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class AskResult:
    answer: str
    citations: tuple[Citation, ...]
    trace: AskTrace


def ask_question(
    question: str,
    embedder: Embedder,
    store: Store,
    reranker: Reranker,
    generator: Generator[RerankedHit],
    decider: Decider,
    *,
    rewriter: Rewriter | None = None,
    use_cache: bool = True,
    caches: PolicyCaches | None = None,
    embedding_model: str = BGE_MODEL_NAME,
    reranker_model: str = RERANKER_MODEL_NAME,
) -> AskResult:
    """Answer one policy question. Hybrid retrieval is the default path."""
    started = time.perf_counter()
    timings: dict[str, float] = {}
    normalized = normalize_question(question)
    generator_model = _generator_model(generator)
    configured_jev = _decider_model(decider)
    stamp = store.corpus_stamp()
    cache_key = (
        normalized,
        embedding_model,
        reranker_model,
        generator_model,
        configured_jev,
        stamp,
    )
    if use_cache and caches is not None:
        cached = caches.answers.get(cache_key)
        if isinstance(cached, AskResult):
            return replace(cached, trace=replace(cached.trace, cache_hit=True))

    active_embedder: Embedder = embedder
    if caches is not None:
        active_embedder = CachingEmbedder(
            embedder, caches.query_vectors, embedding_model
        )
    fact_keys = tuple(sorted({fact.fact_key for fact in CHECKED_FACTS}))
    edition_choice: str | None = None
    decisions: list[DecisionView] = []

    def take(
        name: str, state: str, instructions: str, choices: Sequence[str]
    ) -> str:
        decision = decider.choose(name, state, instructions, choices)
        decisions.append(
            DecisionView(name, decision.choice, decision.probability)
        )
        return decision.choice

    def edition_scope() -> str:
        nonlocal edition_choice
        explicit = explicit_version(normalized)
        if explicit is not None:
            return explicit
        if edition_choice is None:
            edition_choice = take(
                "edition",
                normalized,
                "Choose current or previous only when the question selects "
                "one edition over the other. Otherwise choose unspecified.",
                ("current", "previous", "unspecified"),
            )
        return edition_choice

    route_started = time.perf_counter()
    fact_choice = take(
        "fact",
        normalized,
        "Choose the checked fact this question looks up, or none.",
        (*fact_keys, "none"),
    )
    timings["route"] = _elapsed(route_started)
    if fact_choice != "none":
        scope = edition_scope()
        rows = apply_edition(store.facts_for(fact_choice), scope)
        if rows:
            answer, citations = _fact_answer(rows)
            result = _result(
                answer=answer,
                citations=citations,
                question=normalized,
                embedding_model=embedding_model,
                reranker_model=reranker_model,
                generator_model=generator_model,
                route="structured",
                fact_key=fact_choice,
                edition=scope,
                decisions=tuple(decisions),
                found=None,
                ranked=(),
                triage="skipped",
                sufficiency="skipped",
                rewrite=None,
                timings=timings,
                started=started,
                stamp=stamp,
                jev_model=_decider_model(decider),
            )
            _store_answer(caches, use_cache, cache_key, result)
            return result

    scope = explicit_version(normalized) or "unspecified"
    search_question = normalized
    rewrite_text: str | None = None
    found, ranked = _search(
        search_question,
        active_embedder,
        store,
        reranker,
        caches,
        reranker_model,
        scope,
        timings,
    )
    band = triage_band([hit.rerank_score for hit in ranked])
    sufficiency = "skipped"
    needs_more = band == "weak"
    if band == "middle":
        decision = decider.choose(
            "sufficient",
            _sufficient_state(normalized, ranked, found.fused),
            _NEED_MORE,
            ("yes", "no"),
        )
        needs_more = decision.choice == "no"
        sufficiency = decision.choice
        decisions.append(
            DecisionView("sufficient", decision.choice, decision.probability)
        )
    if needs_more:
        omitted = [
            hit
            for hit in found.fused
            if hit.chunk_id not in {item.chunk_id for item in ranked}
        ]
        rewrite_text = normalize_question(
            _rewrite(_rewrite_request(normalized, omitted), rewriter)
        )
        ranked = _rank(
            search_question,
            found,
            reranker,
            caches,
            reranker_model,
            scope,
            timings,
            keep_all=False,
            rerank_k=RERANK_K + EXTRA_K,
        )
        logger.info(
            "Expand kept %s",
            ", ".join(hit.chunk_id for hit in ranked) or "(none)",
        )
    generate_started = time.perf_counter()
    generation = generator.generate(normalized, ranked)
    timings["generate"] = _elapsed(generate_started)
    answer, citations = _grounded_answer(generation, ranked)
    result = _result(
        answer=answer,
        citations=citations,
        question=normalized,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        generator_model=generator_model,
        route="unstructured",
        fact_key=None,
        edition=scope,
        decisions=tuple(decisions),
        found=found,
        ranked=ranked,
        triage=band,
        sufficiency=sufficiency,
        rewrite=rewrite_text,
        timings=timings,
        started=started,
        stamp=stamp,
        jev_model=_decider_model(decider),
    )
    _store_answer(caches, use_cache, cache_key, result)
    return result


def apply_edition(items: Sequence[ScopedT], scope: str) -> list[ScopedT]:
    """Keep one edition per document, or the explicit version, or every edition."""
    if scope == "unspecified":
        return list(items)
    if scope not in {"current", "previous"}:
        return [item for item in items if item.version == scope]
    versions: dict[str, list[str]] = {}
    for item in items:
        seen = versions.setdefault(item.document_slug, [])
        if item.version not in seen:
            seen.append(item.version)
    chosen = {
        slug: (ordered[-1] if scope == "current" else ordered[0])
        for slug, ordered in (
            (slug, sorted(values, key=_version_key))
            for slug, values in versions.items()
        )
    }
    return [
        item for item in items if item.version == chosen[item.document_slug]
    ]


def _search(
    question: str,
    embedder: Embedder,
    store: Store,
    reranker: Reranker,
    caches: PolicyCaches | None,
    reranker_model: str,
    scope: str,
    timings: dict[str, float],
    *,
    keep_all: bool = False,
    fused_k: int | None = FUSED_K,
) -> tuple[RetrievalResult, list[RerankedHit]]:
    retrieve_started = time.perf_counter()
    found = retrieve(question, embedder, store, fused_k=fused_k)
    timings["retrieve"] = _elapsed(retrieve_started)
    ranked = _rank(
        question,
        found,
        reranker,
        caches,
        reranker_model,
        scope,
        timings,
        keep_all=keep_all,
        rerank_k=RERANK_K,
    )
    return found, ranked


def _rank(
    question: str,
    found: RetrievalResult,
    reranker: Reranker,
    caches: PolicyCaches | None,
    reranker_model: str,
    scope: str,
    timings: dict[str, float],
    *,
    keep_all: bool,
    rerank_k: int = RERANK_K,
) -> list[RerankedHit]:
    rerank_started = time.perf_counter()
    limit = len(found.fused) if keep_all else rerank_k
    ranked = rerank(
        question,
        found.fused,
        reranker,
        k=limit,
        cache=None if caches is None else caches.rerank_scores,
        model_name=reranker_model,
    )
    timings["rerank"] = _elapsed(rerank_started)
    kept = apply_edition(ranked, scope)
    logger.info(
        "Edition %s kept %s",
        scope,
        ", ".join(hit.chunk_id for hit in kept) or "(none)",
    )
    return kept


def _rewrite(question: str, rewriter: Rewriter | None) -> str:
    if rewriter is None:
        rewriter = OllamaRewriter()
    return rewriter(question)


def _grounded_answer(
    generation: GenerationResult, chunks: Sequence[RerankedHit]
) -> tuple[str, tuple[Citation, ...]]:
    if not generation.sufficient:
        return REFUSAL_ANSWER, ()
    citations = tuple(
        citation
        for citation in generation.citations
        if any(
            chunk.document == citation.document
            and chunk.version == citation.version
            and section_covers(citation.section, chunk.section)
            for chunk in chunks
        )
    )
    if not citations:
        return REFUSAL_ANSWER, ()
    return generation.answer, citations


def _fact_answer(facts: Sequence[PolicyFact]) -> tuple[str, tuple[Citation, ...]]:
    ordered = sorted(
        facts,
        key=lambda fact: (
            fact.document_slug,
            _version_key(fact.version),
            fact.section,
        ),
    )
    answer = "; ".join(
        f"{fact.value} ({document_title(fact.document_slug)} "
        f"v{fact.version} section {fact.section})"
        for fact in ordered
    )
    citations = tuple(
        Citation(
            document=document_title(fact.document_slug),
            version=fact.version,
            section=fact.section,
        )
        for fact in ordered
    )
    return answer, citations


def _result(
    *,
    answer: str,
    citations: tuple[Citation, ...],
    question: str,
    embedding_model: str,
    reranker_model: str,
    generator_model: str,
    route: str,
    fact_key: str | None,
    edition: str,
    decisions: tuple[DecisionView, ...],
    found: RetrievalResult | None,
    ranked: Sequence[RerankedHit],
    triage: str,
    sufficiency: str,
    rewrite: str | None,
    timings: dict[str, float],
    started: float,
    stamp: tuple[int, str],
    jev_model: str,
) -> AskResult:
    timings["total"] = _elapsed(started)
    dense: tuple[DenseHitView, ...] = ()
    keyword: tuple[KeywordHitView, ...] = ()
    fused: tuple[str, ...] = ()
    if found is not None:
        dense = tuple(
            DenseHitView(hit.chunk_id, hit.distance) for hit in found.dense
        )
        keyword = tuple(
            KeywordHitView(hit.chunk_id, hit.rank) for hit in found.keyword
        )
        fused = tuple(hit.chunk_id for hit in found.fused)
    trace = AskTrace(
        question=question,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        generator_model=generator_model,
        jev_model=jev_model,
        corpus=CorpusStampView(stamp[0], stamp[1]),
        route=route,
        fact_key=fact_key,
        edition=edition,
        decisions=decisions,
        dense=dense,
        keyword=keyword,
        fused=fused,
        rerank_scores=tuple(
            RerankView(hit.chunk_id, hit.rerank_score) for hit in ranked
        ),
        triage=triage,
        sufficiency=sufficiency,
        rewrite=rewrite,
        final_chunk_ids=tuple(hit.chunk_id for hit in ranked),
        citations=citations,
        answer=answer,
        cache_hit=False,
        timings_ms=tuple(timings.items()),
    )
    return AskResult(answer=answer, citations=citations, trace=trace)


def _store_answer(
    caches: PolicyCaches | None,
    use_cache: bool,
    key: tuple[object, ...],
    result: AskResult,
) -> None:
    if use_cache and caches is not None:
        caches.answers.put(key, result)


def _sufficient_state(
    question: str,
    chunks: Sequence[RerankedHit],
    fused: Sequence[FusedHit],
) -> str:
    excerpts = "\n".join(
        f"{chunk.document} v{chunk.version} section {chunk.section}: {chunk.text}"
        for chunk in chunks
    )
    shown = {chunk.chunk_id for chunk in chunks}
    others = ", ".join(
        f"{hit.document} v{hit.version} section {hit.section} {hit.section_title}"
        for hit in fused
        if hit.chunk_id not in shown
    )
    held = f"\nOther retrieved sections: {others}" if others else ""
    return f"{excerpts}{held}\nQuestion: {question}"


def _rewrite_request(question: str, omitted: Sequence[FusedHit]) -> str:
    if not omitted:
        return question
    titles = ", ".join(
        f"{hit.document} v{hit.version} section {hit.section} {hit.section_title}"
        for hit in omitted
    )
    return (
        f"{question}\n"
        "The first excerpts were not enough. "
        f"Also search these sections: {titles}"
    )


def _generator_model(generator: Generator[RerankedHit]) -> str:
    model = getattr(generator, "model", "")
    return str(model)


def _decider_model(decider: Decider) -> str:
    model = getattr(decider, "model", "")
    return str(model)


def _version_key(version: str) -> tuple[int, ...]:
    numbers: list[int] = []
    for part in version.split("."):
        numbers.append(int(part) if part.isdigit() else 0)
    return tuple(numbers)


def _elapsed(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Answer a question from the policy corpus."
    )
    parser.add_argument("--question", required=True, help="User question")
    args = parser.parse_args(argv)

    try:
        with PgPolicyStore(policy_database_url()) as store:
            result = ask_question(
                args.question,
                SentenceTransformerEmbedder(BGE_MODEL_NAME),
                store,
                CrossEncoderReranker(),
                OllamaGenerator(
                    system_prompt=SYSTEM_PROMPT,
                    build_user_prompt=build_generation_prompt,
                    temperature=0,
                ),
                JevDecider(criteria=CHOICE_CRITERIA),
                caches=PolicyCaches(),
            )
    except Exception as exc:
        logger.error("Ask failed: %s", exc)
        return 1
    json.dump(asdict(result), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
