from __future__ import annotations

import logging
from collections.abc import Sequence

import pytest

from fakes import FakeEmbedder
from mini_rag.adapters import Citation, GenerationResult
from mini_rag.db import InMemoryPolicyStore
from mini_rag.generator.ollama import OllamaGenerator, parse_generation_result
from mini_rag.policy.ask import AskResult, DecisionView, ask_question
from mini_rag.policy.cache import CachingEmbedder, PolicyCaches, QueryVectorCache
from mini_rag.policy.generation import SYSTEM_PROMPT, build_generation_prompt
from mini_rag.policy.models import PolicyChunk, PolicyFact
from mini_rag.policy.rerank import RerankScoreCache, rerank
from mini_rag.policy.retrieve import FusedHit


class CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__(dimension=2)
        self.calls = 0

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return super().embed_texts(texts)


class FlatReranker:
    def __init__(self, score: float) -> None:
        self._score = score
        self.calls = 0

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        self.calls += 1
        return [self._score for _passage in passages]


class ScriptedDecider:
    def __init__(
        self,
        choices: dict[str, str],
        probabilities: dict[str, float] | None = None,
    ) -> None:
        self._choices = choices
        self._probabilities = probabilities or {}
        self.calls: list[tuple[str, str, str, tuple[str, ...]]] = []

    def choose(
        self,
        name: str,
        state: str,
        instructions: str,
        choices: Sequence[str],
    ) -> object:
        self.calls.append((name, state, instructions, tuple(choices)))
        choice = self._choices[name]
        if choice not in choices:
            raise ValueError(choice)
        from mini_rag.adapters import Decision

        return Decision(choice, self._probabilities.get(name, 1.0))


class ScriptedGenerator:
    def __init__(self, result: GenerationResult) -> None:
        self._result = result
        self.questions: list[str] = []
        self.chunk_ids: list[tuple[str, ...]] = []
        self.model = "fake-generator"

    def generate(
        self, question: str, chunks: Sequence[object]
    ) -> GenerationResult:
        self.questions.append(question)
        self.chunk_ids.append(tuple(getattr(chunk, "chunk_id") for chunk in chunks))
        return self._result


def _chunk(version: str, section: str, text: str) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=f"time-usage:v{version}:section-{section}",
        document_slug="time-usage",
        document="Time & Usage Policy",
        version=version,
        section=section,
        section_title="Token Allocation",
        parent_section=None,
        parent_title=None,
        text=text,
        embed_input=text,
        source_file=f"time-v{version}.docx",
        search_text=text,
        embedding=(1.0, 0.0),
    )


def _store() -> InMemoryPolicyStore:
    store = InMemoryPolicyStore()
    store.upsert(
        [
            _chunk("1.0", "5", "1000000 tokens allocation."),
            _chunk("2.0", "6", "500000 tokens allocation."),
        ],
        [
            PolicyFact("time-usage", "1.0", "token_allocation", "1000000", "5"),
            PolicyFact("time-usage", "2.0", "token_allocation", "500000", "6"),
        ],
    )
    return store


def _ask(
    question: str,
    *,
    fact: str = "none",
    edition: str = "unspecified",
    sufficient: str = "yes",
    score: float = 2.0,
    generation: GenerationResult | None = None,
    caches: PolicyCaches | None = None,
    use_cache: bool = True,
    rewriter: CallableRewriter | None = None,
    probability: float | None = None,
    embedder: CountingEmbedder | None = None,
    store: InMemoryPolicyStore | None = None,
    decider: ScriptedDecider | None = None,
    generator: ScriptedGenerator | None = None,
) -> tuple[AskResult, ScriptedDecider, ScriptedGenerator, CountingEmbedder]:
    embedder = embedder or CountingEmbedder()
    decider = decider or ScriptedDecider(
        {"fact": fact, "edition": edition, "sufficient": sufficient},
        None if probability is None else {"sufficient": probability},
    )
    generator = generator or ScriptedGenerator(
        generation
        or GenerationResult(
            answer="1000000 and 500000",
            sufficient=True,
            section="6",
            citations=(
                Citation("Time & Usage Policy", "1.0", "5"),
                Citation("Time & Usage Policy", "2.0", "6"),
            ),
        )
    )
    result = ask_question(
        question,
        embedder,
        store or _store(),
        FlatReranker(score),
        generator,
        decider,
        rewriter=rewriter or _raise_rewrite,
        use_cache=use_cache,
        caches=caches,
        embedding_model="test-embed",
        reranker_model="test-rerank",
    )
    return result, decider, generator, embedder


class CallableRewriter:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def __call__(self, question: str) -> str:
        self.calls += 1
        return self.text


def _raise_rewrite(_question: str) -> str:
    raise AssertionError("rewrite was not expected")


def _kinds(decider: ScriptedDecider) -> list[str]:
    return [call[0] for call in decider.calls]


def test_unspecified_question_keeps_both_editions() -> None:
    result, decider, generator, _embedder = _ask("how many tokens do I get?")

    assert result.trace.route == "unstructured"
    assert result.trace.edition == "unspecified"
    assert result.trace.triage == "strong"
    assert result.trace.sufficiency == "skipped"
    assert result.trace.rewrite is None
    assert generator.chunk_ids == [
        (
            "time-usage:v1.0:section-5",
            "time-usage:v2.0:section-6",
        )
    ]
    assert {citation.version for citation in result.citations} == {"1.0", "2.0"}
    assert _kinds(decider) == ["fact"]
    assert result.trace.decisions == (DecisionView("fact", "none", 1.0),)


def test_ask_logs_the_edition_filter(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        _ask("how many tokens do I get?")

    kept = (
        "Edition unspecified kept "
        "time-usage:v1.0:section-5, time-usage:v2.0:section-6"
    )
    assert kept in caplog.text
    assert "allocation" not in caplog.text


def test_previous_fact_keeps_the_lower_edition() -> None:
    result, decider, generator, _embedder = _ask(
        "what did the previous policy allow?",
        fact="token_allocation",
        edition="previous",
    )

    assert result.answer == "1000000 (Time & Usage Policy v1.0 section 5)"
    assert "Otherwise choose unspecified" in decider.calls[1][2]
    assert generator.questions == []


def test_explicit_version_does_not_ask_for_an_edition() -> None:
    result, decider, generator, _embedder = _ask(
        "how many tokens in version 2.0?"
    )

    assert _kinds(decider) == ["fact"]
    assert result.trace.edition == "2.0"
    assert generator.chunk_ids == [("time-usage:v2.0:section-6",)]


def test_citation_outside_the_reranked_chunks_is_refused() -> None:
    result, _decider, _generator, _embedder = _ask(
        "how many tokens?",
        generation=GenerationResult(
            answer="invented",
            sufficient=True,
            section="9",
            citations=(Citation("Time & Usage Policy", "2.0", "9"),),
        ),
    )

    assert result.answer == "The provided policy does not answer this question."
    assert result.citations == ()


def test_subsection_citation_matches_the_parent_chunk() -> None:
    result, _decider, _generator, _embedder = _ask(
        "how many tokens?",
        generation=GenerationResult(
            answer="Eat one spoonful.",
            sufficient=True,
            section="6.3",
            citations=(Citation("Time & Usage Policy", "2.0", "6.3"),),
        ),
    )

    assert result.answer == "Eat one spoonful."
    assert result.citations == (Citation("Time & Usage Policy", "2.0", "6.3"),)


def test_middle_no_rewrites_once_and_answers_the_original_question() -> None:
    rewriter = CallableRewriter("token allocation")
    result, decider, generator, _embedder = _ask(
        "how many tokens?",
        score=0.0,
        sufficient="no",
        rewriter=rewriter,
    )

    assert result.trace.triage == "middle"
    assert result.trace.sufficiency == "no"
    assert rewriter.calls == 1
    assert result.trace.rewrite == "token allocation"
    assert generator.questions == ["how many tokens"]
    assert _kinds(decider).count("sufficient") == 1
    assert result.trace.decisions[-1] == DecisionView("sufficient", "no", 1.0)


def test_not_enough_adds_the_next_three_chunks() -> None:
    store = InMemoryPolicyStore()
    store.upsert(
        [_chunk("1.0", str(section), f"{section} tokens.") for section in range(1, 8)]
    )
    result, decider, generator, _embedder = _ask(
        "how many tokens?",
        score=0.0,
        sufficient="no",
        rewriter=CallableRewriter("token sections"),
        store=store,
        generation=GenerationResult(
            answer="several sections",
            sufficient=True,
            section="1",
            citations=(Citation("Time & Usage Policy", "1.0", "1"),),
        ),
    )

    assert "sufficient" in _kinds(decider)
    assert result.trace.final_chunk_ids == tuple(
        f"time-usage:v1.0:section-{section}" for section in range(1, 7)
    )
    assert "time-usage:v1.0:section-7" not in result.trace.final_chunk_ids
    assert generator.chunk_ids[-1] == result.trace.final_chunk_ids


def test_yes_keeps_the_top_chunks_even_when_the_probability_is_low() -> None:
    store = InMemoryPolicyStore()
    store.upsert(
        [_chunk("1.0", str(section), f"{section} tokens.") for section in range(1, 5)]
    )
    rewriter = CallableRewriter("token sections")
    result, _decider, _generator, _embedder = _ask(
        "how many tokens?",
        score=0.0,
        sufficient="yes",
        probability=0.53,
        rewriter=rewriter,
        store=store,
    )

    assert result.trace.sufficiency == "yes"
    assert rewriter.calls == 0
    assert len(result.trace.final_chunk_ids) == 3
    assert "time-usage:v1.0:section-4" not in result.trace.final_chunk_ids


def test_confident_yes_keeps_the_top_chunks() -> None:
    store = InMemoryPolicyStore()
    store.upsert(
        [_chunk("1.0", str(section), f"{section} tokens.") for section in range(1, 5)]
    )
    rewriter = CallableRewriter("token sections")
    result, _decider, _generator, _embedder = _ask(
        "how many tokens?",
        score=0.0,
        sufficient="yes",
        probability=0.9,
        rewriter=rewriter,
        store=store,
    )

    assert result.trace.sufficiency == "yes"
    assert rewriter.calls == 0
    assert len(result.trace.final_chunk_ids) == 3
    assert "time-usage:v1.0:section-4" not in result.trace.final_chunk_ids


def test_weak_band_skips_sufficiency() -> None:
    rewriter = CallableRewriter("token allocation")
    result, decider, _generator, _embedder = _ask(
        "how many tokens?",
        score=-2.0,
        rewriter=rewriter,
    )

    assert result.trace.triage == "weak"
    assert result.trace.sufficiency == "skipped"
    assert "sufficient" not in _kinds(decider)
    assert rewriter.calls == 1


def test_fact_lookup_returns_stored_values_for_both_editions() -> None:
    result, decider, generator, embedder = _ask(
        "how many tokens?",
        fact="token_allocation",
    )

    assert result.trace.route == "structured"
    assert result.trace.triage == "skipped"
    assert "1000000" in result.answer
    assert "500000" in result.answer
    assert {citation.version for citation in result.citations} == {"1.0", "2.0"}
    assert generator.questions == []
    assert embedder.calls == 0
    assert _kinds(decider) == ["fact", "edition"]


def test_missing_fact_falls_through_to_retrieval() -> None:
    result, _decider, generator, embedder = _ask(
        "how many tokens?",
        fact="caffeine_mg_per_day",
    )

    assert result.trace.route == "unstructured"
    assert generator.questions == ["how many tokens"]
    assert embedder.calls == 1


def test_current_fact_keeps_the_highest_edition() -> None:
    result, _decider, _generator, _embedder = _ask(
        "what did the previous policy allow?",
        fact="token_allocation",
        edition="current",
    )

    assert result.answer == "500000 (Time & Usage Policy v2.0 section 6)"
    assert result.citations == (Citation("Time & Usage Policy", "2.0", "6"),)


def test_answer_cache_hits_after_normalization_and_misses_after_ingest() -> None:
    caches = PolicyCaches()
    embedder = CountingEmbedder()
    decider = ScriptedDecider({"fact": "none", "edition": "unspecified"})
    generator = ScriptedGenerator(
        GenerationResult(
            "500000",
            True,
            "6",
            (Citation("Time & Usage Policy", "2.0", "6"),),
        )
    )
    store = _store()
    shared = {
        "embedder": embedder,
        "store": store,
        "decider": decider,
        "generator": generator,
        "caches": caches,
    }
    first, _, _, _ = _ask("how many tokens?", **shared)
    second, _, _, _ = _ask("  how many tokens ? ", **shared)

    assert first.trace.cache_hit is False
    assert second.trace.cache_hit is True
    assert second.answer == first.answer
    assert embedder.calls == 1

    store.upsert([_chunk("2.0", "6", "500000 tokens allocation.")])
    third, _, _, _ = _ask("how many tokens?", **shared)
    assert third.trace.cache_hit is False
    assert embedder.calls == 1


def test_use_cache_false_runs_the_pipeline_again() -> None:
    caches = PolicyCaches()
    decider = ScriptedDecider({"fact": "none", "edition": "unspecified"})
    _ask("how many tokens?", caches=caches, decider=decider, use_cache=False)
    _ask("how many tokens?", caches=caches, decider=decider, use_cache=False)

    assert _kinds(decider).count("fact") == 2


def test_rerank_score_cache_skips_the_second_model_call() -> None:
    cache = RerankScoreCache()
    hit = FusedHit(
        chunk_id="time-usage:v2.0:section-6",
        document_slug="time-usage",
        document="Time & Usage Policy",
        version="2.0",
        section="6",
        section_title="Token Allocation",
        parent_section=None,
        parent_title=None,
        text="500000 tokens.",
        rrf_score=0.1,
    )
    reranker = FlatReranker(2.0)

    rerank("tokens", [hit], reranker, cache=cache, model_name="test-rerank")
    rerank("tokens", [hit], reranker, cache=cache, model_name="test-rerank")

    assert reranker.calls == 1


def test_query_vector_cache_skips_the_second_embed() -> None:
    inner = CountingEmbedder()
    cache = QueryVectorCache()
    embedder = CachingEmbedder(inner, cache, "test-embed")

    first = embedder.embed_texts(["tokens"])
    second = embedder.embed_texts(["tokens"])

    assert first == second
    assert inner.calls == 1


def test_parse_generation_result_reads_citations_and_clears_them_on_refusal() -> None:
    cited = parse_generation_result(
        '{"answer": "500000", "sufficient": true, "section": "6", '
        '"citations": [{"document": "Time & Usage Policy", "version": "2.0", '
        '"section": "6"}]}'
    )
    refused = parse_generation_result(
        '{"answer": "no", "sufficient": false, "section": "6", '
        '"citations": [{"document": "Time & Usage Policy", "version": "2.0", '
        '"section": "6"}]}'
    )

    assert cited.citations == (Citation("Time & Usage Policy", "2.0", "6"),)
    assert refused.citations == ()
    assert refused.section is None


def test_policy_generator_sets_temperature_zero() -> None:
    captured: dict[str, object] = {}

    def post(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        captured["payload"] = payload
        return {"message": {"content": '{"answer": "no", "sufficient": false}'}}

    OllamaGenerator(
        host="http://example",
        model="fake",
        system_prompt=SYSTEM_PROMPT,
        build_user_prompt=build_generation_prompt,
        temperature=0,
        post=post,
    ).generate("tokens", [])

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["options"] == {"temperature": 0}
