from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from fakes import FakeEmbedder
from mini_rag.db import InMemoryPolicyStore, PgPolicyStore
from mini_rag.policy.models import POLICY_EMBEDDING_DIM, PolicyChunk
from mini_rag.policy.retrieve import (
    QUERY_PREFIX,
    explicit_version,
    hybrid_finds_what_dense_misses,
    reciprocal_rank_fusion,
    retrieve,
)

MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "migrations"
    / "policy"
    / "001_policy_chunks.sql"
)


class RecordingEmbedder(FakeEmbedder):
    def __init__(self, dimension: int) -> None:
        super().__init__(dimension=dimension)
        self.texts: list[str] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return super().embed_texts(texts)


def _chunk(
    version: str,
    section: str,
    text: str,
    *,
    slug: str = "time-usage",
    search_text: str | None = None,
) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=f"{slug}:v{version}:section-{section}",
        document_slug=slug,
        document="Time & Usage Policy",
        version=version,
        section=section,
        section_title="Token Allocation",
        parent_section=None,
        parent_title=None,
        text=text,
        embed_input=text,
        source_file=f"{slug}-v{version}.docx",
        search_text=text if search_text is None else search_text,
        embedding=(1.0, 0.0),
    )


def test_hybrid_finds_a_chunk_dense_search_misses() -> None:
    store = InMemoryPolicyStore()
    store.upsert(
        [
            _chunk("1.0", "1", "general office hours and meetings."),
            _chunk("1.0", "4", "nuclear shelter under the conference table."),
        ]
    )

    result = retrieve(
        "nuclear shelter conference table",
        RecordingEmbedder(2),
        store,
        k=2,
        fused_k=1,
    )

    assert result.dense[0].chunk_id == "time-usage:v1.0:section-1"
    assert hybrid_finds_what_dense_misses(
        result, "time-usage:v1.0:section-4", k=1
    )


def test_fusion_prefers_chunk_in_both_lists() -> None:
    fused = reciprocal_rank_fusion(
        [["both", "dense-only"], ["both", "sparse-only"]]
    )

    assert fused[0][0] == "both"
    assert fused[0][1] > fused[1][1]


def test_retrieve_keeps_ten_fused_chunks() -> None:
    store = InMemoryPolicyStore()
    chunks = [
        _chunk("1.0", str(index), f"body {index}.") for index in range(1, 13)
    ]
    store.upsert(chunks)

    result = retrieve("unrelated question", RecordingEmbedder(2), store)

    assert len(result.dense) == 12
    assert len(result.fused) == 10
    assert [hit.chunk_id for hit in result.fused] == [
        chunk.chunk_id for chunk in chunks[:10]
    ]


def test_retrieve_keeps_both_editions_when_version_is_unspecified() -> None:
    store = InMemoryPolicyStore()
    store.upsert(
        [
            _chunk("1.0", "5", "1000000 tokens."),
            _chunk("2.0", "6", "500000 tokens."),
        ]
    )

    result = retrieve(
        "how many tokens do I get?",
        RecordingEmbedder(2),
        store,
    )

    assert {hit.version for hit in result.fused} == {"1.0", "2.0"}


def test_retrieve_drops_other_edition_when_version_is_named() -> None:
    store = InMemoryPolicyStore()
    store.upsert(
        [
            _chunk("1.0", "5", "1000000 tokens."),
            _chunk("2.0", "6", "500000 tokens."),
        ]
    )

    result = retrieve(
        "how many tokens in version 2.0",
        RecordingEmbedder(2),
        store,
    )

    assert [hit.version for hit in result.fused] == ["2.0"]
    assert explicit_version("previous policy") is None


def test_retrieve_logs_each_step(caplog: pytest.LogCaptureFixture) -> None:
    store = InMemoryPolicyStore()
    store.upsert(
        [
            _chunk("1.0", "5", "1000000 tokens allocation."),
            _chunk("2.0", "6", "500000 tokens allocation."),
        ]
    )

    with caplog.at_level(logging.INFO):
        retrieve("how many tokens in version 2.0", RecordingEmbedder(2), store)

    messages = [record.getMessage() for record in caplog.records]
    assert messages[0] == "Retrieve how many tokens in version 2.0"
    assert any(message.startswith("Dense ") for message in messages)
    assert any(message.startswith("Keyword ") for message in messages)
    assert any(message.startswith("RRF ") for message in messages)
    assert any(
        message.startswith("Version filter 2.0 kept ") for message in messages
    )
    assert any(message.startswith("Fused ") for message in messages)
    assert all("allocation" not in message for message in messages)


def test_retrieve_keeps_both_editions_for_previous() -> None:
    store = InMemoryPolicyStore()
    store.upsert(
        [
            _chunk("1.0", "5", "1000000 tokens."),
            _chunk("2.0", "6", "500000 tokens."),
        ]
    )

    result = retrieve(
        "how many tokens in the previous policy",
        RecordingEmbedder(2),
        store,
    )

    assert {hit.version for hit in result.fused} == {"1.0", "2.0"}


def test_retrieve_embeds_the_prefixed_question() -> None:
    store = InMemoryPolicyStore()
    store.upsert([_chunk("2.0", "6", "500000 tokens.")])
    embedder = RecordingEmbedder(2)
    question = "how many tokens do I get?"

    retrieve(question, embedder, store)

    assert embedder.texts == [f"{QUERY_PREFIX}{question}"]


def test_in_memory_keyword_search_ranks_rare_word_first() -> None:
    store = InMemoryPolicyStore()
    rare = _chunk("2.0", "6", "Limit caffeine to 400 mg.", slug="health-wellness")
    common = _chunk(
        "1.0",
        "1",
        "This policy applies to every employee.",
        slug="hr",
        search_text="Purpose\nThis policy applies to every employee.",
    )
    store.upsert([common, rare])

    hits = store.search_keyword("what is the caffeine limit", k=20)

    assert hits[0].chunk_id == rare.chunk_id
    assert common.chunk_id not in {hit.chunk_id for hit in hits}
    assert store.search_keyword("the") == []


def test_explicit_version_ignores_a_bare_number() -> None:
    assert explicit_version("section 2 allows 2 hours") is None
    assert explicit_version("see v1 and version 2.0") is None
    assert explicit_version("v1.0") == "1.0"


@pytest.mark.integration
def test_pg_keyword_search_ors_terms_and_keeps_both_editions() -> None:
    import psycopg

    from mini_rag.policy.config import policy_database_url

    url = policy_database_url()
    with psycopg.connect(url) as connection:
        connection.execute(MIGRATION.read_text(encoding="utf-8"))
        connection.commit()

    width = (1.0,) + (0.0,) * (POLICY_EMBEDDING_DIM - 1)
    first = replace(
        _chunk("1.0", "1", "zyzzx allocation.", slug="keyword-probe"),
        embedding=width,
    )
    second = replace(
        _chunk("2.0", "1", "zyzzx quota.", slug="keyword-probe"),
        embedding=width,
    )
    try:
        with PgPolicyStore(url) as store:
            store.upsert([first, second])
            hits = store.search_keyword("zyzzx employee", k=20)
        versions = {hit.version for hit in hits if hit.document_slug == "keyword-probe"}
        assert versions == {"1.0", "2.0"}
    finally:
        with psycopg.connect(url) as connection:
            connection.execute(
                "DELETE FROM policy_chunks WHERE document_slug = %s",
                ("keyword-probe",),
            )
            connection.commit()
