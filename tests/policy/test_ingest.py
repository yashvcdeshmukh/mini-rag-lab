from __future__ import annotations

from pathlib import Path

import pytest

from fakes import FakeEmbedder
from mini_rag.db import InMemoryPolicyStore
from mini_rag.policy.facts import CHECKED_FACTS, validate_facts
from mini_rag.policy.ingest import ingest_corpus
from mini_rag.policy.models import POLICY_EMBEDDING_DIM, PolicyChunk

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_DOCS = REPO_ROOT / "policy_docs"


class FixedCounter:
    def count(self, text: str) -> int:
        return 10


class BoomEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__(dimension=POLICY_EMBEDDING_DIM)
        self.calls = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return super().embed_texts(texts)


def test_ingest_stores_both_versions_and_facts() -> None:
    store = InMemoryPolicyStore()

    summaries = ingest_corpus(
        POLICY_DOCS,
        FakeEmbedder(dimension=POLICY_EMBEDDING_DIM),
        store,
        FixedCounter(),
    )

    assert len(summaries) == 7
    assert store.chunks_for("time-usage", "1.0")
    assert store.chunks_for("time-usage", "2.0")
    assert store.facts[("time-usage", "1.0", "token_allocation")].value == "1000000"
    assert store.facts[("time-usage", "2.0", "token_allocation")].value == "500000"
    assert store.facts[("hr", "1.0", "pet_leave_days")].value == "5"
    assert store.facts[("hr", "2.0", "pet_leave_days")].value == "5"
    sample = next(iter(store.chunks.values()))
    assert len(sample.embedding) == POLICY_EMBEDDING_DIM


def test_reingest_embeds_again_and_keeps_both_versions() -> None:
    store = InMemoryPolicyStore()
    counter = FixedCounter()
    ingest_corpus(
        POLICY_DOCS,
        FakeEmbedder(dimension=POLICY_EMBEDDING_DIM),
        store,
        counter,
    )
    v1_ids = {chunk.chunk_id for chunk in store.chunks_for("time-usage", "1.0")}
    embedder = BoomEmbedder()

    ingest_corpus(POLICY_DOCS, embedder, store, counter)

    assert embedder.calls == 1
    assert {chunk.chunk_id for chunk in store.chunks_for("time-usage", "1.0")} == v1_ids


def test_reingest_of_one_edition_leaves_the_other(tmp_path: Path) -> None:
    store = InMemoryPolicyStore()
    counter = FixedCounter()
    embedder = FakeEmbedder(dimension=POLICY_EMBEDDING_DIM)
    ingest_corpus(POLICY_DOCS, embedder, store, counter)
    v2_only = tmp_path / "docs"
    v2_only.mkdir()
    source = next(POLICY_DOCS.glob("*Time and Usage Policy v2*"))
    (v2_only / source.name).write_bytes(source.read_bytes())

    ingest_corpus(v2_only, embedder, store, counter)

    assert store.chunks_for("time-usage", "1.0")
    assert store.chunks_for("time-usage", "2.0")
    assert ("time-usage", "1.0", "token_allocation") in store.facts


def test_fact_for_missing_section_fails() -> None:
    chunk = PolicyChunk(
        chunk_id="hr:v1.0:section-1",
        document_slug="hr",
        document="HR Policy",
        version="1.0",
        section="1",
        section_title="Purpose",
        parent_section=None,
        parent_title=None,
        text="Purpose text.",
        embed_input="HR Policy v1.0 — 1. Purpose\nPurpose text.",
        source_file="hr.pdf",
        search_text="Purpose\nPurpose text.",
        embedding=(1.0, 0.0),
    )

    with pytest.raises(ValueError, match="was not chunked"):
        validate_facts(CHECKED_FACTS, [chunk])


def test_ingest_rejects_wrong_embedding_width(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.docx").write_text("not a real docx")

    with pytest.raises(ValueError, match="768-dimensional"):
        ingest_corpus(
            docs,
            FakeEmbedder(dimension=384),
            InMemoryPolicyStore(),
            FixedCounter(),
        )
