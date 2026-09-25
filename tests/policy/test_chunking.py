from __future__ import annotations

from pathlib import Path

import pytest

from mini_rag.policy.chunking import (
    MAX_EMBED_TOKENS,
    chunk_document,
    parse_policy_text,
)
from mini_rag.policy.extract import load_policy_dir
from mini_rag.policy.models import ExtractedDocument, document_slug

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_DOCS = REPO_ROOT / "policy_docs"


class LengthCounter:
    def count(self, text: str) -> int:
        return len(text)


class FixedCounter:
    def __init__(self, tokens: int) -> None:
        self._tokens = tokens

    def count(self, text: str) -> int:
        return self._tokens


def test_slug_drops_policy_suffix_and_ampersand() -> None:
    assert document_slug("Time & Usage Policy") == "time-usage"
    assert document_slug("Health & Wellness Policy") == "health-wellness"
    assert document_slug("HR Policy") == "hr"
    assert document_slug("Preparedness Policy") == "preparedness"


def test_short_sections_stay_whole() -> None:
    document = ExtractedDocument(
        source_file="hr.txt",
        text=(
            "HR Policy — Version 2.0\n"
            "1. Purpose\n"
            "This policy stands alone.\n"
            "2. Scope\n"
            "It applies to everyone.\n"
        ),
    )

    chunks = chunk_document(document, FixedCounter(10))

    assert [chunk.section for chunk in chunks] == ["1", "2"]
    assert chunks[0].chunk_id == "hr:v2.0:section-1"
    assert chunks[0].parent_section is None
    assert chunks[0].text == "This policy stands alone."
    assert chunks[0].embed_input.startswith("HR Policy v2.0 — 1. Purpose\n")
    assert "HR Policy v2.0" not in chunks[0].text
    assert chunks[0].embedding == ()


def test_over_limit_section_splits_on_subsections() -> None:
    intro = "word " * 80
    document = ExtractedDocument(
        source_file="prep.txt",
        text=(
            "Preparedness Policy — Version 2.0\n"
            "4. Nuclear Apocalypse Protocol\n"
            f"4.1 Shelter Location. {intro}\n"
            f"4.2 Duration. {intro}\n"
        ),
    )

    chunks = chunk_document(document, LengthCounter())

    assert [chunk.section for chunk in chunks] == ["4.1", "4.2"]
    assert {chunk.parent_section for chunk in chunks} == {"4"}
    assert {chunk.parent_title for chunk in chunks} == {"Nuclear Apocalypse Protocol"}
    assert chunks[0].chunk_id == "preparedness:v2.0:section-4.1"
    prefix = (
        "Preparedness Policy v2.0 — 4. Nuclear Apocalypse Protocol"
        " — 4.1 Shelter Location\n"
    )
    assert chunks[0].embed_input.startswith(prefix)
    assert all(len(chunk.embed_input) <= MAX_EMBED_TOKENS for chunk in chunks)


def test_over_limit_subsection_windows_with_overlap() -> None:
    sentence = "Employees must remain indoors for the full shelter period. "
    document = ExtractedDocument(
        source_file="prep.txt",
        text=(
            "Preparedness Policy — Version 2.0\n"
            "4. Nuclear Apocalypse Protocol\n"
            f"4.1 Shelter Location. {sentence * 30}\n"
        ),
    )

    chunks = chunk_document(document, LengthCounter())

    assert len(chunks) >= 2
    assert all(chunk.section == "4.1" for chunk in chunks)
    assert chunks[0].chunk_id == "preparedness:v2.0:section-4.1:part-1"
    assert chunks[1].chunk_id == "preparedness:v2.0:section-4.1:part-2"
    first_sentence = chunks[0].text.split(". ")[-1]
    assert chunks[1].text.startswith(first_sentence)


def test_parse_requires_header_and_body() -> None:
    with pytest.raises(ValueError, match="title and version"):
        parse_policy_text("1. Purpose\nSome text.")

    with pytest.raises(ValueError, match="no body text"):
        parse_policy_text("HR Policy — Version 2.0\n1. Purpose\n")


def test_real_policies_chunk_one_section_each() -> None:
    documents = load_policy_dir(POLICY_DOCS)
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document, FixedCounter(10))
    ]

    by_file = {document.source_file: document for document in documents}
    assert len(by_file) == 7
    health = next(
        chunk
        for chunk in chunks
        if chunk.document_slug == "health-wellness" and chunk.section == "4"
    )
    assert "Height Range" in health.text
    assert "190" in health.text
    caffeine = next(
        chunk
        for chunk in chunks
        if chunk.document_slug == "health-wellness" and chunk.section == "5"
    )
    assert "190" not in caffeine.text
    versions = {(chunk.document_slug, chunk.version) for chunk in chunks}
    assert ("time-usage", "1.0") in versions
    assert ("time-usage", "2.0") in versions
    assert all(chunk.parent_section is None for chunk in chunks)
    assert all(":part-" not in chunk.chunk_id for chunk in chunks)
