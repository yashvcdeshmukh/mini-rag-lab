from __future__ import annotations

import pytest

from fakes import InMemoryDatabase
from mini_rag.db import PgVectorDatabase
from mini_rag.models import ChunkRecord, make_chunk_id


def _record(
    *,
    version: str,
    section: str,
    title: str,
    text: str,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=make_chunk_id(version, section),
        document="Employee Expense Policy",
        version=version,
        section=section,
        section_title=title,
        text=text,
        embedding=[1.0, 0.0],
    )


def _six_records(version: str = "2.0") -> list[ChunkRecord]:
    titles = [
        "Meals",
        "Hotels",
        "Airfare",
        "Ground Transportation",
        "Receipts",
        "Submission Deadline",
    ]
    return [
        _record(
            version=version,
            section=str(number),
            title=title,
            text=f"{title} policy text.",
        )
        for number, title in enumerate(titles, start=1)
    ]


def test_upsert_stores_six_records() -> None:
    db = InMemoryDatabase()
    db.upsert(_six_records())

    assert db.count() == 6
    assert db.get("expense-policy:v2.0:section-1").section_title == "Meals"


def test_reingest_overwrites_text_and_keeps_six_rows() -> None:
    db = InMemoryDatabase()
    db.upsert(_six_records())
    updated = _six_records()
    updated[0] = _record(
        version="2.0",
        section="1",
        title="Meals",
        text="Updated meals rule.",
    )

    db.upsert(updated)

    assert db.count() == 6
    assert db.get("expense-policy:v2.0:section-1").text == "Updated meals rule."


def test_version_bump_deletes_stale_rows() -> None:
    db = InMemoryDatabase()
    db.upsert(_six_records("2.0"))

    db.upsert(_six_records("2.1"))

    assert db.count() == 6
    assert db.get("expense-policy:v2.1:section-1").version == "2.1"
    with pytest.raises(KeyError):
        db.get("expense-policy:v2.0:section-1")


def test_upsert_rejects_mixed_documents() -> None:
    db = InMemoryDatabase()
    left = _record(version="2.0", section="1", title="Meals", text="A.")
    right = ChunkRecord(
        chunk_id="other:v2.0:section-1",
        document="Other Policy",
        version="2.0",
        section="1",
        section_title="Meals",
        text="B.",
        embedding=[1.0, 0.0],
    )

    with pytest.raises(ValueError, match="single document"):
        db.upsert([left, right])


def test_pgvector_requires_open_connection() -> None:
    db = PgVectorDatabase("postgresql://unused")

    with pytest.raises(RuntimeError, match="not open"):
        db.count()
