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
    embedding: list[float] | None = None,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=make_chunk_id(version, section),
        document="Employee Expense Policy",
        version=version,
        section=section,
        section_title=title,
        text=text,
        embedding=embedding if embedding is not None else [1.0, 0.0],
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


def test_pgvector_search_requires_open_connection() -> None:
    db = PgVectorDatabase("postgresql://unused")

    with pytest.raises(RuntimeError, match="not open"):
        db.search([1.0, 0.0])


def _ranked_records() -> list[ChunkRecord]:
    """Six distinct, non-unit vectors so cosine cannot cheat with 1 - dot."""
    return [
        _record(
            version="2.0",
            section="1",
            title="Meals",
            text="Meals policy text.",
            embedding=[2.0, 0.0],
        ),
        _record(
            version="2.0",
            section="2",
            title="Hotels",
            text="Hotels policy text.",
            embedding=[2.0, 1.0],
        ),
        _record(
            version="2.0",
            section="3",
            title="Airfare",
            text="Airfare policy text.",
            embedding=[1.0, 2.0],
        ),
        _record(
            version="2.0",
            section="4",
            title="Ground Transportation",
            text="Ground policy text.",
            embedding=[0.0, 2.0],
        ),
        _record(
            version="2.0",
            section="5",
            title="Receipts",
            text="Receipts policy text.",
            embedding=[-2.0, 0.0],
        ),
        _record(
            version="2.0",
            section="6",
            title="Submission Deadline",
            text="Deadline policy text.",
            embedding=[0.0, -2.0],
        ),
    ]


def test_search_returns_three_nearest_meals_first() -> None:
    db = InMemoryDatabase()
    db.upsert(_ranked_records())

    results = db.search([1.0, 0.0], k=3)

    assert [chunk.section_title for chunk in results] == [
        "Meals",
        "Hotels",
        "Airfare",
    ]
    assert [chunk.section for chunk in results] == ["1", "2", "3"]
    assert all(isinstance(chunk.distance, float) for chunk in results)
    assert results[0].distance < results[1].distance < results[2].distance
    assert results[0].distance == pytest.approx(0.0)
    assert len(results) == 3


def test_search_does_not_clamp_k() -> None:
    db = InMemoryDatabase()
    db.upsert(_ranked_records())

    results = db.search([1.0, 0.0], k=6)

    assert len(results) == 6
    assert results[-1].section_title == "Receipts"


def test_search_uses_full_cosine_on_non_unit_vectors() -> None:
    db = InMemoryDatabase()
    db.upsert(
        [
            _record(
                version="2.0",
                section="1",
                title="Meals",
                text="Meals.",
                embedding=[3.0, 4.0],
            ),
            _record(
                version="2.0",
                section="2",
                title="Hotels",
                text="Hotels.",
                embedding=[0.0, 1.0],
            ),
        ]
    )

    results = db.search([6.0, 8.0], k=1)

    assert results[0].section_title == "Meals"
    assert results[0].distance == pytest.approx(0.0)
