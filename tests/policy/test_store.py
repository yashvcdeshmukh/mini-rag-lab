from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mini_rag.db import InMemoryPolicyStore, PgPolicyStore
from mini_rag.policy.models import PolicyChunk, PolicyFact

MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "migrations"
    / "policy"
    / "001_policy_chunks.sql"
)


def _chunk(version: str, section: str, text: str) -> PolicyChunk:
    chunk_id = f"time-usage:v{version}:section-{section}"
    return PolicyChunk(
        chunk_id=chunk_id,
        document_slug="time-usage",
        document="Time & Usage Policy",
        version=version,
        section=section,
        section_title="Token Allocation",
        parent_section=None,
        parent_title=None,
        text=text,
        embed_input=(
            f"Time & Usage Policy v{version} — {section}. Token Allocation\n{text}"
        ),
        source_file=f"time-v{version}.docx",
        search_text=f"Token Allocation\n{text}",
        embedding=(1.0, 0.0),
    )


def test_in_memory_store_searches_nearest_and_counts() -> None:
    store = InMemoryPolicyStore()
    near = _chunk("2.0", "6", "500000 tokens.")
    far = replace(_chunk("1.0", "5", "1000000 tokens."), embedding=(0.0, 1.0))

    store.upsert([near, far])

    hits = store.search((1.0, 0.0), k=1)
    assert store.count() == 2
    assert hits[0].chunk_id == near.chunk_id
    assert hits[0].distance == 0.0


def test_pg_policy_store_requires_open_connection() -> None:
    store = PgPolicyStore("postgresql://unused")

    with pytest.raises(RuntimeError, match="not open"):
        store.count()


@pytest.mark.integration
def test_pg_policy_store_round_trip() -> None:
    import psycopg

    from mini_rag.policy.config import policy_database_url

    url = policy_database_url()
    with psycopg.connect(url) as connection:
        connection.execute(MIGRATION.read_text(encoding="utf-8"))
        connection.commit()

    first = _chunk("1.0", "5", "1000000 tokens.")
    second = _chunk("2.0", "6", "500000 tokens.")
    fact = PolicyFact("time-usage", "2.0", "token_allocation", "500000", "6")
    with PgPolicyStore(url) as store:
        store.upsert([first, second], [fact])
        store.upsert([second], [fact])

    with psycopg.connect(url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_id FROM policy_chunks
                WHERE document_slug = %s AND version = %s
                """,
                ("time-usage", "1.0"),
            )
            v1_rows = cursor.fetchall()
            cursor.execute(
                """
                SELECT value FROM policy_facts
                WHERE document_slug = %s AND version = %s AND fact_key = %s
                """,
                ("time-usage", "2.0", "token_allocation"),
            )
            fact_row = cursor.fetchone()
    assert v1_rows == [("time-usage:v1.0:section-5",)]
    assert fact_row == ("500000",)
