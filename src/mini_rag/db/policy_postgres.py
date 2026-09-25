from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Any

from mini_rag.db.policy_memory import _require_embeddings
from mini_rag.policy.models import (
    KeywordHit,
    PolicyChunk,
    PolicyFact,
    RetrievedPolicyChunk,
)

_UPSERT_CHUNK_SQL = """
INSERT INTO policy_chunks (
    chunk_id, document_slug, document, version, section, section_title,
    parent_section, parent_title, text, embedding, source_file, search_text
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
)
ON CONFLICT (chunk_id) DO UPDATE SET
    document_slug = EXCLUDED.document_slug,
    document = EXCLUDED.document,
    version = EXCLUDED.version,
    section = EXCLUDED.section,
    section_title = EXCLUDED.section_title,
    parent_section = EXCLUDED.parent_section,
    parent_title = EXCLUDED.parent_title,
    text = EXCLUDED.text,
    embedding = EXCLUDED.embedding,
    source_file = EXCLUDED.source_file,
    search_text = EXCLUDED.search_text,
    ingested_at = now()
"""

_SEARCH_SQL = """
SELECT chunk_id, document_slug, document, version, section, section_title,
       parent_section, parent_title, text,
       embedding <=> %s::vector AS distance
FROM policy_chunks
ORDER BY embedding <=> %s::vector ASC
LIMIT %s
"""

_KEYWORD_SQL = """
SELECT chunk.chunk_id, chunk.document_slug, chunk.document, chunk.version,
       chunk.section, chunk.section_title, chunk.parent_section,
       chunk.parent_title, chunk.text,
       ts_rank(to_tsvector('english', chunk.search_text), query.tsq) AS rank
FROM policy_chunks AS chunk
JOIN (
    SELECT to_tsquery('english', array_to_string(lexemes, ' | ')) AS tsq
    FROM (
        SELECT tsvector_to_array(to_tsvector('english', %s)) AS lexemes
    ) AS raw
    WHERE cardinality(raw.lexemes) > 0
) AS query ON to_tsvector('english', chunk.search_text) @@ query.tsq
ORDER BY rank DESC, chunk.chunk_id ASC
LIMIT %s
"""

_UPSERT_FACT_SQL = """
INSERT INTO policy_facts (document_slug, version, fact_key, value, section)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (document_slug, version, fact_key) DO UPDATE SET
    value = EXCLUDED.value,
    section = EXCLUDED.section
"""


class PgPolicyStore:
    """Postgres adapter for policy_rag. One transaction per corpus upsert."""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._conn: Any = None

    def __enter__(self) -> PgPolicyStore:
        import psycopg
        from pgvector.psycopg import register_vector

        connection = psycopg.connect(self._database_url, autocommit=True)
        register_vector(connection)
        self._conn = connection
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        connection = self._conn
        self._conn = None
        if connection is not None:
            connection.close()

    def upsert(
        self,
        records: Sequence[PolicyChunk],
        facts: Sequence[PolicyFact] | None = None,
    ) -> None:
        if not records:
            raise ValueError("upsert requires at least one chunk")
        _require_embeddings(records)
        connection = self._require_connection()
        pairs = sorted({(chunk.document_slug, chunk.version) for chunk in records})
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.executemany(
                    _UPSERT_CHUNK_SQL, [_chunk_params(chunk) for chunk in records]
                )
                _delete_stale_chunks(cursor, records, pairs)
                if facts is not None:
                    if facts:
                        cursor.executemany(
                            _UPSERT_FACT_SQL, [_fact_params(fact) for fact in facts]
                        )
                    _delete_stale_facts(cursor, facts, pairs)

    def search(
        self, query_vector: Sequence[float], k: int = 3
    ) -> list[RetrievedPolicyChunk]:
        connection = self._require_connection()
        query = list(query_vector)
        with connection.cursor() as cursor:
            cursor.execute(_SEARCH_SQL, (query, query, k))
            rows = cursor.fetchall()
        return [
            RetrievedPolicyChunk(
                chunk_id=row[0],
                document_slug=row[1],
                document=row[2],
                version=row[3],
                section=row[4],
                section_title=row[5],
                parent_section=row[6],
                parent_title=row[7],
                text=row[8],
                distance=float(row[9]),
            )
            for row in rows
        ]

    def search_keyword(self, question: str, k: int = 20) -> list[KeywordHit]:
        connection = self._require_connection()
        with connection.cursor() as cursor:
            cursor.execute(_KEYWORD_SQL, (question, k))
            rows = cursor.fetchall()
        return [
            KeywordHit(
                chunk_id=row[0],
                document_slug=row[1],
                document=row[2],
                version=row[3],
                section=row[4],
                section_title=row[5],
                parent_section=row[6],
                parent_title=row[7],
                text=row[8],
                rank=float(row[9]),
            )
            for row in rows
        ]

    def count(self) -> int:
        connection = self._require_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM policy_chunks")
            row = cursor.fetchone()
        if row is None:
            return 0
        return int(row[0])

    def corpus_stamp(self) -> tuple[int, str]:
        connection = self._require_connection()
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT count(*), COALESCE(max(ingested_at)::text, '')
                FROM policy_chunks
                """
            )
            row = cursor.fetchone()
        if row is None:
            return (0, "")
        return (int(row[0]), str(row[1]))

    def facts_for(self, fact_key: str) -> list[PolicyFact]:
        connection = self._require_connection()
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT document_slug, version, fact_key, value, section
                FROM policy_facts
                WHERE fact_key = %s
                ORDER BY document_slug, version
                """,
                (fact_key,),
            )
            rows = cursor.fetchall()
        return [
            PolicyFact(
                document_slug=row[0],
                version=row[1],
                fact_key=row[2],
                value=row[3],
                section=row[4],
            )
            for row in rows
        ]

    def _require_connection(self) -> Any:
        if self._conn is None:
            raise RuntimeError("Database connection is not open")
        return self._conn


def _chunk_params(chunk: PolicyChunk) -> tuple[Any, ...]:
    return (
        chunk.chunk_id,
        chunk.document_slug,
        chunk.document,
        chunk.version,
        chunk.section,
        chunk.section_title,
        chunk.parent_section,
        chunk.parent_title,
        chunk.text,
        list(chunk.embedding),
        chunk.source_file,
        chunk.search_text,
    )


def _fact_params(fact: PolicyFact) -> tuple[str, str, str, str, str]:
    return (
        fact.document_slug,
        fact.version,
        fact.fact_key,
        fact.value,
        fact.section,
    )


def _delete_stale_chunks(
    cursor: Any, chunks: Sequence[PolicyChunk], pairs: list[tuple[str, str]]
) -> None:
    for slug, version in pairs:
        ids = [
            chunk.chunk_id
            for chunk in chunks
            if chunk.document_slug == slug and chunk.version == version
        ]
        cursor.execute(
            """
            DELETE FROM policy_chunks
            WHERE document_slug = %s AND version = %s AND NOT (chunk_id = ANY(%s))
            """,
            (slug, version, ids),
        )


def _delete_stale_facts(
    cursor: Any, facts: Sequence[PolicyFact], pairs: list[tuple[str, str]]
) -> None:
    for slug, version in pairs:
        keys = [
            fact.fact_key
            for fact in facts
            if fact.document_slug == slug and fact.version == version
        ]
        if not keys:
            cursor.execute(
                "DELETE FROM policy_facts WHERE document_slug = %s AND version = %s",
                (slug, version),
            )
            continue
        cursor.execute(
            """
            DELETE FROM policy_facts
            WHERE document_slug = %s AND version = %s AND NOT (fact_key = ANY(%s))
            """,
            (slug, version, keys),
        )
