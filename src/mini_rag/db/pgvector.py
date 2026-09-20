from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Any

from mini_rag.models import ChunkRecord

_UPSERT_SQL = """
INSERT INTO chunks (
    chunk_id, document, version, section, section_title, text, embedding
) VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (chunk_id) DO UPDATE SET
    document = EXCLUDED.document,
    version = EXCLUDED.version,
    section = EXCLUDED.section,
    section_title = EXCLUDED.section_title,
    text = EXCLUDED.text,
    embedding = EXCLUDED.embedding,
    ingested_at = now()
"""

_DELETE_STALE_SQL = """
DELETE FROM chunks
WHERE document = %s
  AND chunk_id NOT IN ({placeholders})
"""


class PgVectorDatabase:
    """Postgres adapter. Open with a context manager; one transaction per upsert."""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._conn: Any = None

    def __enter__(self) -> PgVectorDatabase:
        import psycopg
        from pgvector.psycopg import register_vector

        connection = psycopg.connect(self._database_url)
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

    def upsert(self, records: Sequence[ChunkRecord]) -> None:
        if not records:
            return
        connection = self._require_connection()
        document = _single_document(records)
        chunk_ids = [record.chunk_id for record in records]
        placeholders = ",".join(["%s"] * len(chunk_ids))
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.executemany(
                    _UPSERT_SQL,
                    [
                        (
                            record.chunk_id,
                            record.document,
                            record.version,
                            record.section,
                            record.section_title,
                            record.text,
                            record.embedding,
                        )
                        for record in records
                    ],
                )
                cursor.execute(
                    _DELETE_STALE_SQL.format(placeholders=placeholders),
                    (document, *chunk_ids),
                )

    def count(self) -> int:
        connection = self._require_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM chunks")
            row = cursor.fetchone()
        if row is None:
            raise RuntimeError("COUNT(*) returned no row")
        return int(row[0])

    def _require_connection(self) -> Any:
        if self._conn is None:
            raise RuntimeError("Database connection is not open")
        return self._conn


def _single_document(records: Sequence[ChunkRecord]) -> str:
    documents = {record.document for record in records}
    if len(documents) != 1:
        raise ValueError("upsert expects records from a single document")
    return next(iter(documents))
