from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from mini_rag.adapters import DatabaseAdapter, Embedder
from mini_rag.db import PgVectorDatabase
from mini_rag.embedder import SentenceTransformerEmbedder
from mini_rag.expense.chunking import parse_policy_file
from mini_rag.expense.config import database_url
from mini_rag.expense.models import ChunkRecord, RetrievedChunk, build_chunk_records

EXPECTED_CHUNKS = 6
logger = logging.getLogger(__name__)


def ingest_policy(
    policy_path: Path,
    embedder: Embedder,
    database: DatabaseAdapter[ChunkRecord, RetrievedChunk],
    *,
    expected_chunks: int = EXPECTED_CHUNKS,
) -> list[ChunkRecord]:
    policy = parse_policy_file(policy_path)
    logger.info("Parsed %s sections from %s", len(policy.sections), policy_path)

    texts = [section.text for section in policy.sections]
    embeddings = embedder.embed_texts(texts)
    logger.info(
        "Computed %s embeddings with dimension %s",
        len(embeddings),
        embedder.dimension,
    )

    records = build_chunk_records(policy, embeddings)
    database.upsert(records)
    stored = database.count()
    if stored != expected_chunks:
        raise ValueError(f"Expected {expected_chunks} stored chunks, found {stored}")
    logger.info("Upserted %s rows", stored)
    return records


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Ingest the expense policy into pgvector."
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("policy.md"),
        help="Path to the policy markdown file",
    )
    args = parser.parse_args(argv)

    try:
        with PgVectorDatabase(database_url()) as database:
            ingest_policy(args.policy, SentenceTransformerEmbedder(), database)
    except Exception as exc:
        logger.error("Ingest failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
