from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from mini_rag.adapters import Embedder
from mini_rag.db import InMemoryPolicyStore, PgPolicyStore
from mini_rag.embedder import SentenceTransformerEmbedder, validate_embeddings
from mini_rag.policy.chunking import TokenCounter, chunk_document
from mini_rag.policy.config import policy_database_url
from mini_rag.policy.extract import load_policy_dir
from mini_rag.policy.facts import CHECKED_FACTS, validate_facts
from mini_rag.policy.models import (
    BGE_MODEL_NAME,
    POLICY_EMBEDDING_DIM,
    IngestFileSummary,
    PolicyChunk,
)

logger = logging.getLogger(__name__)


class HuggingFaceTokenCounter:
    """Counts tokens with the embedding model's tokenizer. Does not load the model."""

    def __init__(self, model_name: str = BGE_MODEL_NAME) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:
        encoded: list[int] = self._tokenizer.encode(text, add_special_tokens=False)
        return len(encoded)


def ingest_corpus(
    docs_dir: Path,
    embedder: Embedder,
    store: InMemoryPolicyStore | PgPolicyStore,
    counter: TokenCounter,
) -> list[IngestFileSummary]:
    if embedder.dimension != POLICY_EMBEDDING_DIM:
        raise ValueError(
            f"Policy embeddings must be {POLICY_EMBEDDING_DIM}-dimensional, "
            f"got {embedder.dimension}"
        )
    documents = load_policy_dir(docs_dir)
    chunks: list[PolicyChunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, counter))
    editions = {(chunk.document_slug, chunk.version) for chunk in chunks}
    facts = [
        fact
        for fact in CHECKED_FACTS
        if (fact.document_slug, fact.version) in editions
    ]
    validate_facts(facts, chunks)

    vectors = embedder.embed_texts([chunk.embed_input for chunk in chunks])
    validate_embeddings(vectors, expected_dim=embedder.dimension)
    embedded = [
        chunk.with_embedding(vector)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    logger.info(
        "Computed %s embeddings with dimension %s",
        len(vectors),
        embedder.dimension,
    )
    store.upsert(embedded, facts)

    summaries: list[IngestFileSummary] = []
    for document in documents:
        file_chunks = [
            chunk for chunk in embedded if chunk.source_file == document.source_file
        ]
        summary = IngestFileSummary(
            source_file=document.source_file,
            document_slug=file_chunks[0].document_slug,
            version=file_chunks[0].version,
            chunks=len(file_chunks),
        )
        summaries.append(summary)
        logger.info(
            "%s %s v%s chunks=%s",
            summary.source_file,
            summary.document_slug,
            summary.version,
            summary.chunks,
        )
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Ingest Doofenshmirtz policy documents into policy_rag."
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=Path("policy_docs"),
        help="Directory of policy PDF and Word files",
    )
    args = parser.parse_args(argv)

    try:
        with PgPolicyStore(policy_database_url()) as store:
            ingest_corpus(
                args.docs,
                SentenceTransformerEmbedder(BGE_MODEL_NAME),
                store,
                HuggingFaceTokenCounter(BGE_MODEL_NAME),
            )
    except Exception as exc:
        logger.error("Ingest failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
