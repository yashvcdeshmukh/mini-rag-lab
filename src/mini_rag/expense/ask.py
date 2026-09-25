from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from typing import TypedDict

from mini_rag.adapters import (
    DatabaseAdapter,
    Embedder,
    GenerationResult,
    Generator,
)
from mini_rag.db import PgVectorDatabase
from mini_rag.embedder import SentenceTransformerEmbedder
from mini_rag.expense.config import database_url
from mini_rag.expense.generation import (
    REFUSAL_ANSWER,
    SYSTEM_PROMPT,
    build_generation_prompt,
)
from mini_rag.expense.models import ChunkRecord, RetrievedChunk
from mini_rag.generator import OllamaGenerator

DEFAULT_K = 3
logger = logging.getLogger(__name__)


class RetrievedChunkView(TypedDict):
    section: str
    distance: float


class CitationView(TypedDict):
    document: str
    version: str
    section: str


class AskResponse(TypedDict):
    answer: str
    citation: CitationView | None
    retrieved_chunks: list[RetrievedChunkView]


def format_section(chunk: RetrievedChunk) -> str:
    return f"{chunk.section}. {chunk.section_title}"


def build_ask_response(
    generation: GenerationResult,
    chunks: Sequence[RetrievedChunk],
) -> AskResponse:
    retrieved = list(chunks)[:DEFAULT_K]
    views: list[RetrievedChunkView] = [
        {"section": format_section(chunk), "distance": float(chunk.distance)}
        for chunk in retrieved
    ]
    retrieved_sections = {chunk.section for chunk in retrieved}
    if generation.sufficient and generation.section in retrieved_sections:
        match = next(
            chunk for chunk in retrieved if chunk.section == generation.section
        )
        return {
            "answer": generation.answer,
            "citation": {
                "document": match.document,
                "version": match.version,
                "section": format_section(match),
            },
            "retrieved_chunks": views,
        }
    return {
        "answer": REFUSAL_ANSWER,
        "citation": None,
        "retrieved_chunks": views,
    }


def ask_question(
    question: str,
    embedder: Embedder,
    database: DatabaseAdapter[ChunkRecord, RetrievedChunk],
    generator: Generator[RetrievedChunk],
    *,
    k: int = DEFAULT_K,
) -> AskResponse:
    logger.info("Question: %s", question)
    query_vector = embedder.embed_texts([question])[0]
    chunks = database.search(query_vector, k=k)[:DEFAULT_K]
    logger.info(
        "Retrieved %s",
        [
            f"{format_section(chunk)} ({chunk.distance:.4f})"
            for chunk in chunks
        ],
    )
    generation = generator.generate(question, chunks)
    response = build_ask_response(generation, chunks)
    logger.info("Done")
    return response


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Answer a question from retrieved expense-policy chunks."
    )
    parser.add_argument("--question", required=True, help="User question")
    args = parser.parse_args(argv)

    try:
        with PgVectorDatabase(database_url()) as database:
            response = ask_question(
                args.question,
                SentenceTransformerEmbedder(),
                database,
                OllamaGenerator(
                    system_prompt=SYSTEM_PROMPT,
                    build_user_prompt=build_generation_prompt,
                ),
            )
    except Exception as exc:
        logger.error("Ask failed: %s", exc)
        return 1
    json.dump(response, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
