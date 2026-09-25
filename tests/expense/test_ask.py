from __future__ import annotations

import json

import pytest

from expense.fakes import FakeGenerator, ScriptedEmbedder
from mini_rag.adapters import GenerationResult
from mini_rag.db import InMemoryDatabase
from mini_rag.expense.ask import ask_question, build_ask_response, main
from mini_rag.expense.generation import REFUSAL_ANSWER
from mini_rag.expense.models import ChunkRecord, RetrievedChunk, make_chunk_id

FOOD = "How much can I spend on food each day?"
AIRFARE = "Can I book first-class airfare?"
HOTEL = "My hotel costs $250. What do I need?"
RECEIPT = "Do I need a receipt for a $20 taxi?"
LIMO = "Can I claim a limousine upgrade?"
GYM = "Does the company reimburse gym memberships?"

SECTION_VECTORS: dict[str, list[float]] = {
    "1": [1.0, 0.0],
    "2": [0.0, 1.0],
    "3": [-1.0, 0.0],
    "4": [0.0, -1.0],
    "5": [1.0, 1.0],
    "6": [-1.0, -1.0],
}

QUESTION_VECTORS: dict[str, list[float]] = {
    FOOD: SECTION_VECTORS["1"],
    HOTEL: SECTION_VECTORS["2"],
    AIRFARE: SECTION_VECTORS["3"],
    LIMO: SECTION_VECTORS["4"],
    RECEIPT: SECTION_VECTORS["5"],
    GYM: [0.2, -1.0],
}

TITLES = {
    "1": "Meals",
    "2": "Hotels",
    "3": "Airfare",
    "4": "Ground Transportation",
    "5": "Receipts",
    "6": "Submission Deadline",
}

ANSWERS = {
    FOOD: GenerationResult(
        answer="Employees may claim up to $65 per day for meals.",
        sufficient=True,
        section="1",
    ),
    AIRFARE: GenerationResult(
        answer="Economy is required; business class requires approval.",
        sufficient=True,
        section="3",
    ),
    HOTEL: GenerationResult(
        answer="A manager must approve higher rates before booking.",
        sufficient=True,
        section="2",
    ),
    RECEIPT: GenerationResult(
        answer="No receipt is required under this policy.",
        sufficient=True,
        section="5",
    ),
    LIMO: GenerationResult(
        answer="Luxury vehicle upgrades are not reimbursable.",
        sufficient=True,
        section="4",
    ),
    GYM: GenerationResult(
        answer="Gyms are not mentioned, so I will guess no.",
        sufficient=False,
        section=None,
    ),
}

EXPECTED_CITATION = {
    FOOD: "1. Meals",
    AIRFARE: "3. Airfare",
    HOTEL: "2. Hotels",
    RECEIPT: "5. Receipts",
    LIMO: "4. Ground Transportation",
}


def _record(section: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=make_chunk_id("2.0", section),
        document="Employee Expense Policy",
        version="2.0",
        section=section,
        section_title=TITLES[section],
        text=f"{TITLES[section]} policy text.",
        embedding=list(SECTION_VECTORS[section]),
    )


def _loaded_db() -> InMemoryDatabase:
    database = InMemoryDatabase()
    database.upsert([_record(str(number)) for number in range(1, 7)])
    return database


def _chunk(
    section: str = "1",
    title: str = "Meals",
    distance: float = 0.1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=make_chunk_id("2.0", section),
        document="Employee Expense Policy",
        version="2.0",
        section=section,
        section_title=title,
        text=f"{title} policy text.",
        distance=distance,
    )


def test_build_ask_response_cites_retrieved_section() -> None:
    response = build_ask_response(ANSWERS[FOOD], [_chunk(), _chunk("2", "Hotels", 0.4)])

    assert response["answer"] == ANSWERS[FOOD].answer
    assert response["citation"] == {
        "document": "Employee Expense Policy",
        "version": "2.0",
        "section": "1. Meals",
    }
    assert response["retrieved_chunks"] == [
        {"section": "1. Meals", "distance": 0.1},
        {"section": "2. Hotels", "distance": 0.4},
    ]
    assert all(
        isinstance(item["distance"], float) for item in response["retrieved_chunks"]
    )
    assert "text" not in response["retrieved_chunks"][0]


def test_build_ask_response_refuses_when_insufficient() -> None:
    response = build_ask_response(ANSWERS[GYM], [_chunk("4", "Ground Transportation")])

    assert response["answer"] == REFUSAL_ANSWER
    assert response["citation"] is None


def test_build_ask_response_refuses_when_section_not_retrieved() -> None:
    generation = GenerationResult(
        answer="Employees may claim up to $65 per day for meals.",
        sufficient=True,
        section="1",
    )

    response = build_ask_response(generation, [_chunk("4", "Ground Transportation")])

    assert response["answer"] == REFUSAL_ANSWER
    assert response["citation"] is None


@pytest.mark.parametrize(
    "question",
    [FOOD, AIRFARE, HOTEL, RECEIPT, LIMO, GYM],
)
def test_ask_question_wires_six_assignment_questions(question: str) -> None:
    response = ask_question(
        question,
        ScriptedEmbedder(QUESTION_VECTORS),
        _loaded_db(),
        FakeGenerator(results=ANSWERS),
    )

    assert len(response["retrieved_chunks"]) == 3
    distances = [item["distance"] for item in response["retrieved_chunks"]]
    assert distances == sorted(distances)
    assert all(isinstance(distance, float) for distance in distances)
    encoded = json.dumps(response)
    assert "policy text" not in encoded

    if question == GYM:
        assert response["answer"] == REFUSAL_ANSWER
        assert response["citation"] is None
        return

    assert response["answer"] == ANSWERS[question].answer
    assert response["citation"] is not None
    assert response["citation"]["section"] == EXPECTED_CITATION[question]
    assert response["retrieved_chunks"][0]["section"] == EXPECTED_CITATION[question]


def test_ask_cli_requires_question() -> None:
    with pytest.raises(SystemExit):
        main([])


@pytest.mark.integration
def test_pgvector_retrieves_meals_for_food_question() -> None:
    from mini_rag.db import PgVectorDatabase
    from mini_rag.embedder import SentenceTransformerEmbedder
    from mini_rag.expense.config import database_url

    embedder = SentenceTransformerEmbedder()
    query = embedder.embed_texts([FOOD])[0]
    with PgVectorDatabase(database_url()) as database:
        chunks = database.search(query, k=3)

    assert chunks[0].section == "1"
    assert chunks[0].section_title == "Meals"
    assert len(chunks) == 3
