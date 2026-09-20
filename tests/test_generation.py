from __future__ import annotations

from typing import Any

import pytest

from fakes import FakeGenerator
from mini_rag.generation import (
    REFUSAL_ANSWER,
    SYSTEM_PROMPT,
    GenerationResult,
    OllamaGenerator,
    build_generation_prompt,
    parse_generation_result,
    strip_thinking,
)
from mini_rag.models import RetrievedChunk


def _chunk(
    section: str = "1",
    title: str = "Meals",
    text: str = "Employees may claim up to $65 per day for meals.",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"expense-policy:v2.0:section-{section}",
        document="Employee Expense Policy",
        version="2.0",
        section=section,
        section_title=title,
        text=text,
        distance=0.1,
    )


def test_system_prompt_requires_refusal_and_section_number() -> None:
    assert REFUSAL_ANSWER in SYSTEM_PROMPT
    assert "wording differs" in SYSTEM_PROMPT
    assert '"section": "1"' in SYSTEM_PROMPT


def test_build_generation_prompt_labels_excerpts() -> None:
    prompt = build_generation_prompt(
        "How much can I spend on food each day?",
        [_chunk(), _chunk("2", "Hotels", "Hotels are reimbursable.")],
    )

    assert "Section 1. Meals:" in prompt
    assert "Section 2. Hotels:" in prompt
    assert "How much can I spend on food each day?" in prompt


def test_strip_thinking_removes_leaked_tags() -> None:
    body = '{"answer": "ok", "sufficient": true, "section": "1"}'
    raw = f"<think>reason</think>\n{body}"

    assert strip_thinking(raw) == body


def test_parse_generation_result_coerces_section_to_str() -> None:
    result = parse_generation_result(
        '{"answer": "Economy is required.", "sufficient": true, "section": 3}'
    )

    assert result == GenerationResult(
        answer="Economy is required.",
        sufficient=True,
        section="3",
    )


def test_parse_generation_result_clears_section_when_insufficient() -> None:
    result = parse_generation_result(
        f'{{"answer": "{REFUSAL_ANSWER}", "sufficient": false, "section": "4"}}'
    )

    assert result.sufficient is False
    assert result.section is None
    assert result.answer == REFUSAL_ANSWER


def test_parse_generation_result_rejects_empty_after_strip() -> None:
    with pytest.raises(ValueError, match="empty content"):
        parse_generation_result("<think>only thinking</think>")


def test_fake_generator_returns_configured_result() -> None:
    expected = GenerationResult(
        answer=REFUSAL_ANSWER,
        sufficient=False,
        section=None,
    )
    generator = FakeGenerator(expected)
    chunks = [_chunk()]

    result = generator.generate("Does the company reimburse gym memberships?", chunks)

    assert result == expected
    assert generator.calls[0][0] == "Does the company reimburse gym memberships?"
    assert generator.calls[0][1] == chunks


def test_ollama_generator_reads_content_and_ignores_thinking() -> None:
    captured: dict[str, Any] = {}

    def fake_post(
        url: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        captured["url"] = url
        captured["payload"] = payload
        captured["timeout"] = timeout
        return {
            "message": {
                "thinking": "I should look at meals.",
                "content": (
                    '{"answer": "Employees may claim up to $65 per day for meals.",'
                    ' "sufficient": true, "section": "1"}'
                ),
            }
        }

    generator = OllamaGenerator(
        host="http://localhost:11434",
        model="qwen3:8b",
        post=fake_post,
    )

    result = generator.generate(
        "How much can I spend on food each day?",
        [_chunk()],
    )

    assert result.sufficient is True
    assert result.section == "1"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["think"] is True
    assert captured["payload"]["format"] == "json"
    assert captured["payload"]["model"] == "qwen3:8b"


def test_ollama_generator_errors_when_only_thinking_is_present() -> None:
    def fake_post(
        url: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        return {"message": {"thinking": "still thinking", "content": ""}}

    generator = OllamaGenerator(post=fake_post)

    with pytest.raises(ValueError, match="empty content"):
        generator.generate("question", [_chunk()])


@pytest.mark.integration
def test_ollama_qwen_returns_parseable_json() -> None:
    generator = OllamaGenerator()
    result = generator.generate(
        "How much can I spend on food each day?",
        [_chunk()],
    )

    assert result.answer
    assert isinstance(result.sufficient, bool)
