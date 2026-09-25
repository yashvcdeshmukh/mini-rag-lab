from __future__ import annotations

from collections.abc import Sequence

from mini_rag.policy.rerank import RerankedHit

REFUSAL_ANSWER = "The provided policy does not answer this question."

SYSTEM_PROMPT = """Answer the question using only the policy excerpts below.
Each excerpt names its document, version, and section.
When two excerpts state different values for the same rule, report both
values and cite each document, version, and section. Do not pick a winner.
If no excerpt covers the topic, set sufficient to false.

Return JSON only with this shape:
{"answer": "...", "sufficient": true, "section": "6", "citations": [
{"document": "Time & Usage Policy", "version": "2.0", "section": "6"}]}

Rules:
- citations lists every excerpt you used. document, version, and section must
  match the excerpt.
- section is the stored section number of one cited excerpt, or null.
- When sufficient is false, section must be null, citations must be [], and
  answer must be exactly:
"The provided policy does not answer this question."
"""


def build_generation_prompt(
    question: str, chunks: Sequence[RerankedHit]
) -> str:
    excerpts = "\n\n".join(
        "\n".join(
            (
                f"Document: {chunk.document}",
                f"Version: {chunk.version}",
                f"Section: {chunk.section}. {chunk.section_title}",
                chunk.text,
            )
        )
        for chunk in chunks
    )
    return f"{excerpts}\n\nQuestion: {question}"
