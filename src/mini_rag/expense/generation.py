from __future__ import annotations

from collections.abc import Sequence

from mini_rag.expense.models import RetrievedChunk

REFUSAL_ANSWER = "The provided policy does not answer this question."

SYSTEM_PROMPT = """Answer the question using only the policy excerpts below.
Answer if the excerpts cover the same topic even when the wording differs.
Treat close paraphrases as the same topic. Examples: food means meals;
first-class or business-class means airfare class; limousine means a
luxury vehicle upgrade; a $20 taxi receipt is a receipts question.
If the question asks about a fare class not named in the excerpts, use
the airfare class rules that are present.
If no excerpt covers the topic, set sufficient to false.

Return JSON only with this shape:
{"answer": "...", "sufficient": true, "section": "1"}

Rules:
- section is the stored section number (for example "1"), never a title.
- When sufficient is false, section must be null and answer must be exactly:
"The provided policy does not answer this question."
"""


def build_generation_prompt(
    question: str, chunks: Sequence[RetrievedChunk]
) -> str:
    excerpts = "\n\n".join(
        f"Section {chunk.section}. {chunk.section_title}: {chunk.text}"
        for chunk in chunks
    )
    return f"{excerpts}\n\nQuestion: {question}"
