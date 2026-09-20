from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from mini_rag.config import ollama_host, ollama_model
from mini_rag.models import RetrievedChunk

REFUSAL_ANSWER = "The provided policy does not answer this question."
DEFAULT_TIMEOUT_SECONDS = 120.0
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT = """Answer the question using only the policy excerpts below.
Answer if the excerpts cover the topic even when the wording differs from the policy.
If the excerpts do not contain the answer, set sufficient to false.

Return JSON only with this shape:
{"answer": "...", "sufficient": true, "section": "1"}

Rules:
- section is the stored section number (for example "1"), never a title.
- When sufficient is false, section must be null and answer must be exactly:
"The provided policy does not answer this question."
"""


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    sufficient: bool
    section: str | None


class Generator(Protocol):
    def generate(
        self, question: str, chunks: Sequence[RetrievedChunk]
    ) -> GenerationResult: ...


def build_generation_prompt(
    question: str, chunks: Sequence[RetrievedChunk]
) -> str:
    excerpts = "\n\n".join(
        f"Section {chunk.section}. {chunk.section_title}: {chunk.text}"
        for chunk in chunks
    )
    return f"{excerpts}\n\nQuestion: {question}"


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def parse_generation_result(content: str) -> GenerationResult:
    stripped = strip_thinking(content)
    if not stripped:
        raise ValueError("Generator returned empty content")
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("Generator JSON must be an object")
    if "answer" not in payload or "sufficient" not in payload:
        raise ValueError("Generator JSON must include answer and sufficient")

    sufficient = bool(payload["sufficient"])
    raw_section = payload.get("section")
    section: str | None
    if raw_section is None or raw_section == "":
        section = None
    else:
        section = str(raw_section)
    if not sufficient:
        section = None
    return GenerationResult(
        answer=str(payload["answer"]).strip(),
        sufficient=sufficient,
        section=section,
    )


class OllamaGenerator:
    """Local Qwen generator. Thinking is requested, then discarded."""

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        post: Callable[[str, dict[str, Any], float], dict[str, Any]] | None = None,
    ) -> None:
        self._host = (host or ollama_host()).rstrip("/")
        self._model = model or ollama_model()
        self._timeout = timeout
        self._post = post or _post_json

    def generate(
        self, question: str, chunks: Sequence[RetrievedChunk]
    ) -> GenerationResult:
        payload = {
            "model": self._model,
            "stream": False,
            "think": True,
            "format": "json",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_generation_prompt(question, chunks),
                },
            ],
        }
        response = self._post(
            f"{self._host}/api/chat",
            payload,
            self._timeout,
        )
        message = response.get("message") or {}
        content = message.get("content") or ""
        return parse_generation_result(content)


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    if not isinstance(body, dict):
        raise ValueError("Ollama response must be a JSON object")
    return body
