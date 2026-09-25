from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar

from mini_rag.adapters import Citation, GenerationResult
from mini_rag.config import ollama_host, ollama_model

ChunkT = TypeVar("ChunkT")
DEFAULT_TIMEOUT_SECONDS = 240.0
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


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
        citations=() if not sufficient else _parse_citations(payload),
    )


def _parse_citations(payload: dict[str, Any]) -> tuple[Citation, ...]:
    raw = payload.get("citations")
    if not isinstance(raw, list):
        return ()
    citations: list[Citation] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        document = item.get("document")
        version = item.get("version")
        section = item.get("section")
        fields = (document, version, section)
        if not all(isinstance(value, str) and value.strip() for value in fields):
            continue
        citations.append(
            Citation(
                document=str(document).strip(),
                version=str(version).strip(),
                section=str(section).strip(),
            )
        )
    return tuple(citations)


class OllamaGenerator(Generic[ChunkT]):
    """Local Qwen generator. The caller supplies the prompt. Thinking is discarded."""

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        *,
        system_prompt: str,
        build_user_prompt: Callable[[str, Sequence[ChunkT]], str],
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        post: Callable[[str, dict[str, Any], float], dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> None:
        self._host = (host or ollama_host()).rstrip("/")
        self._model = model or ollama_model()
        self._system_prompt = system_prompt
        self._build_user_prompt = build_user_prompt
        self._timeout = timeout
        self._post = post or _post_json
        self._temperature = temperature

    @property
    def model(self) -> str:
        return self._model

    def generate(self, question: str, chunks: Sequence[ChunkT]) -> GenerationResult:
        payload: dict[str, Any] = {
            "model": self._model,
            "stream": False,
            "think": True,
            "format": "json",
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": self._build_user_prompt(question, chunks),
                },
            ],
        }
        if self._temperature is not None:
            payload["options"] = {"temperature": self._temperature}
        response = self._post(
            f"{self._host}/api/chat",
            payload,
            self._timeout,
        )
        message = response.get("message") or {}
        content = message.get("content") or ""
        return parse_generation_result(content)


class OllamaRewriter:
    """Rewrites a question into one search query. Returns plain text."""

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

    def __call__(self, question: str) -> str:
        payload = {
            "model": self._model,
            "stream": False,
            "think": True,
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the question as one search query for the specific "
                        "rules, numbers, and places needed to answer it. "
                        "Use any section titles included with the question. "
                        "Return only the rewritten question."
                    ),
                },
                {"role": "user", "content": question},
            ],
        }
        response = self._post(f"{self._host}/api/chat", payload, self._timeout)
        message = response.get("message") or {}
        text = strip_thinking(str(message.get("content") or ""))
        if not text:
            raise ValueError("Rewriter returned empty content")
        return text


def _read_ollama(request: urllib.request.Request, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode())
    except TimeoutError as exc:
        raise TimeoutError("Ollama request timed out") from exc
    if not isinstance(body, dict):
        raise ValueError("Ollama response must be a JSON object")
    return body


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        return _read_ollama(request, timeout)
    except TimeoutError:
        return _read_ollama(request, timeout)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
