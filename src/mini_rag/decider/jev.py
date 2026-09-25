from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from mini_rag.adapters import Decision
from mini_rag.config import jev_api_key, jev_base_url, jev_model

JevPost = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]
NOUL_YES = 0.5


class JevDecider:
    """Asks the Jev Decision API for one label and its probability."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        post: JevPost | None = None,
        criteria: Mapping[str, str] | None = None,
    ) -> None:
        self._api_key = jev_api_key() if api_key is None else api_key
        if not self._api_key.strip():
            raise ValueError("TYPESAFE_API_KEY is not set")
        self._model = model or jev_model()
        self._url = f"{(base_url or jev_base_url()).rstrip('/')}/v1/systemone"
        self._timeout = timeout
        self._post = post
        self._criteria = dict(criteria or {})
        self._resolved_model = ""

    @property
    def model(self) -> str:
        return self._resolved_model or self._model

    def choose(
        self,
        name: str,
        state: str,
        instructions: str,
        choices: Sequence[str],
    ) -> Decision:
        if not name:
            raise ValueError("Decider question name is required")
        if not choices:
            raise ValueError("Decider choices are required")
        payload = {
            "model": self._model,
            "state": state,
            "questions": {name: self._question(instructions, choices)},
        }
        response = self._post_json(payload)
        resolved = response.get("model")
        if isinstance(resolved, str) and resolved:
            self._resolved_model = resolved
        return _decision(name, response, choices)

    def _question(self, instructions: str, choices: Sequence[str]) -> dict[str, Any]:
        if set(choices) == {"yes", "no"}:
            return {
                "type": "noul",
                "instructions": instructions,
                "criteria": {
                    "true": self._criterion("yes"),
                    "false": self._criterion("no"),
                },
            }
        return {
            "type": "choice",
            "instructions": instructions,
            "criteria": {label: self._criterion(label) for label in choices},
        }

    def _criterion(self, label: str) -> str:
        return self._criteria.get(label, label.replace("_", " "))

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        if self._post is not None:
            return self._post(self._url, payload, headers, self._timeout)
        return _post_jev(self._url, payload, headers, self._timeout)


def _decision(
    name: str, response: dict[str, Any], choices: Sequence[str]
) -> Decision:
    answers = response.get("answers")
    if not isinstance(answers, dict) or name not in answers:
        raise ValueError(f"Jev response missing answer {name!r}")
    answer = answers[name]
    if not isinstance(answer, dict):
        raise ValueError("Jev answer must be an object")
    kind = answer.get("type")
    if kind == "choice":
        choice = str(answer.get("choice", "")).strip()
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or choice not in probabilities:
            raise ValueError(f"Jev choice {choice!r} has no probability")
        probability = float(probabilities[choice])
    elif kind == "noul":
        if "noul" not in answer:
            raise ValueError("Jev noul answer has no probability")
        noul = float(answer["noul"])
        if noul >= NOUL_YES:
            choice, probability = "yes", noul
        else:
            choice, probability = "no", 1.0 - noul
    else:
        raise ValueError(f"Jev answer type {kind!r} is not choice or noul")
    if choice not in choices:
        raise ValueError(f"Choice {choice!r} is not one of {list(choices)}")
    return Decision(choice=choice, probability=probability)


def _read_jev(request: urllib.request.Request, timeout: float) -> str:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
        if isinstance(raw, str):
            return raw
        if isinstance(raw, (bytes, bytearray)):
            return raw.decode()
        raise ValueError("Jev response body must be text")
    except TimeoutError as exc:
        raise TimeoutError("Jev request timed out") from exc


def _post_jev(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        body = _read_jev(request, timeout)
    except TimeoutError:
        body = _read_jev(request, timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"Jev request failed: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Jev request failed: {exc}") from exc
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("Jev response must be a JSON object")
    return parsed
