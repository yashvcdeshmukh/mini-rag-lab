from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


class _CrossEncoder(Protocol):
    def predict(self, sentences: list[tuple[str, str]]) -> Sequence[float]: ...


def _load_cross_encoder(model_name: str) -> _CrossEncoder:
    from sentence_transformers import CrossEncoder

    return cast(_CrossEncoder, CrossEncoder(model_name))


class CrossEncoderReranker:
    """Local cross-encoder. Scores are returned in passage order."""

    def __init__(self, model_name: str = RERANKER_MODEL_NAME) -> None:
        self._model = _load_cross_encoder(model_name)

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(question, passage) for passage in passages]
        predicted = self._model.predict(pairs)
        return [float(score) for score in predicted]
