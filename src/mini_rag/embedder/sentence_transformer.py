from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from mini_rag.embedder.validate import validate_embeddings

# all-MiniLM-L6-v2 output size. The pgvector column must use the same width.
EMBEDDING_DIM = 384
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class _Encoder(Protocol):
    def get_sentence_embedding_dimension(self) -> int: ...

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> Sequence[Sequence[float]]: ...


def _load_sentence_transformer(model_name: str) -> _Encoder:
    from sentence_transformers import SentenceTransformer

    return cast(_Encoder, SentenceTransformer(model_name))


class SentenceTransformerEmbedder:
    """Local sentence-transformer embedder. Vectors are L2-normalized for cosine."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model = _load_sentence_transformer(model_name)
        dimension = int(self._model.get_sentence_embedding_dimension())
        if model_name == DEFAULT_MODEL_NAME and dimension != EMBEDDING_DIM:
            raise ValueError(
                f"{DEFAULT_MODEL_NAME} produced dimension {dimension}, "
                f"expected {EMBEDDING_DIM}"
            )
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        encoded = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        vectors = [list(map(float, vector)) for vector in encoded]
        validate_embeddings(vectors, expected_dim=self._dimension)
        return vectors
