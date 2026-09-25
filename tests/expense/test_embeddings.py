from __future__ import annotations

import math
from pathlib import Path

import pytest

from fakes import FakeEmbedder
from mini_rag.embedder import (
    EMBEDDING_DIM,
    SentenceTransformerEmbedder,
    validate_embeddings,
)
from mini_rag.expense.chunking import parse_policy_file

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "policy.md"


def test_fake_embedder_returns_one_vector_per_text() -> None:
    policy = parse_policy_file(POLICY_PATH)
    embedder = FakeEmbedder()
    texts = [section.text for section in policy.sections]

    vectors = embedder.embed_texts(texts)

    assert embedder.dimension == EMBEDDING_DIM
    assert len(vectors) == 6
    assert all(len(vector) == EMBEDDING_DIM for vector in vectors)
    assert vectors[0] != vectors[1]


def test_validate_embeddings_rejects_zero_nan_and_wrong_dim() -> None:
    valid = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    validate_embeddings([valid], expected_dim=EMBEDDING_DIM)

    with pytest.raises(ValueError, match="zero vector"):
        validate_embeddings([[0.0] * EMBEDDING_DIM], expected_dim=EMBEDDING_DIM)

    nan_vector = list(valid)
    nan_vector[0] = math.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        validate_embeddings([nan_vector], expected_dim=EMBEDDING_DIM)

    with pytest.raises(ValueError, match="has length 2"):
        validate_embeddings([[1.0, 0.0]], expected_dim=EMBEDDING_DIM)


def test_sentence_transformer_embedder_normalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def get_sentence_embedding_dimension(self) -> int:
            return EMBEDDING_DIM

        def encode(
            self,
            texts: list[str],
            normalize_embeddings: bool,
            convert_to_numpy: bool,
        ) -> list[list[float]]:
            assert normalize_embeddings is True
            assert convert_to_numpy is True
            return [[1.0] + [0.0] * (EMBEDDING_DIM - 1) for _ in texts]

    monkeypatch.setattr(
        "mini_rag.embedder.sentence_transformer._load_sentence_transformer",
        lambda _name: FakeModel(),
    )

    embedder = SentenceTransformerEmbedder()
    vectors = embedder.embed_texts(["meals", "hotels"])

    assert embedder.dimension == EMBEDDING_DIM
    assert len(vectors) == 2
    assert vectors[0][0] == 1.0


@pytest.mark.integration
def test_minilm_embeds_short_texts() -> None:
    embedder = SentenceTransformerEmbedder()
    vectors = embedder.embed_texts(
        ["Employees may claim meals.", "Hotels are reimbursable."]
    )

    assert len(vectors) == 2
    assert len(vectors[0]) == EMBEDDING_DIM
    assert all(math.isfinite(value) for value in vectors[0])
    assert any(value != 0.0 for value in vectors[0])
