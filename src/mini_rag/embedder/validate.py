from __future__ import annotations

import math
from collections.abc import Sequence


def validate_embeddings(
    vectors: Sequence[Sequence[float]],
    *,
    expected_dim: int,
) -> None:
    """Reject vectors that cannot be ranked with cosine distance."""
    for index, vector in enumerate(vectors):
        if len(vector) != expected_dim:
            raise ValueError(
                f"Embedding {index} has length {len(vector)}, expected {expected_dim}"
            )
        if any(not math.isfinite(value) for value in vector):
            raise ValueError(f"Embedding {index} contains NaN or infinity")
        if all(value == 0.0 for value in vector):
            raise ValueError(f"Embedding {index} is a zero vector")
