from __future__ import annotations

import logging
from collections.abc import Sequence

import pytest

from mini_rag.policy.rerank import rerank
from mini_rag.policy.retrieve import QUERY_PREFIX, FusedHit
from mini_rag.reranker.cross_encoder import CrossEncoderReranker


class RecordingReranker:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.question = ""
        self.passages: list[str] = []
        self.calls = 0

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        self.calls += 1
        self.question = question
        self.passages = list(passages)
        return self._scores


def _hit(section: str, text: str) -> FusedHit:
    return FusedHit(
        chunk_id=f"time-usage:v2.0:section-{section}",
        document_slug="time-usage",
        document="Time & Usage Policy",
        version="2.0",
        section=section,
        section_title="Token Allocation",
        parent_section=None,
        parent_title=None,
        text=text,
        rrf_score=1.0,
    )


def test_rerank_keeps_the_highest_three() -> None:
    fused = [
        _hit("1", "first"),
        _hit("2", "second"),
        _hit("3", "third"),
        _hit("4", "fourth"),
    ]
    reranker = RecordingReranker([0.1, 0.9, 0.4, 0.2])

    ranked = rerank("how many tokens?", fused, reranker)

    assert [hit.section for hit in ranked] == ["2", "3", "4"]
    assert ranked[0].rerank_score == 0.9


def test_rerank_sends_the_raw_question_and_the_heading() -> None:
    fused = [_hit("6", "500000 tokens.")]
    question = "how many tokens?"
    reranker = RecordingReranker([1.0])

    rerank(question, fused, reranker)

    assert reranker.question == question
    assert not reranker.question.startswith(QUERY_PREFIX)
    passage = reranker.passages[0]
    assert "6. Token Allocation" in passage
    assert "500000 tokens." in passage


def test_rerank_keeps_fused_order_when_scores_tie() -> None:
    fused = [_hit("1", "first"), _hit("2", "second"), _hit("3", "third")]
    reranker = RecordingReranker([1.0, 1.0, 0.2])

    ranked = rerank("how many tokens?", fused, reranker)

    assert [hit.section for hit in ranked] == ["1", "2", "3"]


def test_rerank_skips_the_model_when_fused_is_empty() -> None:
    reranker = RecordingReranker([])

    assert rerank("how many tokens?", [], reranker) == []
    assert reranker.calls == 0


def test_rerank_logs_the_kept_scores(caplog: pytest.LogCaptureFixture) -> None:
    fused = [_hit("6", "500000 tokens allocation.")]

    with caplog.at_level(logging.INFO):
        rerank("how many tokens?", fused, RecordingReranker([1.25]))

    assert "Rerank time-usage:v2.0:section-6 1.2500" in caplog.text
    assert "allocation" not in caplog.text


def test_cross_encoder_reranker_scores_pairs_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCrossEncoder:
        def __init__(self) -> None:
            self.pairs: list[tuple[str, str]] = []

        def predict(self, sentences: list[tuple[str, str]]) -> list[float]:
            self.pairs = sentences
            return [float(index) for index, _pair in enumerate(sentences)]

    fake = FakeCrossEncoder()
    monkeypatch.setattr(
        "mini_rag.reranker.cross_encoder._load_cross_encoder",
        lambda _name: fake,
    )

    scores = CrossEncoderReranker().score("tokens", ["one", "two"])

    assert fake.pairs == [("tokens", "one"), ("tokens", "two")]
    assert scores == [0.0, 1.0]
