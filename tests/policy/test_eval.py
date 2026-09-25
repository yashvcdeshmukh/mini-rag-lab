from __future__ import annotations

import json
from pathlib import Path

from mini_rag.adapters import Citation
from mini_rag.policy.ask import AskResult, AskTrace, CorpusStampView
from mini_rag.policy.eval import CASES, _mean, score_case, summarize
from mini_rag.policy.generation import REFUSAL_ANSWER
from mini_rag.policy.models import make_chunk_id

_RECORDED_EVAL = Path(__file__).resolve().parents[2] / "examples" / "policy-eval.json"

_V1 = make_chunk_id("1.0", "4", document_slug="preparedness")
_V2 = make_chunk_id("2.0", "4", document_slug="preparedness")
_SHELTER = "under a conference table or in the refrigerator"


def _result(
    *,
    question: str,
    answer: str,
    route: str = "unstructured",
    fact_key: str | None = None,
    edition: str = "unspecified",
    chunk_ids: tuple[str, ...] = (),
    citations: tuple[Citation, ...] = (),
    cache_hit: bool = False,
) -> AskResult:
    trace = AskTrace(
        question=question,
        embedding_model="test",
        reranker_model="test",
        generator_model="test",
        jev_model="test",
        corpus=CorpusStampView(0, ""),
        route=route,
        fact_key=fact_key,
        edition=edition,
        decisions=(),
        dense=(),
        keyword=(),
        fused=(),
        rerank_scores=(),
        triage="skipped",
        sufficiency="skipped",
        rewrite=None,
        final_chunk_ids=chunk_ids,
        citations=citations,
        answer=answer,
        cache_hit=cache_hit,
        timings_ms=(),
    )
    return AskResult(answer=answer, citations=citations, trace=trace)


def test_there_are_eight_questions() -> None:
    assert len(CASES) == 8
    assert sum(case.kind == "structured" for case in CASES) == 1
    assert sum(case.kind == "unstructured" for case in CASES) == 6
    assert sum(case.kind == "refusal" for case in CASES) == 1


def test_missing_chunk_sets_recall_and_mrr_from_the_remaining_id() -> None:
    case = CASES[4]
    result = _result(
        question=case.question,
        answer=_SHELTER,
        chunk_ids=("other", _V2),
    )

    score = score_case(case, result)

    assert score.recall == 0.5
    assert score.mrr == 0.5
    assert score.passed is False
    assert "recall" in score.failed


def test_extra_chunk_lowers_precision_without_failing() -> None:
    case = CASES[4]
    result = _result(
        question=case.question,
        answer=_SHELTER,
        chunk_ids=(_V1, _V2, "other"),
    )

    score = score_case(case, result)

    assert score.precision == 2 / 3
    assert score.recall == 1.0
    assert score.passed is True
    assert score.failed == ()


def test_fact_case_leaves_retrieval_metrics_null() -> None:
    case = CASES[0]
    citations = (
        Citation("Time & Usage Policy", "1.0", "5"),
        Citation("Time & Usage Policy", "2.0", "6"),
    )
    result = _result(
        question=case.question,
        answer=(
            "1000000 (Time & Usage Policy v1.0 section 5); "
            "500000 (Time & Usage Policy v2.0 section 6)"
        ),
        route="structured",
        fact_key="token_allocation",
        citations=citations,
    )

    score = score_case(case, result)

    assert score.recall is None
    assert score.precision is None
    assert score.mrr is None
    assert score.accuracy == 1.0
    assert score.passed is True


def test_missing_figure_sets_accuracy_to_zero() -> None:
    case = CASES[0]
    result = _result(
        question=case.question,
        answer="1000000 (Time & Usage Policy v1.0 section 5)",
        route="structured",
        fact_key="token_allocation",
        citations=(
            Citation("Time & Usage Policy", "1.0", "5"),
            Citation("Time & Usage Policy", "2.0", "6"),
        ),
    )

    score = score_case(case, result)

    assert score.accuracy == 0.0
    assert score.passed is False
    assert "accuracy" in score.failed


def test_exact_refusal_sets_accuracy_to_one() -> None:
    case = CASES[7]
    result = _result(question=case.question, answer=REFUSAL_ANSWER)

    score = score_case(case, result)

    assert score.accuracy == 1.0
    assert score.recall is None
    assert score.passed is True


def test_refusal_with_a_citation_fails() -> None:
    case = CASES[7]
    citation = Citation("time-usage", "1.0", "5")
    result = _result(
        question=case.question,
        answer=REFUSAL_ANSWER,
        citations=(citation,),
    )

    score = score_case(case, result)

    assert score.accuracy == 0.0
    assert "citations" in score.failed
    assert score.passed is False


def test_wrong_edition_fails() -> None:
    case = CASES[0]
    result = _result(
        question=case.question,
        answer="1000000 and 500000",
        route="structured",
        fact_key="token_allocation",
        edition="current",
        citations=(
            Citation("Time & Usage Policy", "1.0", "5"),
            Citation("Time & Usage Policy", "2.0", "6"),
        ),
    )

    score = score_case(case, result)

    assert score.failed == ("edition",)


def test_cache_hit_fails() -> None:
    case = CASES[7]
    result = _result(
        question=case.question,
        answer=REFUSAL_ANSWER,
        cache_hit=True,
    )

    score = score_case(case, result)

    assert "cache_hit" in score.failed
    assert score.passed is False


def test_averages_skip_null_retrieval_metrics() -> None:
    fact = score_case(
        CASES[0],
        _result(
            question=CASES[0].question,
            answer="1000000 and 500000",
            route="structured",
            fact_key="token_allocation",
            citations=(
                Citation("Time & Usage Policy", "1.0", "5"),
                Citation("Time & Usage Policy", "2.0", "6"),
            ),
        ),
    )
    shelter = score_case(
        CASES[4],
        _result(
            question=CASES[4].question,
            answer=_SHELTER,
            chunk_ids=("other", _V2),
        ),
    )

    report = summarize((fact, shelter))
    averages = report["averages"]

    assert isinstance(averages, dict)
    assert averages["recall"] == 0.5
    assert averages["precision"] == 0.5
    assert averages["mrr"] == 0.5
    assert averages["accuracy"] == 1.0
    assert report["passed"] == 1
    assert report["failed"] == 1


def test_recorded_harness_run_matches_the_golden_set() -> None:
    """Check a saved live run. This does not call the models or the database."""
    report = json.loads(_RECORDED_EVAL.read_text())
    cases = report["cases"]
    assert isinstance(cases, list)
    assert [case["question"] for case in cases] == [case.question for case in CASES]

    retrieval = [case for case in cases if case["recall"] is not None]
    assert len(retrieval) == sum(1 for case in CASES if case.chunk_ids)
    assert all(case["recall"] == 1.0 for case in retrieval)

    averages = report["averages"]
    assert isinstance(averages, dict)
    assert averages["recall"] == _mean([case["recall"] for case in cases])
    assert averages["precision"] == _mean([case["precision"] for case in cases])
    assert averages["mrr"] == _mean([case["mrr"] for case in cases])
    assert averages["accuracy"] == _mean([case["accuracy"] for case in cases])
    assert report["passed"] == sum(1 for case in cases if case["passed"])
    assert report["failed"] == sum(1 for case in cases if not case["passed"])
