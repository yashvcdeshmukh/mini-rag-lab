"""Score a fixed set of policy questions for recall, precision, MRR, and accuracy."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from mini_rag.adapters import Citation
from mini_rag.policy.ask import AskResult
from mini_rag.policy.generation import REFUSAL_ANSWER
from mini_rag.policy.models import make_chunk_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalCase:
    """One question and the checks the corpus determines."""

    question: str
    kind: str
    figures: tuple[str, ...] = ()
    chunk_ids: tuple[str, ...] = ()
    fact_key: str | None = None
    edition: str | None = None
    citations: tuple[Citation, ...] = ()


@dataclass(frozen=True)
class CaseScore:
    """Metrics and failed checks for one case."""

    question: str
    recall: float | None
    precision: float | None
    mrr: float | None
    accuracy: float
    passed: bool
    failed: tuple[str, ...]


CASES: tuple[EvalCase, ...] = (
    EvalCase(
        question="How many tokens does an employee receive?",
        kind="structured",
        figures=("1000000", "500000"),
        fact_key="token_allocation",
        edition="unspecified",
        citations=(
            Citation("Time & Usage Policy", "1.0", "5"),
            Citation("Time & Usage Policy", "2.0", "6"),
        ),
    ),
    EvalCase(
        question="What should I do with cake left in the weekend fridge?",
        kind="unstructured",
        figures=("spoonful",),
        chunk_ids=(make_chunk_id("2.0", "7", document_slug="hr"),),
    ),
    EvalCase(
        question="What has to be in a work email?",
        kind="unstructured",
        figures=("joke",),
        chunk_ids=(
            make_chunk_id("1.0", "3", document_slug="hr"),
            make_chunk_id("2.0", "3", document_slug="hr"),
        ),
    ),
    EvalCase(
        question="What am I allowed to wear at the office?",
        kind="unstructured",
        figures=("pajamas",),
        chunk_ids=(make_chunk_id("2.0", "4", document_slug="hr"),),
    ),
    EvalCase(
        question="Where should employees take cover during a nuclear detonation?",
        kind="unstructured",
        figures=("conference table", "refrigerator"),
        chunk_ids=(
            make_chunk_id("1.0", "4", document_slug="preparedness"),
            make_chunk_id("2.0", "4", document_slug="preparedness"),
        ),
    ),
    EvalCase(
        question="Where should employees go during a zombie apocalypse?",
        kind="unstructured",
        figures=("third-floor",),
        chunk_ids=(make_chunk_id("2.0", "3", document_slug="preparedness"),),
    ),
    EvalCase(
        question="What should employees do first if an AI apocalypse starts?",
        kind="unstructured",
        figures=("whiteboard",),
        chunk_ids=(make_chunk_id("2.0", "7", document_slug="preparedness"),),
    ),
    EvalCase(
        question="What is the dress code for visiting the moon?",
        kind="refusal",
    ),
)


def score_case(case: EvalCase, result: AskResult) -> CaseScore:
    """Score one ask result. Precision and MRR never fail a case."""
    failed: list[str] = []
    if case.kind == "structured" and result.trace.route != "structured":
        failed.append("route")
    elif case.kind == "unstructured" and result.trace.route != "unstructured":
        failed.append("route")
    if result.trace.fact_key != case.fact_key:
        failed.append("fact_key")
    if case.edition is not None and result.trace.edition != case.edition:
        failed.append("edition")
    if _citations_missed(case, result):
        failed.append("citations")
    if result.trace.cache_hit:
        failed.append("cache_hit")

    metrics = _retrieval_metrics(case.chunk_ids, result.trace.final_chunk_ids)
    recall = precision = mrr = None
    if metrics is not None:
        recall, precision, mrr = metrics
        if recall != 1.0:
            failed.append("recall")

    accuracy = _accuracy(case, result)
    if accuracy != 1.0:
        failed.append("accuracy")
    return CaseScore(
        question=case.question,
        recall=recall,
        precision=precision,
        mrr=mrr,
        accuracy=accuracy,
        passed=not failed,
        failed=tuple(failed),
    )


def summarize(scores: Sequence[CaseScore]) -> dict[str, object]:
    """One JSON-ready report: per-case metrics, averages, and pass/fail counts."""
    return {
        "cases": [
            {
                "question": score.question,
                "recall": score.recall,
                "precision": score.precision,
                "mrr": score.mrr,
                "accuracy": score.accuracy,
                "passed": score.passed,
                "failed": list(score.failed),
            }
            for score in scores
        ],
        "averages": {
            "recall": _mean([score.recall for score in scores]),
            "precision": _mean([score.precision for score in scores]),
            "mrr": _mean([score.mrr for score in scores]),
            "accuracy": _mean([score.accuracy for score in scores]),
        },
        "passed": sum(1 for score in scores if score.passed),
        "failed": sum(1 for score in scores if not score.passed),
    }


def _retrieval_metrics(
    expected: tuple[str, ...], retrieved: tuple[str, ...]
) -> tuple[float, float, float] | None:
    if not expected:
        return None
    expected_ids = set(expected)
    retrieved_ids = set(retrieved)
    overlap = len(expected_ids & retrieved_ids)
    recall = overlap / len(expected_ids)
    precision = overlap / len(retrieved_ids) if retrieved_ids else 0.0
    rank = next(
        (
            index
            for index, chunk_id in enumerate(retrieved, start=1)
            if chunk_id in expected_ids
        ),
        None,
    )
    mrr = 0.0 if rank is None else 1.0 / rank
    return recall, precision, mrr


def _accuracy(case: EvalCase, result: AskResult) -> float:
    if case.kind == "refusal":
        exact = result.answer == REFUSAL_ANSWER and not result.citations
        return 1.0 if exact else 0.0
    if all(figure in result.answer for figure in case.figures):
        return 1.0
    return 0.0


def _citations_missed(case: EvalCase, result: AskResult) -> bool:
    if case.kind == "refusal":
        return bool(result.citations)
    returned = {
        (citation.document, citation.version, citation.section)
        for citation in result.citations
    }
    return any(
        (citation.document, citation.version, citation.section) not in returned
        for citation in case.citations
    )


def _mean(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _live_scores() -> tuple[CaseScore, ...]:
    from mini_rag.db import PgPolicyStore
    from mini_rag.decider import JevDecider
    from mini_rag.embedder import SentenceTransformerEmbedder
    from mini_rag.generator import OllamaGenerator
    from mini_rag.policy.ask import ask_question
    from mini_rag.policy.config import policy_database_url
    from mini_rag.policy.facts import CHOICE_CRITERIA
    from mini_rag.policy.generation import SYSTEM_PROMPT, build_generation_prompt
    from mini_rag.policy.models import BGE_MODEL_NAME
    from mini_rag.reranker import CrossEncoderReranker

    scores: list[CaseScore] = []
    with PgPolicyStore(policy_database_url()) as store:
        embedder = SentenceTransformerEmbedder(BGE_MODEL_NAME)
        reranker = CrossEncoderReranker()
        generator = OllamaGenerator(
            system_prompt=SYSTEM_PROMPT,
            build_user_prompt=build_generation_prompt,
            temperature=0,
        )
        decider = JevDecider(criteria=CHOICE_CRITERIA)
        for case in CASES:
            try:
                result = ask_question(
                    case.question,
                    embedder,
                    store,
                    reranker,
                    generator,
                    decider,
                    use_cache=False,
                    caches=None,
                )
            except Exception as exc:
                logger.error("Case failed: %s: %s", case.question, exc)
                scores.append(
                    CaseScore(
                        question=case.question,
                        recall=None,
                        precision=None,
                        mrr=None,
                        accuracy=0.0,
                        passed=False,
                        failed=("error",),
                    )
                )
                continue
            scores.append(score_case(case, result))
    return tuple(scores)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argparse.ArgumentParser(
        description="Score the fixed policy questions and print the metrics."
    ).parse_args(argv)
    try:
        payload = summarize(_live_scores())
    except Exception as exc:
        logger.error("Eval failed: %s", exc)
        return 1
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    failed = payload["failed"]
    return 1 if isinstance(failed, int) and failed else 0


if __name__ == "__main__":
    sys.exit(main())
