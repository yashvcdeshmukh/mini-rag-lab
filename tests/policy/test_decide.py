from typing import Any

import pytest

from mini_rag.decider import JevDecider
from mini_rag.policy.facts import CHECKED_FACTS, CHOICE_CRITERIA


def test_choice_request_uses_the_decision_api() -> None:
    captured: dict[str, Any] = {}

    def post(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        captured["timeout"] = timeout
        return {
            "model": "jev-1.13.0",
            "answers": {
                "fact": {
                    "type": "choice",
                    "choice": "none",
                    "probabilities": {"none": 0.91, "token_allocation": 0.09},
                }
            },
        }

    decider = JevDecider(
        api_key="jv_live_test",
        model="jev-latest",
        base_url="https://api.typesafe.ai",
        post=post,
        criteria=CHOICE_CRITERIA,
    )
    decision = decider.choose(
        "fact",
        "how many tokens",
        "Choose the checked fact this question looks up, or none.",
        ("token_allocation", "none"),
    )

    assert decision.choice == "none"
    assert decision.probability == 0.91
    assert decider.model == "jev-1.13.0"
    assert captured["url"] == "https://api.typesafe.ai/v1/systemone"
    assert captured["headers"]["Authorization"] == "Bearer jv_live_test"
    payload = captured["payload"]
    assert payload["model"] == "jev-latest"
    assert payload["state"] == "how many tokens"
    assert "temperature" not in payload
    assert "seed" not in payload
    question = payload["questions"]["fact"]
    assert question["type"] == "choice"
    assert question["criteria"]["none"] == CHOICE_CRITERIA["none"]
    assert question["criteria"]["token_allocation"] == (
        CHOICE_CRITERIA["token_allocation"]
    )


def test_yes_no_is_a_noul_question() -> None:
    captured: dict[str, Any] = {}

    def post(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        captured["payload"] = payload
        return {
            "model": "jev-1.13.0",
            "answers": {"sufficient": {"type": "noul", "noul": 0.2}},
        }

    decider = JevDecider(api_key="test", post=post, criteria=CHOICE_CRITERIA)
    decision = decider.choose(
        "sufficient",
        "excerpt",
        "Do these excerpts answer the question?",
        ("yes", "no"),
    )

    assert decision.choice == "no"
    assert decision.probability == pytest.approx(0.8)
    question = captured["payload"]["questions"]["sufficient"]
    assert question["type"] == "noul"
    assert question["criteria"]["true"] == CHOICE_CRITERIA["yes"]


def test_jev_rejects_a_choice_outside_the_menu() -> None:
    def post(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        return {
            "answers": {
                "fact": {
                    "type": "choice",
                    "choice": "nope",
                    "probabilities": {"nope": 0.4},
                }
            }
        }

    decider = JevDecider(api_key="test", post=post)

    with pytest.raises(ValueError, match="not one of"):
        decider.choose("fact", "question", "Pick one.", ("none",))


def test_jev_decider_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr("mini_rag.config.load_dotenv", lambda: None)

    with pytest.raises(ValueError, match="TYPESAFE_API_KEY is not set"):
        JevDecider()


def test_missing_criterion_falls_back_to_the_label() -> None:
    captured: dict[str, Any] = {}

    def post(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        captured["payload"] = payload
        return {
            "answers": {
                "fact": {
                    "type": "choice",
                    "choice": "token_allocation",
                    "probabilities": {"token_allocation": 1.0},
                }
            }
        }

    JevDecider(api_key="test", post=post).choose(
        "fact",
        "tokens",
        "Pick one.",
        ("token_allocation",),
    )

    criteria = captured["payload"]["questions"]["fact"]["criteria"]
    assert criteria["token_allocation"] == "token allocation"


def test_every_checked_fact_has_a_criterion() -> None:
    keys = {fact.fact_key for fact in CHECKED_FACTS}
    assert keys <= set(CHOICE_CRITERIA)
