import pytest

from mini_rag.policy.config import policy_database_url


def test_policy_database_url_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "POLICY_DATABASE_URL", "postgresql://rag:rag@localhost:5432/policy_rag"
    )

    assert policy_database_url() == "postgresql://rag:rag@localhost:5432/policy_rag"


def test_policy_database_url_requires_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLICY_DATABASE_URL", raising=False)
    monkeypatch.setattr("mini_rag.policy.config.load_dotenv", lambda: None)

    with pytest.raises(ValueError, match="POLICY_DATABASE_URL is not set"):
        policy_database_url()
