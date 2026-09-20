import pytest

from mini_rag.config import database_url


def test_database_url_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/mini_rag")

    assert database_url() == "postgresql://rag:rag@localhost:5432/mini_rag"


def test_database_url_requires_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr("mini_rag.config.load_dotenv", lambda: None)

    with pytest.raises(ValueError, match="DATABASE_URL is not set"):
        database_url()
