import pytest

from mini_rag.config import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    database_url,
    ollama_host,
    ollama_model,
)


def test_database_url_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/mini_rag")

    assert database_url() == "postgresql://rag:rag@localhost:5432/mini_rag"


def test_database_url_requires_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr("mini_rag.config.load_dotenv", lambda: None)

    with pytest.raises(ValueError, match="DATABASE_URL is not set"):
        database_url()


def test_ollama_settings_use_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setattr("mini_rag.config.load_dotenv", lambda: None)

    assert ollama_host() == DEFAULT_OLLAMA_HOST
    assert ollama_model() == DEFAULT_OLLAMA_MODEL


def test_ollama_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")

    assert ollama_host() == "http://127.0.0.1:11434"
    assert ollama_model() == "qwen3:8b"
