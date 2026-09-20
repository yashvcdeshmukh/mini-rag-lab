from __future__ import annotations

import os

from dotenv import load_dotenv

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"


def database_url() -> str:
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL is not set")
    return url


def ollama_host() -> str:
    load_dotenv()
    return os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)


def ollama_model() -> str:
    load_dotenv()
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
