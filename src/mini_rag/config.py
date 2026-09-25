from __future__ import annotations

import os

from dotenv import load_dotenv

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_JEV_BASE_URL = "https://api.typesafe.ai"
DEFAULT_JEV_MODEL = "jev-latest"


def ollama_host() -> str:
    load_dotenv()
    return os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)


def ollama_model() -> str:
    load_dotenv()
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def jev_api_key() -> str:
    load_dotenv()
    key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not key:
        raise ValueError("TYPESAFE_API_KEY is not set")
    return key


def jev_base_url() -> str:
    load_dotenv()
    return os.getenv("TYPESAFE_BASE_URL", DEFAULT_JEV_BASE_URL).rstrip("/")


def jev_model() -> str:
    load_dotenv()
    return os.getenv("TYPESAFE_DEFAULT_MODEL", DEFAULT_JEV_MODEL)
