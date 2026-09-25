from __future__ import annotations

import os

from dotenv import load_dotenv


def policy_database_url() -> str:
    load_dotenv()
    url = os.getenv("POLICY_DATABASE_URL")
    if not url:
        raise ValueError("POLICY_DATABASE_URL is not set")
    return url
