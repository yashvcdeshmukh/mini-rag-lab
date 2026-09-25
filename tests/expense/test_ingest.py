from __future__ import annotations

from pathlib import Path

import pytest

from fakes import FakeEmbedder
from mini_rag.db import InMemoryDatabase
from mini_rag.expense.ingest import ingest_policy

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "policy.md"


def test_ingest_policy_writes_six_records() -> None:
    database = InMemoryDatabase()

    records = ingest_policy(POLICY_PATH, FakeEmbedder(), database)

    assert len(records) == 6
    assert database.count() == 6
    assert records[0].chunk_id == "expense-policy:v2.0:section-1"
    assert records[0].section_title == "Meals"
    assert records[0].document == "Employee Expense Policy"
    assert len(records[0].embedding) == FakeEmbedder().dimension


def test_ingest_policy_is_idempotent() -> None:
    database = InMemoryDatabase()
    ingest_policy(POLICY_PATH, FakeEmbedder(), database)

    records = ingest_policy(POLICY_PATH, FakeEmbedder(), database)

    assert database.count() == 6
    assert records[5].chunk_id == "expense-policy:v2.0:section-6"


def test_ingest_policy_fails_when_store_count_is_wrong() -> None:
    class BrokenDatabase(InMemoryDatabase):
        def count(self) -> int:
            return 5

    with pytest.raises(ValueError, match="Expected 6 stored chunks"):
        ingest_policy(POLICY_PATH, FakeEmbedder(), BrokenDatabase())


def test_ingest_policy_fails_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ingest_policy(tmp_path / "missing.md", FakeEmbedder(), InMemoryDatabase())
