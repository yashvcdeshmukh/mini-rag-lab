from pathlib import Path

import pytest

from mini_rag.chunking import parse_policy, parse_policy_file
from mini_rag.models import (
    ParsedPolicy,
    PolicySection,
    build_chunk_records,
    make_chunk_id,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "policy.md"

EXPECTED_SECTIONS = [
    ("1", "Meals", "Employees may claim up to $65 per day"),
    ("2", "Hotels", "Hotels are reimbursable up to $225 per night"),
    ("3", "Airfare", "Employees must purchase economy airfare"),
    ("4", "Ground Transportation", "Luxury vehicle upgrades are not reimbursable"),
    ("5", "Receipts", "Receipts are required for individual expenses of $25 or more"),
    ("6", "Submission Deadline", "Expense reports must be submitted within 30 days"),
]


def test_policy_file_splits_into_six_sections() -> None:
    policy = parse_policy_file(POLICY_PATH)

    assert policy.document == "Employee Expense Policy"
    assert policy.version == "2.0"
    assert len(policy.sections) == 6

    for section, (number, title, snippet) in zip(
        policy.sections, EXPECTED_SECTIONS, strict=True
    ):
        assert section.section == number
        assert section.section_title == title
        assert snippet in section.text
        assert section.text.endswith(".")


def test_header_is_not_stored_as_a_chunk() -> None:
    policy = parse_policy_file(POLICY_PATH)
    titles = [section.section_title for section in policy.sections]
    assert "Employee Expense Policy" not in titles


def test_section_bodies_keep_whole_sentences() -> None:
    meals = parse_policy_file(POLICY_PATH).sections[0]

    assert meals.text == (
        "Employees may claim up to $65 per day for meals while traveling overnight.\n"
        "Alcohol is not reimbursable."
    )


def test_build_chunk_records_attaches_ids_and_embeddings() -> None:
    policy = parse_policy_file(POLICY_PATH)
    embeddings = [[float(i), 0.1] for i in range(len(policy.sections))]

    records = build_chunk_records(policy, embeddings)

    assert len(records) == 6
    assert records[0].chunk_id == "expense-policy:v2.0:section-1"
    assert records[0].document == "Employee Expense Policy"
    assert records[0].version == "2.0"
    assert records[0].section == "1"
    assert records[0].section_title == "Meals"
    assert records[0].text == policy.sections[0].text
    assert records[0].embedding == [0.0, 0.1]
    assert records[5].chunk_id == "expense-policy:v2.0:section-6"


def test_build_chunk_records_rejects_embedding_count_mismatch() -> None:
    policy = parse_policy_file(POLICY_PATH)

    with pytest.raises(ValueError, match="Expected 6 embeddings"):
        build_chunk_records(policy, [[0.1]])


def test_build_chunk_records_rejects_duplicate_chunk_ids() -> None:
    policy = ParsedPolicy(
        document="Employee Expense Policy",
        version="2.0",
        sections=(
            PolicySection("1", "Meals", "Meal text."),
            PolicySection("1", "Meals again", "Other meal text."),
        ),
    )

    with pytest.raises(ValueError, match="Duplicate chunk_id"):
        build_chunk_records(policy, [[0.1], [0.2]])


def test_make_chunk_id_normalizes_version_prefix() -> None:
    assert make_chunk_id("2.0", "1") == "expense-policy:v2.0:section-1"
    assert make_chunk_id("v2.0", "1") == "expense-policy:v2.0:section-1"
    assert make_chunk_id("V2.1", "3") == "expense-policy:v2.1:section-3"


def test_parse_policy_requires_header_and_sections() -> None:
    with pytest.raises(ValueError, match="title and version"):
        parse_policy("## 1. Meals\nSome text.")

    with pytest.raises(ValueError, match="No numbered sections"):
        parse_policy("# Employee Expense Policy — Version 2.0\n")
