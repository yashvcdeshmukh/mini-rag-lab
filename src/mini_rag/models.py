from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# ID namespace used in chunk_id. This is not a slug of `document`; the
# assignment example uses "expense-policy" while the title stays human-readable.
DOCUMENT_SLUG = "expense-policy"


@dataclass(frozen=True)
class PolicySection:
    """One numbered policy section, before an embedding is attached."""

    section: str
    section_title: str
    text: str


@dataclass(frozen=True)
class ParsedPolicy:
    document: str
    version: str
    sections: tuple[PolicySection, ...]


@dataclass(frozen=True)
class ChunkRecord:
    """Stored chunk: original text, embedding, and source metadata."""

    chunk_id: str
    document: str
    version: str
    section: str
    section_title: str
    text: str
    embedding: list[float]


def normalize_version(version: str) -> str:
    """Strip a leading v so IDs use v2.0 rather than vv2.0."""
    stripped = version.strip()
    if len(stripped) >= 2 and stripped[0] in {"v", "V"} and stripped[1].isdigit():
        return stripped[1:]
    return stripped


def make_chunk_id(
    version: str,
    section: str,
    *,
    document_slug: str = DOCUMENT_SLUG,
) -> str:
    return f"{document_slug}:v{normalize_version(version)}:section-{section}"


def build_chunk_records(
    policy: ParsedPolicy,
    embeddings: Sequence[Sequence[float]],
    *,
    document_slug: str = DOCUMENT_SLUG,
) -> list[ChunkRecord]:
    if len(embeddings) != len(policy.sections):
        raise ValueError(
            f"Expected {len(policy.sections)} embeddings, got {len(embeddings)}"
        )

    records: list[ChunkRecord] = []
    for section, embedding in zip(policy.sections, embeddings, strict=True):
        records.append(
            ChunkRecord(
                chunk_id=make_chunk_id(
                    policy.version,
                    section.section,
                    document_slug=document_slug,
                ),
                document=policy.document,
                version=policy.version,
                section=section.section,
                section_title=section.section_title,
                text=section.text,
                embedding=list(embedding),
            )
        )

    chunk_ids = [record.chunk_id for record in records]
    if len(chunk_ids) != len(set(chunk_ids)):
        duplicates = sorted(
            {chunk_id for chunk_id in chunk_ids if chunk_ids.count(chunk_id) > 1}
        )
        raise ValueError(f"Duplicate chunk_id values: {duplicates}")

    return records
