from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
POLICY_EMBEDDING_DIM = 768


@dataclass(frozen=True)
class ExtractedDocument:
    """Normalized policy text. The chunker never sees the original file bytes."""

    source_file: str
    text: str


@dataclass(frozen=True)
class PolicyFact:
    document_slug: str
    version: str
    fact_key: str
    value: str
    section: str


@dataclass(frozen=True)
class PolicyChunk:
    """One stored section. `text` is the body. `embed_input` is what the model sees."""

    chunk_id: str
    document_slug: str
    document: str
    version: str
    section: str
    section_title: str
    parent_section: str | None
    parent_title: str | None
    text: str
    embed_input: str
    source_file: str
    search_text: str
    embedding: tuple[float, ...]

    def with_embedding(self, embedding: Sequence[float]) -> PolicyChunk:
        return replace(self, embedding=tuple(embedding))


@dataclass(frozen=True)
class RetrievedPolicyChunk:
    """A stored policy chunk plus its cosine distance from the query vector."""

    chunk_id: str
    document_slug: str
    document: str
    version: str
    section: str
    section_title: str
    parent_section: str | None
    parent_title: str | None
    text: str
    distance: float


@dataclass(frozen=True)
class KeywordHit:
    """A stored policy chunk plus its keyword rank score."""

    chunk_id: str
    document_slug: str
    document: str
    version: str
    section: str
    section_title: str
    parent_section: str | None
    parent_title: str | None
    text: str
    rank: float


@dataclass(frozen=True)
class IngestFileSummary:
    source_file: str
    document_slug: str
    version: str
    chunks: int


def normalize_version(version: str) -> str:
    """Strip a leading v so IDs use v2.0 rather than vv2.0."""
    stripped = version.strip()
    if len(stripped) >= 2 and stripped[0] in {"v", "V"} and stripped[1].isdigit():
        return stripped[1:]
    return stripped


def make_chunk_id(version: str, section: str, *, document_slug: str) -> str:
    return f"{document_slug}:v{normalize_version(version)}:section-{section}"


DOCUMENT_TITLES = {
    "health-wellness": "Health & Wellness Policy",
    "hr": "HR Policy",
    "preparedness": "Preparedness Policy",
    "time-usage": "Time & Usage Policy",
}


def document_title(slug: str) -> str:
    """Return the policy title stored for a document slug."""
    try:
        return DOCUMENT_TITLES[slug]
    except KeyError:
        raise ValueError(f"Unknown document slug {slug!r}") from None


def document_slug(title: str) -> str:
    """Turn a policy title into an id. 'Time & Usage Policy' becomes time-usage."""
    name = title.strip()
    if name.lower().endswith(" policy"):
        name = name[: -len(" policy")]
    slug = []
    previous_dash = False
    for char in name.casefold().replace("&", " "):
        if char.isalnum():
            slug.append(char)
            previous_dash = False
        elif not previous_dash:
            slug.append("-")
            previous_dash = True
    return "".join(slug).strip("-")


def embed_prefix(
    document: str,
    version: str,
    section: str,
    section_title: str,
    parent_section: str | None,
    parent_title: str | None,
) -> str:
    heading = _heading(section, section_title)
    version_label = normalize_version(version)
    if parent_section and parent_title:
        parent = f"{parent_section}. {parent_title}"
        return f"{document} v{version_label} — {parent} — {heading}"
    return f"{document} v{version_label} — {heading}"


def _heading(section: str, title: str) -> str:
    if "." in section:
        return f"{section} {title}"
    return f"{section}. {title}"


def section_covers(chunk_section: str, fact_section: str) -> bool:
    return chunk_section == fact_section or chunk_section.startswith(f"{fact_section}.")
