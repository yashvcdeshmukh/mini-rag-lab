from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from mini_rag.policy.models import (
    ExtractedDocument,
    PolicyChunk,
    document_slug,
    embed_prefix,
    make_chunk_id,
    normalize_version,
)

MAX_EMBED_TOKENS = 512
WINDOW_TOKENS = 400

_HEADER_RE = re.compile(
    r"^(?P<title>.+?)\s+[—-]\s+Version\s+(?P<version>\S+)\s*$"
)
_TOP_RE = re.compile(r"^(?P<number>\d+)\.\s+(?P<title>\S.*?)\s*$")
_SUB_RE = re.compile(r"^(?P<number>\d+\.\d+)\s+(?P<rest>\S.*?)\s*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


@dataclass(frozen=True)
class _Section:
    number: str
    title: str
    body: str


@dataclass(frozen=True)
class ParsedPolicyText:
    document: str
    version: str
    sections: tuple[_Section, ...]


def chunk_document(
    document: ExtractedDocument,
    counter: TokenCounter,
) -> list[PolicyChunk]:
    parsed = parse_policy_text(document.text)
    slug = document_slug(parsed.document)
    version = normalize_version(parsed.version)
    chunks: list[PolicyChunk] = []
    for section in parsed.sections:
        chunks.extend(
            _chunks_for_section(
                section,
                document=parsed.document,
                version=version,
                slug=slug,
                source_file=document.source_file,
                counter=counter,
            )
        )
    _reject_duplicate_ids(chunks)
    return chunks


def parse_policy_text(text: str) -> ParsedPolicyText:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header_at: int | None = None
    title = ""
    version = ""
    for index, line in enumerate(lines):
        header = _HEADER_RE.match(line)
        if header is not None:
            title = header.group("title").strip()
            version = header.group("version").strip()
            header_at = index
            break
    if header_at is None:
        raise ValueError(
            "Policy header with title and version was not found. "
            "Expected a line like 'HR Policy — Version 2.0'."
        )

    sections: list[_Section] = []
    current_number = ""
    current_title = ""
    current_lines: list[str] = []

    def finish() -> None:
        if not current_number:
            return
        body = "\n".join(current_lines).strip()
        if not body:
            raise ValueError(f"Section {current_number} has no body text")
        sections.append(_Section(current_number, current_title, body))

    for line in lines[header_at + 1 :]:
        heading = _TOP_RE.match(line)
        if heading is not None:
            finish()
            current_number = heading.group("number")
            current_title = heading.group("title").strip()
            current_lines = []
            continue
        if not current_number:
            continue
        current_lines.append(line)
    finish()
    if not sections:
        raise ValueError("No numbered sections found")
    return ParsedPolicyText(title, version, tuple(sections))


def _chunks_for_section(
    section: _Section,
    *,
    document: str,
    version: str,
    slug: str,
    source_file: str,
    counter: TokenCounter,
) -> list[PolicyChunk]:
    prefix = embed_prefix(document, version, section.number, section.title, None, None)
    if counter.count(f"{prefix}\n{section.body}") <= MAX_EMBED_TOKENS:
        return [
            _make_chunk(
                document=document,
                version=version,
                slug=slug,
                section=section.number,
                section_title=section.title,
                parent_section=None,
                parent_title=None,
                text=section.body,
                source_file=source_file,
                part=None,
            )
        ]

    subsections = _split_subsections(section.body)
    if not subsections:
        return _windowed_chunks(
            document=document,
            version=version,
            slug=slug,
            section=section.number,
            section_title=section.title,
            parent_section=None,
            parent_title=None,
            body=section.body,
            source_file=source_file,
            counter=counter,
        )

    chunks: list[PolicyChunk] = []
    for subsection in subsections:
        sub_prefix = embed_prefix(
            document,
            version,
            subsection.number,
            subsection.title,
            section.number,
            section.title,
        )
        if counter.count(f"{sub_prefix}\n{subsection.body}") <= MAX_EMBED_TOKENS:
            chunks.append(
                _make_chunk(
                    document=document,
                    version=version,
                    slug=slug,
                    section=subsection.number,
                    section_title=subsection.title,
                    parent_section=section.number,
                    parent_title=section.title,
                    text=subsection.body,
                    source_file=source_file,
                    part=None,
                )
            )
            continue
        chunks.extend(
            _windowed_chunks(
                document=document,
                version=version,
                slug=slug,
                section=subsection.number,
                section_title=subsection.title,
                parent_section=section.number,
                parent_title=section.title,
                body=subsection.body,
                source_file=source_file,
                counter=counter,
            )
        )
    return chunks


def _windowed_chunks(
    *,
    document: str,
    version: str,
    slug: str,
    section: str,
    section_title: str,
    parent_section: str | None,
    parent_title: str | None,
    body: str,
    source_file: str,
    counter: TokenCounter,
) -> list[PolicyChunk]:
    prefix = embed_prefix(
        document,
        version,
        section,
        section_title,
        parent_section,
        parent_title,
    )
    parts = _window_body(body, counter, prefix)
    chunks: list[PolicyChunk] = []
    for index, part in enumerate(parts, start=1):
        chunks.append(
            _make_chunk(
                document=document,
                version=version,
                slug=slug,
                section=section,
                section_title=section_title,
                parent_section=parent_section,
                parent_title=parent_title,
                text=part,
                source_file=source_file,
                part=index if len(parts) > 1 else None,
            )
        )
    return chunks


def _window_body(body: str, counter: TokenCounter, prefix: str) -> list[str]:
    sentences = _sentences(body)
    windows: list[list[str]] = []
    current: list[str] = []

    def token_count(parts: Sequence[str]) -> int:
        return counter.count(f"{prefix}\n{' '.join(parts)}")

    for sentence in sentences:
        if token_count([sentence]) > WINDOW_TOKENS:
            if current:
                windows.append(current)
                current = []
            windows.extend(_word_windows(sentence, counter, prefix))
            continue
        trial = [*current, sentence]
        if current and token_count(trial) > WINDOW_TOKENS:
            windows.append(current)
            current = [current[-1], sentence]
            if token_count(current) > WINDOW_TOKENS:
                current = [sentence]
        else:
            current = trial
    if current:
        windows.append(current)
    if not windows:
        raise ValueError("Section body produced no windows")
    return [" ".join(parts) for parts in windows]


def _word_windows(sentence: str, counter: TokenCounter, prefix: str) -> list[list[str]]:
    words = sentence.split()
    if not words:
        raise ValueError("Cannot window an empty sentence")
    windows: list[list[str]] = []
    current: list[str] = []
    for word in words:
        trial = [*current, word]
        if current and counter.count(f"{prefix}\n{' '.join(trial)}") > WINDOW_TOKENS:
            windows.append(current)
            current = [word]
        else:
            current = trial
    if current:
        windows.append(current)
    return windows


def _sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        raise ValueError("Section body is empty")
    return [part for part in _SENTENCE_RE.split(compact) if part]


def _split_subsections(body: str) -> list[_Section]:
    subsections: list[_Section] = []
    number = ""
    title = ""
    lines: list[str] = []
    preamble: list[str] = []
    found = False

    def finish() -> None:
        if not number:
            return
        text = "\n".join(line for line in lines if line).strip()
        if not text:
            raise ValueError(f"Section {number} has no body text")
        subsections.append(_Section(number, title, text))

    for line in body.splitlines():
        match = _SUB_RE.match(line)
        if match is None:
            if number:
                lines.append(line)
            else:
                preamble.append(line)
            continue
        finish()
        found = True
        number = match.group("number")
        title, first = _subsection_title(match.group("rest"))
        lines = [*preamble, first] if preamble else ([first] if first else [])
        preamble = []
    finish()
    if not found:
        return []
    return subsections


def _subsection_title(rest: str) -> tuple[str, str]:
    if ". " in rest:
        title, body = rest.split(". ", 1)
        return title.strip(), body.strip()
    return rest.strip(), ""


def _make_chunk(
    *,
    document: str,
    version: str,
    slug: str,
    section: str,
    section_title: str,
    parent_section: str | None,
    parent_title: str | None,
    text: str,
    source_file: str,
    part: int | None,
) -> PolicyChunk:
    if not text.strip():
        raise ValueError(f"Section {section} has no body text")
    prefix = embed_prefix(
        document,
        version,
        section,
        section_title,
        parent_section,
        parent_title,
    )
    embed_input = f"{prefix}\n{text}"
    chunk_id = make_chunk_id(version, section, document_slug=slug)
    if part is not None:
        chunk_id = f"{chunk_id}:part-{part}"
    search_bits = [section_title]
    if parent_title:
        search_bits.append(parent_title)
    search_bits.append(text)
    return PolicyChunk(
        chunk_id=chunk_id,
        document_slug=slug,
        document=document,
        version=normalize_version(version),
        section=section,
        section_title=section_title,
        parent_section=parent_section,
        parent_title=parent_title,
        text=text,
        embed_input=embed_input,
        source_file=source_file,
        search_text="\n".join(search_bits),
        embedding=(),
    )


def _reject_duplicate_ids(chunks: Sequence[PolicyChunk]) -> None:
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) == len(set(chunk_ids)):
        return
    duplicates = sorted(
        {chunk_id for chunk_id in chunk_ids if chunk_ids.count(chunk_id) > 1}
    )
    raise ValueError(f"Duplicate chunk_id values: {duplicates}")
