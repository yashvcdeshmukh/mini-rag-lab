from __future__ import annotations

import re
from pathlib import Path

from mini_rag.expense.models import ParsedPolicy, PolicySection

HEADER_RE = re.compile(
    r"^#\s+(?P<title>.+?)\s+[—–-]\s+Version\s+(?P<version>\S+)\s*$",
    re.MULTILINE,
)
SECTION_HEADING_RE = re.compile(
    r"^##\s+(?P<number>\d+)\.\s+(?P<title>.+?)\s*$",
    re.MULTILINE,
)


def parse_policy_file(path: str | Path) -> ParsedPolicy:
    return parse_policy(Path(path).read_text(encoding="utf-8"))


def parse_policy(markdown: str) -> ParsedPolicy:
    header = HEADER_RE.search(markdown)
    if header is None:
        raise ValueError(
            "Policy header with title and version was not found. "
            "Expected a line like '# Title — Version 2.0'."
        )

    sections = _parse_sections(markdown)
    if not sections:
        raise ValueError("No numbered sections found")

    return ParsedPolicy(
        document=header.group("title").strip(),
        version=header.group("version").strip(),
        sections=tuple(sections),
    )


def _parse_sections(markdown: str) -> list[PolicySection]:
    headings = list(SECTION_HEADING_RE.finditer(markdown))
    sections: list[PolicySection] = []

    for index, match in enumerate(headings):
        body_start = match.end()
        if index + 1 < len(headings):
            body_end = headings[index + 1].start()
        else:
            body_end = len(markdown)
        text = markdown[body_start:body_end].strip()
        if not text:
            raise ValueError(f"Section {match.group('number')} has no body text")

        sections.append(
            PolicySection(
                section=match.group("number"),
                section_title=match.group("title").strip(),
                text=text,
            )
        )

    return sections
