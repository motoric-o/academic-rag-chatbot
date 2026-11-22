"""Numbered sections in the Maranatha academic handbook."""

import re
from typing import NamedTuple, TypedDict
from unicodedata import normalize

NUMBERED_HEADING = re.compile(r"([1-9]\d*(?:\.[1-9]\d*){0,2})\.? (.+)")


class Section(TypedDict):
    section_id: str
    title: str
    text: str
    page_start: int
    page_end: int


class Corpus(TypedDict):
    language: str
    source_sha256: str
    sections: list[Section]


class Heading(NamedTuple):
    section_id: str
    title: str
    page: int


def normalize_heading(text: str) -> str:
    return " ".join(normalize("NFKC", text).split())


def numbered_headings(outline: list[list], page_count: int) -> list[Heading]:
    headings: list[Heading] = []
    seen: set[str] = set()
    for _, title, page in outline:
        match = NUMBERED_HEADING.fullmatch(normalize_heading(title))
        if not match:
            continue
        section_id, name = match.groups()
        parent, _, _ = section_id.rpartition(".")
        if section_id in seen or (parent and parent not in seen):
            raise ValueError(f"Invalid bookmark hierarchy at section {section_id}")
        if not 1 <= page <= page_count or (headings and page < headings[-1].page):
            raise ValueError(f"Invalid bookmark page for section {section_id}: {page}")
        headings.append(Heading(section_id, name, page))
        seen.add(section_id)
    if not headings:
        raise ValueError("The PDF has no numbered section bookmarks")
    return headings
