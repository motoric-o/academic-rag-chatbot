"""Make token-bounded chunks from the numbered handbook sections."""

import re
from collections.abc import Callable
from typing import NotRequired, TypedDict

from regqa.ingest.structure import Section

PARAGRAPH = re.compile(r"\n(?:[ \t]*\n)+|(?<=\n)(?=[ \t]*(?:\d+|[a-z])[.)][ \t]+)")
PARAGRAPH_LABEL = re.compile(r"^[ \t]*([1-9]\d*)[.)][ \t]+")
# Keep initials, numbered list markers and common abbreviations with their next word.
SENTENCE = re.compile(
    r"(?<!\b[A-Za-z\d]\.)(?<!\b\d\d\.)(?<!\b[Nn]o\.)(?<!\b[DdMm][rs]\.)"
    r"(?<!\bMrs\.)(?<!\bProf\.)(?<!\b[Mm][ia][nx]\.)(?<!\betc\.)"
    r"(?<=[.!?])\s+(?=[\"\u201c(]*[A-Z0-9])"
)


class Passage(TypedDict):
    chunk_id: str
    section_ids: list[str]
    text: str


class Chunk(Passage):
    # Internal paragraph metadata. Cache consumers use the shared Passage fields.
    paragraph_ids: NotRequired[list[str]]


class ChunkLimitError(ValueError):
    """A heading or indivisible sentence cannot fit the candidate."""


def join_parts(parts: list[tuple[int, str]]) -> str:
    text = []
    previous = None
    for number, part in parts:
        if text:
            text.append(" " if number == previous else "\n\n")
        text.append(part)
        previous = number
    return "".join(text)


def make_chunks(
    sections: list[Section],
    count: Callable[[str], int],
    *,
    limit: int,
    overlap_ratio: float,
    header_template: str,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        section_id = section["section_id"]
        header = section_header(section, header_template)
        parts = split_section(header, section["text"], count, limit, int(limit * overlap_ratio))
        for i, (text, paragraphs) in enumerate(parts, 1):
            chunk: Chunk = {
                "chunk_id": f"{section_id}:{i}",
                "section_ids": [section_id],
                "text": text,
            }
            if paragraphs is not None:
                chunk["paragraph_ids"] = paragraphs
            chunks.append(chunk)
    return chunks


def section_header(section: Section, template: str) -> str:
    section_id = section["section_id"]
    header = template.format(chapter=section_id.split(".")[0], article=section_id)
    return f"{header} {section['title']}"


def split_section(
    header: str, body: str, count: Callable[[str], int], limit: int, overlap: int
) -> list[tuple[str, list[str] | None]]:
    full = f"{header}\n{body}".rstrip()
    if count(full) <= limit:
        return [(full, None)]

    def fits(parts: list[tuple[int, str]]) -> bool:
        return count(f"{header}\n{join_parts(parts)}") <= limit

    groups = []
    labels = {}
    current_label = None
    for number, paragraph in enumerate(PARAGRAPH.split(body)):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        match = PARAGRAPH_LABEL.match(paragraph)
        if match:
            current_label = match[1]
        if current_label is not None:
            labels[number] = current_label
        if fits([(number, paragraph)]):
            groups.append((number, paragraph))
        else:
            # ponytail: keep unknown sentence boundaries intact. Use a verified
            # segmenter if real handbook sentences can no longer fit the candidate grid.
            groups.extend((number, sentence) for sentence in SENTENCE.split(paragraph))

    if not groups:
        raise ChunkLimitError(f"{header}: heading exceeds the {limit}-token limit")

    def pack(parts: list[tuple[int, str]]) -> tuple[str, list[str]]:
        ids = list(dict.fromkeys(labels[number] for number, _ in parts if number in labels))
        return f"{header}\n{join_parts(parts)}", ids

    result: list[tuple[str, list[str] | None]] = []
    current: list[tuple[int, str]] = []
    for unit in groups:
        if not fits([unit]):
            raise ChunkLimitError(f"{header}: a sentence exceeds the {limit}-token limit")
        if current and not fits([*current, unit]):
            result.append(pack(current))
            retained: list[tuple[int, str]] = []
            for previous in reversed(current):
                tail = [previous, *retained]
                if (
                    previous[0] != unit[0]
                    or overlap == 0
                    or count(join_parts(tail)) > overlap
                    or not fits([*tail, unit])
                ):
                    break
                retained = tail
            current = retained
        current.append(unit)
    result.append(pack(current))
    return result
