"""Validated corpus files and non-overwriting artifact publication."""

import hashlib
import json
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from regqa.ingest.structure import Corpus


def read_corpus(data: bytes, *, language: str, synthetic: bool = False) -> Corpus:
    corpus = json.loads(data)
    if not isinstance(corpus, dict) or corpus.get("language") != language:
        raise ValueError(f"Expected a {language} corpus")
    notes = corpus.get("notes", "")
    if not isinstance(notes, str) or synthetic != ("SYNTHETIC_FIXTURE" in notes):
        raise ValueError("SYNTHETIC_FIXTURE corpus requires --synthetic and vice versa")
    if not synthetic and not re.fullmatch(r"[0-9a-f]{64}", str(corpus.get("source_sha256", ""))):
        raise ValueError("Corpus is missing the source PDF SHA-256")
    sections = corpus.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError("Corpus contains no sections")
    ids = set()
    for section in sections:
        if not isinstance(section, dict) or not all(
            isinstance(section.get(key), str) for key in ("section_id", "title", "text")
        ):
            raise ValueError("Each corpus section needs string section_id, title and text")
        section_id = section["section_id"]
        if (
            not section_id
            or section_id != section_id.strip()
            or section_id in ids
            or not section["title"].strip()
        ):
            raise ValueError(f"Invalid or duplicate section ID or title: {section_id!r}")
        start, end = section.get("page_start"), section.get("page_end")
        if type(start) is not int or type(end) is not int or not 1 <= start <= end:
            raise ValueError(f"Invalid page range in section {section_id}")
        ids.add(section_id)
    return cast(Corpus, corpus)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def publish(directory: Path, payloads: dict[str, bytes]) -> None:
    """Stage complete files and roll back links created by a failed publication."""
    for name in payloads:
        if (directory / name).exists():
            raise FileExistsError(f"Output already exists: {directory / name}")
    links = []
    with TemporaryDirectory(dir=directory) as staging:
        try:
            for name, data in payloads.items():
                (Path(staging) / name).write_bytes(data)
            for name in payloads:
                output = directory / name
                os.link(Path(staging) / name, output)
                links.append(output)
        except BaseException:
            for output in links:
                output.unlink()
            raise
