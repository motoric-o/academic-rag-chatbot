"""Extract the text-backed 2024 handbook into numbered sections."""

import hashlib
from collections import defaultdict
from pathlib import Path
from textwrap import dedent

import pymupdf

from regqa.ingest.structure import Corpus, Heading, Section, normalize_heading, numbered_headings


def parse_pdf(path: Path) -> Corpus:
    pdf_bytes = path.read_bytes()
    rows: list[tuple[int, str]] = []
    starts: list[tuple[int, Heading]] = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        if pdf.needs_pass:
            raise ValueError("The PDF is password-protected")
        headings = numbered_headings(pdf.get_toc(), len(pdf))
        first_page = headings[0].page
        by_page: dict[int, list[Heading]] = defaultdict(list)
        for heading in headings:
            by_page[heading.page].append(heading)

        for page_number in range(first_page, len(pdf) + 1):
            page = pdf[page_number - 1]
            lines = page.get_text("text", sort=True).rstrip().splitlines()
            # ponytail: the 2024 edition numbers body pages from 1 in a standalone footer.
            # Use positioned footer detection if another edition uses a different layout.
            footer = str(page_number - first_page + 1)
            if len(lines) > 1 and not lines[-2].strip() and lines[-1].strip() == footer:
                lines.pop()

            normalized = [normalize_heading(line).casefold() for line in lines]
            for heading in by_page[page_number]:
                matches = [
                    index
                    for index, line in enumerate(normalized)
                    if line
                    in {
                        f"{heading.section_id} {heading.title}".casefold(),
                        f"{heading.section_id}. {heading.title}".casefold(),
                    }
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected one heading for {heading.section_id} on page {page_number}"
                    )
                start = len(rows) + matches[0]
                if starts and start <= starts[-1][0]:
                    raise ValueError(f"Section {heading.section_id} is out of bookmark order")
                starts.append((start, heading))
            rows.extend((page_number, line.rstrip()) for line in lines)

    sections: list[Section] = []
    ends = [start for start, _ in starts[1:]] + [len(rows)]
    for (start, heading), end in zip(starts, ends, strict=True):
        content = rows[start + 1 : end]
        sections.append(
            {
                "section_id": heading.section_id,
                "title": heading.title,
                "text": dedent("\n".join(line for _, line in content)).strip(),
                "page_start": heading.page,
                "page_end": next(
                    (number for number, line in reversed(content) if line.strip()), heading.page
                ),
            }
        )
    if not any(section["text"] for section in sections):
        raise ValueError("No extractable section text found")
    return {
        "language": "id",
        "source_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "sections": sections,
    }
