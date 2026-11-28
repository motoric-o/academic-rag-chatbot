"""Parse raw model outputs according to the output contract (docs/project_spec.md §8).

Owner: Member C (Francisco Valentino).
Consumers: Member B (metrics), Member D (eval/frontend).

Contract:
1. Answer must be grounded only in the numbered context passages.
2. End with `Sources: [n], [m]` listing the passage indices used.
3. If unanswerable from context, model replies exactly `NOT_FOUND`.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ParsedOutput:
    """Structured representation of model response after parsing."""

    raw_output: str
    answer_text: str
    cited_ids: list[int]
    abstained: bool
    format_ok: bool
    error_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_SOURCES_REGEX = re.compile(
    r"(?i)\bSources?\s*:\s*([\[\]0-9\s,]+)$",
    re.MULTILINE,
)
_DIGIT_EXTRACTOR = re.compile(r"\b(\d+)\b")


def parse_output(raw_output: str | None) -> ParsedOutput:
    """Parse raw generation text into ParsedOutput according to §8.

    Args:
        raw_output: Raw string from model generation.

    Returns:
        ParsedOutput with extracted fields and format_ok flag.
    """
    if raw_output is None:
        return ParsedOutput(
            raw_output="",
            answer_text="",
            cited_ids=[],
            abstained=False,
            format_ok=False,
            error_reason="raw_output is None",
        )

    text = raw_output.strip()

    # Case 1: Exact or clean abstention
    if text.upper() == "NOT_FOUND" or text.upper().startswith("NOT_FOUND."):
        return ParsedOutput(
            raw_output=raw_output,
            answer_text="NOT_FOUND",
            cited_ids=[],
            abstained=True,
            format_ok=True,
            error_reason=None,
        )

    # If it contains NOT_FOUND along with other verbose text without citation,
    # it's an informal abstention / malformed abstention
    if "NOT_FOUND" in text.upper() and len(text.split()) < 15 and not _SOURCES_REGEX.search(text):
        return ParsedOutput(
            raw_output=raw_output,
            answer_text=text,
            cited_ids=[],
            abstained=True,
            format_ok=True,
            error_reason=None,
        )

    # Case 2: Substantive answer with Sources: [n], [m]
    match = _SOURCES_REGEX.search(text)
    if not match:
        # Missing Sources: [...] citation line
        return ParsedOutput(
            raw_output=raw_output,
            answer_text=text,
            cited_ids=[],
            abstained=False,
            format_ok=False,
            error_reason="Missing 'Sources: [n]' citation line",
        )

    answer_text = text[: match.start()].strip()
    digits = [int(d) for d in _DIGIT_EXTRACTOR.findall(match.group(1))]

    if not answer_text:
        return ParsedOutput(
            raw_output=raw_output,
            answer_text="",
            cited_ids=digits,
            abstained=False,
            format_ok=False,
            error_reason="Empty answer text before citation line",
        )

    if not digits:
        return ParsedOutput(
            raw_output=raw_output,
            answer_text=answer_text,
            cited_ids=[],
            abstained=False,
            format_ok=False,
            error_reason="Sources line present but contains no passage numbers",
        )

    # Deduplicate while preserving order
    seen: set[int] = set()
    deduped_ids: list[int] = []
    for d in digits:
        if d not in seen:
            seen.add(d)
            deduped_ids.append(d)

    return ParsedOutput(
        raw_output=raw_output,
        answer_text=answer_text,
        cited_ids=deduped_ids,
        abstained=False,
        format_ok=True,
        error_reason=None,
    )
