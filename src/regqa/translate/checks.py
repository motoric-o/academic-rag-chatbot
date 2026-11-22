"""Check details that a translation must preserve."""

import re
from collections import Counter
from unicodedata import normalize

ID_MONTHS = (
    "januari februari maret april mei juni juli agustus september oktober november desember"
).split()
EN_MONTHS = (
    "january february march april may june july august september october november december"
).split()
MONTHS = dict(zip(ID_MONTHS, EN_MONTHS, strict=True))
MONTH_PATTERN = "|".join(sorted(MONTHS.keys() | MONTHS.values()))
DATED = re.compile(r"\b(\d{1,2}) (" + MONTH_PATTERN + r")(?: (\d{4}))?\b")
DATED_MONTH_FIRST = re.compile(r"\b(" + MONTH_PATTERN + r") (\d{1,2})(?:,? (\d{4}))?\b")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
PERCENT = re.compile("(" + NUMBER.pattern + r")\s*%")
NUMERIC_DATE = re.compile(r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b")


def normalized(text: str) -> str:
    return " ".join(normalize("NFKC", text).casefold().split())


def date_parts(text: str) -> Counter[tuple[str, str, str]]:
    found: Counter[tuple[str, str, str]] = Counter()
    for day, month, year in DATED.findall(text):
        found[str(int(day)), MONTHS.get(month, month), year] += 1
    # Remove day-first dates before accepting month-first dates in the remainder.
    remainder = DATED.sub(" ", text)
    for month, day, year in DATED_MONTH_FIRST.findall(remainder):
        found[str(int(day)), MONTHS.get(month, month), year] += 1
    return found


def contains_term(text: str, term: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(normalized(term)) + r"(?!\w)", text) is not None


def translation_issues(
    source: str,
    target: str,
    glossary: dict[str, str],
    *,
    min_ratio: float,
    max_ratio: float,
) -> list[str]:
    original = normalized(source)
    english = normalized(target)
    if not original:
        return ["empty source gained text"] if english else []
    if not english:
        return ["translation is empty"]

    issues = []
    if Counter(NUMBER.findall(original)) != Counter(NUMBER.findall(english)):
        issues.append("number count or value changed")
    if date_parts(original) - date_parts(english) or Counter(
        NUMERIC_DATE.findall(original)
    ) - Counter(NUMERIC_DATE.findall(english)):
        issues.append("date changed or missing")
    if original.count("%") != english.count("%") or Counter(PERCENT.findall(original)) != Counter(
        PERCENT.findall(english)
    ):
        issues.append("percentage value changed")

    ratio = len(english.replace(" ", "")) / len(original.replace(" ", ""))
    if not min_ratio <= ratio <= max_ratio:
        issues.append(f"length ratio {ratio:.2f} is outside [{min_ratio:g}, {max_ratio:g}]")
    for term, equivalent in glossary.items():
        if contains_term(original, term) and not contains_term(english, equivalent):
            issues.append(f"glossary term missing: {term} -> {equivalent}")
    return issues
