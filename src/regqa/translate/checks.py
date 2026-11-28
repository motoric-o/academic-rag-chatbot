"""Check details that a translation must preserve."""

import re
from collections import Counter
from pathlib import Path
from unicodedata import normalize

import yaml

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


def load_glossary(path: Path) -> dict[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    glossary: dict[str, str] = {}
    for group in ("terms", "modal_verbs"):
        terms = data.get(group) if isinstance(data, dict) else None
        if not isinstance(terms, dict) or not terms or not all(
            isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip()
            for k, v in terms.items()
        ):
            raise ValueError(f"{path}: {group} must map terms to non-empty translations")
        glossary.update(terms)
    return glossary


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


PLURALS = {
    "chapter": "chapters",
    "part": "parts",
    "article": "articles",
    "paragraph": "paragraphs",
    "thesis": "theses",
    "seminar": "seminars",
    "course": "courses",
    "topic": "topics",
    "exam": "exams",
    "program": "programs",
    "prerequisite": "prerequisites",
    "conversion": "conversions",
    "internship": "internships",
    "coordinator": "coordinators",
    "semester": "semesters",
    "supervisor": "supervisors",
    "sanction": "sanctions",
}
NOUN_FORMS = {**PLURALS, **{plural: singular for singular, plural in PLURALS.items()}}


def glossary_matches(text: str, forms: dict[str, str]) -> set[str]:
    if not forms:
        return set()
    alternatives = "|".join(re.escape(form) for form in sorted(forms, key=len, reverse=True))
    return {forms[match] for match in re.findall(rf"(?<!\w)(?:{alternatives})(?!\w)", text)}


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
    source_forms = {normalized(term): term for term in glossary}
    english_forms = {}
    for equivalent in glossary.values():
        canonical = normalized(equivalent)
        english_forms[canonical] = canonical
        prefix, space, noun = canonical.rpartition(" ")
        if noun in NOUN_FORMS:
            english_forms[prefix + space + NOUN_FORMS[noun]] = canonical
    required = glossary_matches(original, source_forms)
    used = glossary_matches(english, english_forms)
    for term, equivalent in glossary.items():
        if term in required and normalized(equivalent) not in used:
            issues.append(f"glossary term missing: {term} -> {equivalent}")
    return issues
