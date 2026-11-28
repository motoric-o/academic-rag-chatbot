"""Check PDF structure and section-by-section translation on small inputs."""

import json
from pathlib import Path

import pymupdf
import pytest

from regqa.ingest.parse_pdf import parse_pdf
from regqa.ingest.structure import Corpus
from regqa.translate.checks import translation_issues
from regqa.translate.translate_sections import TranslationSettings, translate_sections


def test_numbered_bookmarks_keep_text_between_sections(tmp_path: Path) -> None:
    source = tmp_path / "handbook.pdf"
    with pymupdf.open() as pdf:
        first = pdf.new_page()
        first.insert_text((72, 72), "Daftar Isi\n1 Rules")
        page = pdf.new_page()
        page.insert_text((72, 72), "1 Rules\n1.1 Leave\n1. Submit a form.")
        page.insert_text((280, 800), "1")
        continuation = pdf.new_page()
        continuation.insert_text((72, 72), "Get approval.\n1.2 Internship\nSign the log.")
        continuation.insert_text((280, 800), "2")
        pdf.set_toc([[1, "1 Rules", 2], [2, "1.1 Leave", 2], [2, "1.2 Internship", 3]])
        pdf.save(source)

    sections = parse_pdf(source)["sections"]
    assert [section["section_id"] for section in sections] == ["1", "1.1", "1.2"]
    assert sections[0]["text"] == ""
    assert sections[1]["page_end"] == 3
    assert sections[1]["text"].split() == "1. Submit a form. Get approval.".split()
    assert sections[2]["text"].strip() == "Sign the log."


def test_translate_sections_preserves_ids_and_checks_each_part() -> None:
    source: Corpus = {
        "language": "id",
        "source_sha256": "fixture",
        "sections": [
            {
                "section_id": "1",
                "title": "Kredit semester",
                "text": "Min. 60 SKS.\n\nBatas 2.5%.",
                "page_start": 2,
                "page_end": 3,
            }
        ],
    }
    settings = TranslationSettings(
        temperature=0,
        timeout_ms=1000,
        max_input_chars=12,
        min_length_ratio=0.6,
        max_length_ratio=2.5,
    )
    prompts: list[str] = []
    parts = iter(
        [
            {"title": "Semester credits", "text": "Min. 60 credit units."},
            {"title": "", "text": "Limit 2.5%."},
        ]
    )

    def generate(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps(next(parts))

    draft = translate_sections(source, {"SKS": "credit units"}, settings, generate)
    assert len(prompts) == 2
    assert all("credit units" in prompt for prompt in prompts)
    assert draft["language"] == "en"
    assert draft["sections"] == [
        {
            **source["sections"][0],
            "title": "Semester credits",
            "text": "Min. 60 credit units.\n\nLimit 2.5%.",
        }
    ]
    with pytest.raises(ValueError, match="section 1.*number count or value changed"):
        translate_sections(
            source,
            {"SKS": "credit units"},
            settings,
            lambda _prompt: '{"title":"Semester credits","text":"Min. 61 credit units."}',
        )


def test_dates_percentages_and_numbers() -> None:
    source = "12 Januari 2024: IPK 2.5, bobot 40% dan 60%."
    translated = "January 12, 2024: cumulative GPA 2.5, weights of 40% and 60%."
    glossary = {"IPK": "cumulative GPA"}
    assert not translation_issues(source, translated, glossary, min_ratio=0.6, max_ratio=1.8)
    issues = translation_issues(
        source,
        translated.replace("January", "March").replace("2.5", "5.2").replace("40%", "40"),
        glossary,
        min_ratio=0.6,
        max_ratio=1.8,
    )
    assert set(issues) == {
        "date changed or missing",
        "percentage value changed",
        "number count or value changed",
    }


def test_glossary_prefers_full_phrases_and_still_checks_standalone_terms() -> None:
    glossary = {
        "Mata Kuliah": "course",
        "wajib": "must",
        "Mata Kuliah Wajib": "Required Courses",
        "Tugas Akhir": "Undergraduate Thesis",
        "Seminar Tugas Akhir": "Undergraduate Thesis Seminar",
    }
    limits = {"min_ratio": 0.6, "max_ratio": 1.8}
    assert not translation_issues("Mata Kuliah Wajib", "Required Course", glossary, **limits)
    assert not translation_issues("Mata Kuliah", "Courses", glossary, **limits)
    assert translation_issues(
        "Mata Kuliah Wajib. Mahasiswa wajib lulus.",
        "Required Courses. Students should pass.",
        glossary,
        **limits,
    ) == ["glossary term missing: wajib -> must"]
    assert translation_issues(
        "Seminar Tugas Akhir dan Tugas Akhir",
        "Undergraduate Thesis Seminar and final project",
        glossary,
        **limits,
    ) == ["glossary term missing: Tugas Akhir -> Undergraduate Thesis"]
