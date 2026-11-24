"""CPU-only unit tests for Experiment A1 (src/regqa/analysis/tokenizer_stats.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from regqa.analysis.tokenizer_stats import (
    analyze_tokenizers,
    compare_language_tokenization,
    format_tokenizer_markdown,
)


class _WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))


def test_compare_language_tokenization_id_and_en() -> None:
    id_sections = [
        {"section_id": "8.1", "title": "Syarat KP", "text": "Menyelesaikan minimal 60 SKS."},
        {"section_id": "8.2", "title": "Jenis KP", "text": ""},
    ]
    en_sections = [
        {"section_id": "8.1", "title": "PW Req", "text": "Complete at least 60 credit units."},
        {"section_id": "8.2", "title": "PW Types", "text": ""},
    ]

    comp = compare_language_tokenization(
        tokenizer_id="mock/qwen3",
        tokenizer=_WhitespaceTokenizer(),
        id_sections=id_sections,
        en_sections=en_sections,
    )
    assert comp.id_stats is not None
    assert comp.en_stats is not None
    assert comp.id_stats.n_sections == 2
    assert comp.id_stats.total_tokens == 8  # (2+4) + 2
    assert comp.en_stats.total_tokens == 10  # (2+6) + 2
    assert comp.matched_sections == 2
    assert comp.token_ratio_id_over_en == pytest.approx(0.8)

    md = format_tokenizer_markdown([comp])
    assert "mock/qwen3" in md
    assert "0.800" in md


def test_analyze_tokenizers_id_only_from_json(tmp_path: Path) -> None:
    id_path = tmp_path / "corpus.id.json"
    id_path.write_text(
        json.dumps(
            {
                "language": "id",
                "sections": [
                    {"section_id": "1", "title": "Pendahuluan", "text": "Satu dua tiga."}
                ],
            }
        ),
        encoding="utf-8",
    )
    results = analyze_tokenizers(
        ["mock/qwen3"],
        id_corpus_path=id_path,
        tokenizer_loader=lambda _tid: _WhitespaceTokenizer(),
    )
    assert len(results) == 1
    assert results[0].id_stats is not None
    assert results[0].en_stats is None
    assert results[0].token_ratio_id_over_en is None
    assert results[0].id_stats.total_tokens == 4
