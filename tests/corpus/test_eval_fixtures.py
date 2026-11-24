"""Check the Week 1 fixtures against the eval contract."""

import json
from pathlib import Path

import pytest

from regqa.evaluation.validate_eval import FIXTURE, validate_eval

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_fixtures_have_resolvable_gold_facts() -> None:
    corpus = json.loads((FIXTURES / "mini_corpus.json").read_text(encoding="utf-8"))
    texts = {section["section_id"]: section["text"] for section in corpus["sections"]}
    questions = validate_eval(FIXTURES / "mini_eval.jsonl", section_ids=set(texts), synthetic=True)

    assert FIXTURE in corpus["notes"]
    assert len(questions) == 10
    assert sum(not question.answerable for question in questions) == 2
    assert any(len(question.gold_spans) > 1 for question in questions)
    for question in questions:
        evidence = " ".join(texts[span.section_id] for span in question.gold_spans).casefold()
        for fact in question.key_facts:
            assert any(variant.casefold() in evidence for variant in fact)


def test_fixtures_require_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match=FIXTURE):
        validate_eval(FIXTURES / "mini_eval.jsonl")
    with pytest.raises(ValueError, match="unknown section ID"):
        validate_eval(FIXTURES / "mini_eval.jsonl", section_ids={"99"}, synthetic=True)
