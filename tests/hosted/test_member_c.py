"""Unit tests for Member C modules (parser, prompt_builder, base adapter, api).

Owner: Member C (Francisco Valentino; see docs/project_spec.md §0.2 & §13).
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from regqa.api.routes import (
    ask_question,
    compare_models,
    get_health,
    get_models,
    get_section,
)
from regqa.api.schemas import AskRequest, CompareRequest
from regqa.generation.parser import parse_output
from regqa.generation.prompt_builder import PromptBuilder
from regqa.models.base import GenerationConfig, MockAdapter


# -----------------------------------------------------------------------------
# Parser Tests (§8)
# -----------------------------------------------------------------------------
def test_parser_exact_not_found() -> None:
    res = parse_output("NOT_FOUND")
    assert res.abstained is True
    assert res.format_ok is True
    assert res.answer_text == "NOT_FOUND"
    assert res.cited_ids == []
    assert res.error_reason is None


def test_parser_not_found_with_period_and_whitespace() -> None:
    res = parse_output("  NOT_FOUND.  \n")
    assert res.abstained is True
    assert res.format_ok is True
    assert res.cited_ids == []


def test_parser_valid_answer_with_citations() -> None:
    raw = (
        "Students may take academic leave for up to two consecutive semesters.\n\n"
        "Sources: [1], [3]"
    )
    res = parse_output(raw)
    assert res.abstained is False
    assert res.format_ok is True
    assert res.cited_ids == [1, 3]
    assert "Students may take academic leave" in res.answer_text
    assert "Sources:" not in res.answer_text


def test_parser_citations_deduplication() -> None:
    raw = "The maximum study load is 24 SKS.\nSources: [2], [2], [1]"
    res = parse_output(raw)
    assert res.format_ok is True
    assert res.cited_ids == [2, 1]


def test_parser_missing_sources_line() -> None:
    raw = "Students may take academic leave for two semesters."
    res = parse_output(raw)
    assert res.abstained is False
    assert res.format_ok is False
    assert "Missing 'Sources: [n]'" in (res.error_reason or "")


def test_parser_empty_answer_before_citation() -> None:
    raw = "Sources: [1]"
    res = parse_output(raw)
    assert res.format_ok is False
    assert "Empty answer text" in (res.error_reason or "")


def test_parser_sources_with_no_digits() -> None:
    raw = "Some answer text.\nSources: []"
    res = parse_output(raw)
    assert res.format_ok is False
    assert "contains no passage numbers" in (res.error_reason or "")


# -----------------------------------------------------------------------------
# PromptBuilder Tests (§8)
# -----------------------------------------------------------------------------
def test_prompt_builder_format_passages() -> None:
    builder = PromptBuilder()
    chunks = [
        {"text": "Text of passage one.", "header": "[Chapter 1 > Article 1]"},
        {"text": "Text of passage two.", "header": "[Chapter 1 > Article 2]"},
    ]
    formatted = builder.format_passages(chunks)
    assert "[1] [Chapter 1 > Article 1]" in formatted
    assert "[2] [Chapter 1 > Article 2]" in formatted
    assert "Text of passage one." in formatted


def test_prompt_builder_rag_messages_and_hash() -> None:
    builder = PromptBuilder()
    chunks = [{"text": "Passage text"}]
    messages = builder.build_rag_messages("What is the limit?", chunks)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "What is the limit?" in messages[1]["content"]

    h1 = PromptBuilder.compute_prompt_hash(messages)
    h2 = PromptBuilder.compute_prompt_hash(messages)
    assert len(h1) == 64
    assert h1 == h2


def test_prompt_builder_closed_book() -> None:
    builder = PromptBuilder()
    messages = builder.build_closed_book_messages("Question text?")
    assert len(messages) == 2
    assert "Question text?" in messages[1]["content"]


# -----------------------------------------------------------------------------
# MockAdapter Tests (§7)
# -----------------------------------------------------------------------------
def test_mock_adapter_generation() -> None:
    adapter = MockAdapter(model="test-mock", canned_response="NOT_FOUND")
    config = GenerationConfig(temperature=0.0)
    messages = [{"role": "user", "content": "Hello"}]
    res = adapter.generate(messages, config)

    assert res.raw_output == "NOT_FOUND"
    assert res.model == "test-mock"
    assert res.latency_s > 0
    assert res.hardware.get("mock") is True


# -----------------------------------------------------------------------------
# FastAPI Endpoint Unit Tests (§12)
# -----------------------------------------------------------------------------
def test_api_health() -> None:
    res = get_health()
    assert res.status == "ok"
    assert res.mode in ("replay", "live-hosted", "live-local")


def test_api_models() -> None:
    res = get_models()
    models = res.models
    assert len(models) >= 4
    model_ids = [m.id for m in models]
    assert "qwen3-0.6b" in model_ids
    assert "openai/gpt-oss-20b" in model_ids


def test_api_ask() -> None:
    req = AskRequest(
        question="How many semesters of academic leave can I take?",
        model_id="qwen3-0.6b",
        precision="4-bit",
        mode="replay",
    )
    res = ask_question(req)
    assert res.question == req.question
    assert res.format_ok is True
    assert len(res.answer_text) > 0


def test_api_compare() -> None:
    req = CompareRequest(
        question="Can I take academic leave?",
        model_ids=["qwen3-0.6b", "openai/gpt-oss-20b"],
        precision="4-bit",
        mode="replay",
    )
    res = compare_models(req)
    assert len(res.results) == 2
    assert res.results[0].model == "qwen3-0.6b"
    assert res.results[1].model == "openai/gpt-oss-20b"


def test_api_get_section_found() -> None:
    res = get_section("pasal_12")
    assert res.section_id == "pasal_12"
    assert "Academic Leave" in res.title


def test_api_get_section_not_found() -> None:
    with pytest.raises(HTTPException) as excinfo:
        get_section("pasal_non_existent")
    assert excinfo.value.status_code == 404
