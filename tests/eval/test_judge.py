"""Tests for the LLM judge and human-rating helpers (Member D).

Tests the judge prompt builder, response parser, Cohen's kappa,
and rating sheet I/O — all without an actual LLM call.
"""

import csv
import json
import tempfile
from pathlib import Path

import pytest
from regqa.evaluation.judge import (
    build_judge_prompt,
    parse_judge_response,
    cohens_kappa,
    create_rating_sheet_template,
    load_human_ratings,
    compute_agreement,
)


class TestBuildJudgePrompt:
    """Test judge prompt construction."""

    def test_returns_system_and_user_messages(self):
        msgs = build_judge_prompt("Q?", "Gold A.", "Model A.")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_user_message_contains_question_and_answers(self):
        msgs = build_judge_prompt("What is X?", "X is 42.", "X is 43.")
        user_content = msgs[1]["content"]
        assert "What is X?" in user_content
        assert "X is 42." in user_content
        assert "X is 43." in user_content


class TestParseJudgeResponse:
    """Test parsing of judge LLM output."""

    def test_valid_json(self):
        raw = '{"rating": 2, "rationale": "Fully correct."}'
        result = parse_judge_response(raw)
        assert result["rating"] == 2
        assert result["rationale"] == "Fully correct."

    def test_markdown_code_block(self):
        raw = '```json\n{"rating": 1, "rationale": "Partial."}\n```'
        result = parse_judge_response(raw)
        assert result["rating"] == 1

    def test_invalid_rating_raises(self):
        raw = '{"rating": 5, "rationale": "Bad."}'
        with pytest.raises(ValueError):
            parse_judge_response(raw)

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            parse_judge_response("not json")


class TestCohensKappa:
    """Test Cohen's kappa computation."""

    def test_perfect_agreement(self):
        a = [0, 1, 2, 0, 1]
        b = [0, 1, 2, 0, 1]
        assert cohens_kappa(a, b) == 1.0

    def test_complete_disagreement_is_low(self):
        a = [0, 0, 2, 2]
        b = [2, 2, 0, 0]
        kappa = cohens_kappa(a, b)
        assert kappa < 0.0  # worse than chance (systematic disagreement)

    def test_partial_agreement(self):
        a = [0, 1, 2, 0, 1, 2]
        b = [0, 1, 1, 0, 2, 2]
        kappa = cohens_kappa(a, b)
        assert 0.0 < kappa < 1.0

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError):
            cohens_kappa([1, 2], [1])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            cohens_kappa([], [])


class TestRatingSheetIO:
    """Test human rating sheet template creation and loading."""

    def test_create_and_load_template(self):
        predictions = [
            {"qid": "q1", "model": "qwen3-4b", "precision": "4-bit", "answer_text": "Answer 1."},
            {"qid": "q2", "model": "qwen3-8b", "precision": "8-bit", "answer_text": "Answer 2."},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ratings.csv"
            create_rating_sheet_template(path, predictions)

            # Verify file exists and has correct structure
            assert path.exists()
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["qid"] == "q1"
            assert rows[0]["answer"] == "Answer 1."
            assert rows[0]["rating"] == ""  # empty, to be filled by human

    def test_load_filled_ratings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "filled.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["qid", "model", "precision", "answer", "rating", "rater", "notes"]
                )
                writer.writeheader()
                writer.writerow({
                    "qid": "q1", "model": "m1", "precision": "4-bit",
                    "answer": "A", "rating": "2", "rater": "D", "notes": "",
                })
                writer.writerow({
                    "qid": "q2", "model": "m1", "precision": "4-bit",
                    "answer": "B", "rating": "", "rater": "", "notes": "",
                })

            ratings = load_human_ratings(path)
            assert len(ratings) == 1  # only q1 has a rating
            assert ratings[0].rating == 2


class TestComputeAgreement:
    """Test inter-rater agreement computation."""

    def test_agreement_between_two_raters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Rater A
            path_a = Path(tmpdir) / "rater_a.csv"
            with open(path_a, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["qid", "model", "precision", "answer", "rating", "rater", "notes"])
                w.writeheader()
                w.writerow({"qid": "q1", "model": "m1", "precision": "4-bit", "answer": "A", "rating": "2", "rater": "A", "notes": ""})
                w.writerow({"qid": "q2", "model": "m1", "precision": "4-bit", "answer": "B", "rating": "1", "rater": "A", "notes": ""})

            # Rater B
            path_b = Path(tmpdir) / "rater_b.csv"
            with open(path_b, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["qid", "model", "precision", "answer", "rating", "rater", "notes"])
                w.writeheader()
                w.writerow({"qid": "q1", "model": "m1", "precision": "4-bit", "answer": "A", "rating": "2", "rater": "B", "notes": ""})
                w.writerow({"qid": "q2", "model": "m1", "precision": "4-bit", "answer": "B", "rating": "1", "rater": "B", "notes": ""})

            result = compute_agreement(path_a, path_b)
            assert result["kappa"] == 1.0
            assert result["n_overlap"] == 2
            assert result["agreement_pct"] == 100.0
