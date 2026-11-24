"""Tests for the faithfulness checker (Member D).

Uses placeholder mode (no NLI model loaded) to keep tests fast and CPU-only.
"""

import pytest
from regqa.evaluation.faithfulness import (
    split_into_claims,
    check_faithfulness,
    evaluate_faithfulness_batch,
)


class TestSplitIntoClaims:
    """Test sentence splitting for claim extraction."""

    def test_simple_sentences(self):
        text = "Students may take 2 semesters of leave. They must apply via portal."
        claims = split_into_claims(text)
        assert len(claims) == 2
        assert "Students may take 2 semesters" in claims[0]

    def test_strips_sources_trailer(self):
        text = "Academic leave is 2 semesters. Sources: [1], [3]"
        claims = split_into_claims(text)
        assert len(claims) == 1
        assert "Sources" not in claims[0]

    def test_not_found_returns_empty(self):
        assert split_into_claims("NOT_FOUND") == []

    def test_empty_string_returns_empty(self):
        assert split_into_claims("") == []

    def test_single_sentence(self):
        claims = split_into_claims("Only one sentence here.")
        assert len(claims) == 1


class TestCheckFaithfulness:
    """Test faithfulness checking in placeholder mode."""

    def test_no_claims_is_trivially_faithful(self):
        result = check_faithfulness("NOT_FOUND", ["some context"], use_model=False)
        assert result.faithfulness_score == 1.0
        assert result.hallucination_flag is False
        assert result.num_claims == 0

    def test_no_context_marks_all_unsupported(self):
        result = check_faithfulness(
            "Students may take 3 terms of leave.", [], use_model=False
        )
        assert result.num_claims == 1
        assert result.hallucination_flag is True

    def test_placeholder_mode_returns_result(self):
        result = check_faithfulness(
            "Leave is 2 semesters. Apply via portal.",
            ["Students may take academic leave."],
            use_model=False,
        )
        assert result.num_claims == 2
        # In placeholder mode, claims are marked unsupported
        assert isinstance(result.faithfulness_score, float)


class TestEvaluateBatch:
    """Test batch evaluation."""

    def test_batch_returns_aggregate(self):
        predictions = [
            {
                "qid": "q1",
                "answer": "Leave is 2 semesters.",
                "context_passages": ["Leave policy allows 2 semesters."],
            },
            {
                "qid": "q2",
                "answer": "NOT_FOUND",
                "context_passages": [],
            },
        ]
        result = evaluate_faithfulness_batch(predictions, use_model=False)
        assert result["num_predictions"] == 2
        assert "mean_faithfulness" in result
        assert "hallucination_rate" in result
        assert len(result["per_prediction"]) == 2

    def test_empty_batch(self):
        result = evaluate_faithfulness_batch([], use_model=False)
        assert result["num_predictions"] == 0
