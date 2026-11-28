"""Tests for error taxonomy analysis (Member D)."""

from regqa.analysis.error_taxonomy import (
    classify_errors,
    compute_error_taxonomy,
)


class TestClassifyErrors:
    """Test per-prediction error classification."""

    def test_format_failure(self):
        pred = {
            "qid": "q1",
            "model": "m1",
            "precision": "4-bit",
            "format_ok": False,
            "answer_text": "...",
            "abstained": False,
            "cited_ids": [],
        }
        item = {"answerable": True, "key_facts": []}
        ann = classify_errors(pred, item)
        assert "format_failure" in ann.error_tags

    def test_false_answer_on_unanswerable(self):
        pred = {
            "qid": "q1",
            "model": "m1",
            "precision": "4-bit",
            "format_ok": True,
            "answer_text": "Some answer",
            "abstained": False,
            "cited_ids": [],
        }
        item = {"answerable": False, "key_facts": []}
        ann = classify_errors(pred, item)
        assert "false_answer" in ann.error_tags

    def test_over_abstention(self):
        pred = {
            "qid": "q1",
            "model": "m1",
            "precision": "4-bit",
            "format_ok": True,
            "answer_text": "NOT_FOUND",
            "abstained": True,
            "cited_ids": [],
        }
        item = {"answerable": True, "key_facts": [["some fact"]]}
        ann = classify_errors(pred, item)
        assert "over_abstention" in ann.error_tags

    def test_missing_key_fact(self):
        pred = {
            "qid": "q1",
            "model": "m1",
            "precision": "4-bit",
            "format_ok": True,
            "answer_text": "Completely wrong answer",
            "abstained": False,
            "cited_ids": [],
        }
        item = {"answerable": True, "key_facts": [["84 credits"], ["April 15"]]}
        ann = classify_errors(pred, item)
        assert "missing_key_fact" in ann.error_tags

    def test_partial_answer(self):
        pred = {
            "qid": "q1",
            "model": "m1",
            "precision": "4-bit",
            "format_ok": True,
            "answer_text": "Need 84 credits to register",
            "abstained": False,
            "cited_ids": [],
        }
        item = {"answerable": True, "key_facts": [["84 credits"], ["April 15"]]}
        ann = classify_errors(pred, item)
        assert "partial_answer" in ann.error_tags

    def test_hallucination_from_faithfulness(self):
        pred = {
            "qid": "q1",
            "model": "m1",
            "precision": "4-bit",
            "format_ok": True,
            "answer_text": "...",
            "abstained": False,
            "cited_ids": [],
        }
        item = {"answerable": True, "key_facts": []}
        faith = {"hallucination_flag": True}
        ann = classify_errors(pred, item, faith)
        assert "hallucination" in ann.error_tags

    def test_clean_prediction_has_no_errors(self):
        pred = {
            "qid": "q1",
            "model": "m1",
            "precision": "4-bit",
            "format_ok": True,
            "answer_text": "Need 84 credits, deadline April 15.",
            "abstained": False,
            "cited_ids": [1],
        }
        item = {"answerable": True, "key_facts": [["84 credits"], ["April 15"]]}
        ann = classify_errors(pred, item)
        assert ann.error_tags == []


class TestComputeErrorTaxonomy:
    """Test aggregate taxonomy computation."""

    def test_aggregate_counts(self):
        from regqa.analysis.error_taxonomy import ErrorAnnotation

        anns = [
            ErrorAnnotation(
                qid="q1",
                model="m1",
                precision="4-bit",
                error_tags=["hallucination", "missing_key_fact"],
            ),
            ErrorAnnotation(qid="q2", model="m1", precision="4-bit", error_tags=["format_failure"]),
            ErrorAnnotation(qid="q3", model="m1", precision="4-bit", error_tags=[]),  # clean
        ]
        result = compute_error_taxonomy(anns)
        assert result["total_predictions"] == 3
        assert result["total_with_errors"] == 2
        assert result["category_counts"]["hallucination"] == 1
        assert result["category_counts"]["format_failure"] == 1
        assert result["category_counts"]["missing_key_fact"] == 1

    def test_empty_list(self):
        result = compute_error_taxonomy([])
        assert result["total_predictions"] == 0
        assert result["error_rate"] == 0.0
