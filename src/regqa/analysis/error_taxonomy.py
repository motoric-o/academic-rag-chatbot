"""Error taxonomy analysis utilities.

Owner: Member D (spec §0.2 — E5 reliability breakdown).

Classifies model errors into a fixed taxonomy for the reliability analysis
(Experiment E5).  Each prediction is assigned zero or more error tags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
#  Error taxonomy (spec §11, E5)
# ---------------------------------------------------------------------------

ERROR_CATEGORIES = {
    "hallucination": "Answer contains facts not in the provided context",
    "missing_key_fact": "Answer is missing one or more required key facts",
    "wrong_citation": "Citation references a passage that does not support the claim",
    "invalid_citation": "Citation references a passage number not in the context",
    "format_failure": "Output does not follow the required format contract",
    "over_abstention": "Model said NOT_FOUND but the answer was in the context",
    "false_answer": "Model gave an answer to an unanswerable question",
    "partial_answer": "Answer is partially correct but incomplete",
    "contradiction": "Answer contradicts the gold answer or the source text",
}


@dataclass
class ErrorAnnotation:
    """Error annotation for a single prediction."""

    qid: str
    model: str
    precision: str
    error_tags: list[str] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
#  Automatic error classification
# ---------------------------------------------------------------------------


def classify_errors(
    prediction: dict,
    eval_item: dict,
    faithfulness_result: dict | None = None,
) -> ErrorAnnotation:
    """Classify errors for a single prediction against its eval item.

    Parameters
    ----------
    prediction : dict
        Must have: qid, model, precision, parsed (with answer_text, cited_ids,
        abstained, format_ok).
    eval_item : dict
        From eval_set.jsonl: answerable, key_facts, gold_spans, type.
    faithfulness_result : dict | None
        Output from faithfulness checker (optional).

    Returns
    -------
    ErrorAnnotation
        The prediction with its error tags.
    """
    tags: list[str] = []
    parsed = prediction.get("parsed", prediction)
    answer_text = parsed.get("answer_text", prediction.get("answer_text", ""))
    abstained = parsed.get("abstained", prediction.get("abstained", False))
    format_ok = parsed.get("format_ok", prediction.get("format_ok", True))
    cited_ids = parsed.get("cited_ids", prediction.get("cited_ids", []))

    answerable = eval_item.get("answerable", True)
    key_facts = eval_item.get("key_facts", [])

    # 1. Format failure
    if not format_ok:
        tags.append("format_failure")

    # 2. False answer (answered an unanswerable question)
    if not answerable and not abstained and answer_text and answer_text != "NOT_FOUND":
        tags.append("false_answer")

    # 3. Over-abstention (abstained on an answerable question)
    if answerable and abstained:
        tags.append("over_abstention")

    # 4. Missing key facts
    if answerable and not abstained and key_facts:
        answer_lower = answer_text.lower()
        missing = 0
        for fact_variants in key_facts:
            if not any(v.lower() in answer_lower for v in fact_variants):
                missing += 1
        if missing > 0 and missing == len(key_facts):
            tags.append("missing_key_fact")
        elif missing > 0:
            tags.append("partial_answer")

    # 5. Invalid citations (passage numbers not in context)
    # This requires knowing the number of context passages provided.
    # For now, we flag citations > 5 (top-5 cached) as invalid.
    max_valid_passage = prediction.get("top_k", 5)
    for cid in cited_ids:
        if isinstance(cid, int) and (cid < 1 or cid > max_valid_passage):
            tags.append("invalid_citation")
            break

    # 6. Hallucination (from faithfulness checker)
    if faithfulness_result and faithfulness_result.get("hallucination_flag", False):
        tags.append("hallucination")

    return ErrorAnnotation(
        qid=prediction.get("qid", ""),
        model=prediction.get("model", ""),
        precision=prediction.get("precision", ""),
        error_tags=tags,
    )


# ---------------------------------------------------------------------------
#  Aggregate error taxonomy
# ---------------------------------------------------------------------------


def compute_error_taxonomy(annotations: list[ErrorAnnotation]) -> dict[str, Any]:
    """Aggregate error annotations into taxonomy counts and rates.

    Returns
    -------
    dict
        {
            "total_predictions": int,
            "total_with_errors": int,
            "error_rate": float,
            "category_counts": {category: count},
            "category_rates": {category: rate},
            "by_model": {model: {category: count}},
        }
    """
    total = len(annotations)
    if total == 0:
        return {
            "total_predictions": 0,
            "total_with_errors": 0,
            "error_rate": 0.0,
            "category_counts": {},
            "category_rates": {},
            "by_model": {},
        }

    category_counts: dict[str, int] = {cat: 0 for cat in ERROR_CATEGORIES}
    by_model: dict[str, dict[str, int]] = {}
    total_with_errors = 0

    for ann in annotations:
        if ann.error_tags:
            total_with_errors += 1

        model_key = f"{ann.model}_{ann.precision}"
        if model_key not in by_model:
            by_model[model_key] = {cat: 0 for cat in ERROR_CATEGORIES}

        for tag in ann.error_tags:
            if tag in category_counts:
                category_counts[tag] += 1
                by_model[model_key][tag] += 1

    category_rates = {cat: count / total for cat, count in category_counts.items()}

    return {
        "total_predictions": total,
        "total_with_errors": total_with_errors,
        "error_rate": total_with_errors / total,
        "category_counts": category_counts,
        "category_rates": category_rates,
        "by_model": by_model,
    }
