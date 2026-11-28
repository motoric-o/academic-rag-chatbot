"""Secondary LLM judge and human-agreement helpers.

Owner: Member D (spec §11.1).

The LLM judge uses a model that is NOT under test (e.g. Gemini Flash via
AI Studio free tier) to rate answers on a 0-1-2 scale.  Its scores are
compared against human ratings to compute inter-rater agreement (Cohen's
kappa).  The judge is *secondary* — human ratings are the gold standard.

This module also contains helpers for the human rating sheet and
inter-rater agreement computation.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------


@dataclass
class JudgeRating:
    """A single rating from the LLM judge."""

    qid: str
    model: str
    precision: str
    answer: str
    gold_answer: str
    rating: int  # 0, 1, or 2
    rationale: str = ""


@dataclass
class HumanRating:
    """A single human rating entry."""

    qid: str
    model: str
    precision: str
    rating: int  # 0, 1, or 2
    rater: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
#  LLM Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are a strict academic regulation QA evaluator. You will be given:
1. A QUESTION about academic regulations
2. The GOLD (correct) ANSWER
3. The MODEL'S ANSWER to evaluate

Rate the model's answer on this scale:
  0 = Incorrect or irrelevant — wrong information, contradicts gold, or misses the point
  1 = Partially correct — captures some facts but is incomplete or slightly inaccurate
  2 = Fully correct — captures all key facts from the gold answer accurately

Respond ONLY with valid JSON:
{"rating": <0|1|2>, "rationale": "<brief explanation>"}
"""

JUDGE_USER_TEMPLATE = """\
QUESTION: {question}

GOLD ANSWER: {gold_answer}

MODEL'S ANSWER: {model_answer}

Your JSON rating:"""


def build_judge_prompt(
    question: str, gold_answer: str, model_answer: str
) -> list[dict[str, str]]:
    """Build the chat-format messages for the LLM judge.

    Returns the messages list suitable for a chat-completion API call.
    """
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(
                question=question,
                gold_answer=gold_answer,
                model_answer=model_answer,
            ),
        },
    ]


def parse_judge_response(raw: str) -> dict[str, Any]:
    """Parse the JSON response from the judge model.

    Returns {"rating": int, "rationale": str} or raises ValueError.
    """
    # Try to extract JSON from the response
    raw = raw.strip()
    # Handle markdown code blocks
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Judge response is not valid JSON: {e}\nRaw: {raw}") from e

    rating = data.get("rating")
    if rating not in (0, 1, 2):
        raise ValueError(
            f"Judge rating must be 0, 1, or 2, got: {rating}\nRaw: {raw}"
        )

    return {
        "rating": int(rating),
        "rationale": str(data.get("rationale", "")),
    }


# ---------------------------------------------------------------------------
#  Cohen's Kappa
# ---------------------------------------------------------------------------


def cohens_kappa(ratings_a: list[int], ratings_b: list[int]) -> float:
    """Compute Cohen's kappa for two lists of ordinal ratings (0, 1, 2).

    Parameters
    ----------
    ratings_a, ratings_b : list[int]
        Paired ratings from two raters. Must be the same length.

    Returns
    -------
    float
        Cohen's kappa coefficient. 1.0 = perfect agreement, 0 = chance.

    Raises
    ------
    ValueError
        If the inputs have different lengths or are empty.
    """
    if len(ratings_a) != len(ratings_b):
        raise ValueError(
            f"Rating lists must have the same length: {len(ratings_a)} != {len(ratings_b)}"
        )
    if not ratings_a:
        raise ValueError("Rating lists must not be empty")

    n = len(ratings_a)
    categories = sorted(set(ratings_a) | set(ratings_b))

    # Build confusion matrix
    matrix: dict[tuple[int, int], int] = {}
    for cat_a in categories:
        for cat_b in categories:
            matrix[(cat_a, cat_b)] = 0
    for a, b in zip(ratings_a, ratings_b, strict=True):
        matrix[(a, b)] += 1

    # Observed agreement
    p_o = sum(matrix[(c, c)] for c in categories) / n

    # Expected agreement by chance
    p_e = 0.0
    for c in categories:
        row_sum = sum(matrix[(c, cb)] for cb in categories)
        col_sum = sum(matrix[(ca, c)] for ca in categories)
        p_e += (row_sum / n) * (col_sum / n)

    if p_e == 1.0:
        return 1.0  # Perfect agreement (avoid division by zero)

    return (p_o - p_e) / (1.0 - p_e)


# ---------------------------------------------------------------------------
#  Human rating sheet I/O
# ---------------------------------------------------------------------------


def create_rating_sheet_template(
    output_path: str | Path,
    predictions: list[dict],
) -> Path:
    """Create a CSV template for human rating.

    Parameters
    ----------
    output_path : str | Path
        Path to write the CSV template.
    predictions : list[dict]
        Each dict should have: qid, model, precision, answer_text (or answer).

    Returns
    -------
    Path
        The path to the created CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["qid", "model", "precision", "answer", "rating", "rater", "notes"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pred in predictions:
            writer.writerow(
                {
                    "qid": pred.get("qid", ""),
                    "model": pred.get("model", ""),
                    "precision": pred.get("precision", ""),
                    "answer": pred.get("answer_text", pred.get("answer", "")),
                    "rating": "",  # To be filled by human rater
                    "rater": "",
                    "notes": "",
                }
            )

    return output_path


def load_human_ratings(csv_path: str | Path) -> list[HumanRating]:
    """Load completed human ratings from a CSV file.

    Returns only rows where rating is non-empty.
    """
    ratings: list[HumanRating] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rating_str = row.get("rating", "").strip()
            if rating_str and rating_str in ("0", "1", "2"):
                ratings.append(
                    HumanRating(
                        qid=row.get("qid", ""),
                        model=row.get("model", ""),
                        precision=row.get("precision", ""),
                        rating=int(rating_str),
                        rater=row.get("rater", ""),
                        notes=row.get("notes", ""),
                    )
                )
    return ratings


def compute_agreement(
    ratings_file_a: str | Path,
    ratings_file_b: str | Path,
) -> dict[str, Any]:
    """Compute inter-rater agreement between two rating files.

    Matches ratings by (qid, model, precision) keys and computes Cohen's kappa.

    Returns
    -------
    dict
        {"kappa": float, "n_overlap": int, "agreement_pct": float}
    """
    ratings_a = {
        (r.qid, r.model, r.precision): r.rating
        for r in load_human_ratings(ratings_file_a)
    }
    ratings_b = {
        (r.qid, r.model, r.precision): r.rating
        for r in load_human_ratings(ratings_file_b)
    }

    overlap_keys = sorted(set(ratings_a.keys()) & set(ratings_b.keys()))
    if not overlap_keys:
        return {"kappa": 0.0, "n_overlap": 0, "agreement_pct": 0.0}

    list_a = [ratings_a[k] for k in overlap_keys]
    list_b = [ratings_b[k] for k in overlap_keys]

    exact_match = sum(1 for a, b in zip(list_a, list_b, strict=True) if a == b)

    return {
        "kappa": cohens_kappa(list_a, list_b),
        "n_overlap": len(overlap_keys),
        "agreement_pct": exact_match / len(overlap_keys) * 100,
    }
