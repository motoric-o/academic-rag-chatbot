"""Strict and soft key-fact answer correctness scorer + secondary token F1/EM.

Owner: Member B (see docs/project_spec.md §11.1, §13).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any
from unicodedata import normalize

from regqa.generation.parser import parse_output

_NUMBER_WORDS: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
}
_WORD_NUM_RE = re.compile(
    r"\b(" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")\b"
)
_TRAILING_ZERO_RE = re.compile(r"\b(\d+)\.0+\b")
_PERCENT_WORD_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:percent|per cent)\b")
_NON_ALNUM_RE = re.compile(r"[^\w\s.%-]+")


def normalize_for_match(text: str) -> str:
    """Normalize text for strict/soft key-fact matching with number normalization."""
    cleaned = normalize("NFKC", text).casefold()
    cleaned = _PERCENT_WORD_RE.sub(r"\1%", cleaned)
    cleaned = _WORD_NUM_RE.sub(lambda m: _NUMBER_WORDS[m.group(1)], cleaned)
    cleaned = _TRAILING_ZERO_RE.sub(r"\1", cleaned)
    cleaned = _NON_ALNUM_RE.sub(" ", cleaned)
    return " ".join(cleaned.split())


def compute_token_f1_em(prediction_text: str, gold_text: str) -> tuple[float, bool]:
    """Compute secondary token-level F1 and normalized Exact Match (§11.1)."""
    pred_norm = normalize_for_match(prediction_text)
    gold_norm = normalize_for_match(gold_text)
    em = pred_norm == gold_norm

    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    if not pred_tokens and not gold_tokens:
        return 1.0, em
    if not pred_tokens or not gold_tokens:
        return 0.0, em

    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if common == 0:
        return 0.0, em
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4), em


@dataclass(frozen=True)
class KeyFactScore:
    """Per-prediction key-fact correctness result."""

    qid: str
    answerable: bool
    strict_correct: bool
    soft_recall: float | None
    matched_facts: int
    total_facts: int
    format_ok: bool
    abstained: bool
    token_f1: float
    exact_match: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_parsed(prediction: Mapping[str, Any]) -> tuple[str, bool, bool]:
    if prediction.get("error") is not None:
        return "", False, False
    parsed = prediction.get("parsed")
    if isinstance(parsed, Mapping):
        return (
            str(parsed.get("answer_text", "")),
            bool(parsed.get("format_ok", False)),
            bool(parsed.get("abstained", False)),
        )
    res = parse_output(prediction.get("raw_output"))
    return res.answer_text, res.format_ok, res.abstained


def score_key_facts(
    prediction: Mapping[str, Any],
    eval_item: Mapping[str, Any],
) -> KeyFactScore:
    """Score a single prediction against an eval_set.jsonl item (§11.1)."""
    qid = str(prediction.get("qid") or eval_item.get("id", ""))
    answerable = bool(eval_item.get("answerable", True))
    key_facts: list[list[str]] = list(eval_item.get("key_facts") or [])
    gold_answer = str(eval_item.get("gold_answer", ""))

    answer_text, format_ok, abstained = _extract_parsed(prediction)
    norm_answer = normalize_for_match(answer_text)
    token_f1, exact_match = compute_token_f1_em(answer_text, gold_answer)

    if not answerable:
        strict_correct = bool(abstained and format_ok)
        return KeyFactScore(
            qid=qid,
            answerable=False,
            strict_correct=strict_correct,
            soft_recall=None,
            matched_facts=0,
            total_facts=0,
            format_ok=format_ok,
            abstained=abstained,
            token_f1=token_f1,
            exact_match=exact_match,
        )

    matched = 0
    for variants in key_facts:
        if any(normalize_for_match(v) in norm_answer for v in variants if v.strip()):
            matched += 1

    total = len(key_facts)
    soft_recall = round(matched / total, 4) if total > 0 else 1.0
    strict_correct = bool(format_ok and (not abstained) and (matched == total))

    return KeyFactScore(
        qid=qid,
        answerable=True,
        strict_correct=strict_correct,
        soft_recall=soft_recall,
        matched_facts=matched,
        total_facts=total,
        format_ok=format_ok,
        abstained=abstained,
        token_f1=token_f1,
        exact_match=exact_match,
    )


def evaluate_keyfact_batch(
    predictions: Sequence[Mapping[str, Any]],
    eval_by_qid: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Score a batch of predictions and return aggregate + per-question scores."""
    scores: list[KeyFactScore] = []
    per_qid_runs: dict[str, list[int]] = {}

    for pred in predictions:
        qid = str(pred.get("qid", ""))
        eval_item = eval_by_qid.get(qid)
        if eval_item is None:
            continue
        s = score_key_facts(pred, eval_item)
        scores.append(s)
        per_qid_runs.setdefault(qid, []).append(1 if s.strict_correct else 0)

    n_eval = len(scores)
    ans_scores = [s for s in scores if s.answerable]
    n_ans = len(ans_scores)

    strict_overall = round(sum(s.strict_correct for s in scores) / n_eval, 4) if n_eval else 0.0
    strict_answerable = (
        round(sum(s.strict_correct for s in ans_scores) / n_ans, 4) if n_ans else 0.0
    )
    soft_values = [s.soft_recall for s in ans_scores if s.soft_recall is not None]
    mean_soft = round(sum(soft_values) / len(soft_values), 4) if soft_values else 0.0
    format_valid_rate = round(sum(s.format_ok for s in scores) / n_eval, 4) if n_eval else 0.0
    mean_f1 = round(sum(s.token_f1 for s in scores) / n_eval, 4) if n_eval else 0.0
    mean_em = round(sum(s.exact_match for s in scores) / n_eval, 4) if n_eval else 0.0

    # Per-question mean strict correctness & majority binary score for pairwise stats
    per_question_mean = {
        qid: round(sum(vals) / len(vals), 4) for qid, vals in per_qid_runs.items()
    }
    per_question_binary = {
        qid: 1 if (sum(vals) / len(vals)) >= 0.5 else 0 for qid, vals in per_qid_runs.items()
    }

    return {
        "n_evaluated": n_eval,
        "n_unique_questions": len(per_qid_runs),
        "n_answerable": n_ans,
        "coverage": round(len(per_qid_runs) / len(eval_by_qid), 4) if eval_by_qid else 0.0,
        "strict_correctness": strict_overall,
        "strict_correctness_answerable": strict_answerable,
        "soft_keyfact_recall": mean_soft,
        "format_ok_rate": format_valid_rate,
        "mean_token_f1": mean_f1,
        "exact_match_rate": mean_em,
        "per_question_mean": per_question_mean,
        "per_question_binary": per_question_binary,
        "per_question_runs": per_qid_runs,
        "details": [s.to_dict() for s in scores],
    }
