"""Abstention metrics and over-abstention split by evidence_in_context.

Owner: Member B (see docs/project_spec.md §11.4).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from regqa.generation.parser import parse_output


@dataclass(frozen=True)
class AbstentionSummary:
    """Abstention confusion matrix and evidence-sufficiency breakdown (§11.4)."""

    n_evaluated: int
    tp: int  # unanswerable & abstained
    fn: int  # unanswerable & answered (false answer)
    fp: int  # answerable & abstained (over-abstention)
    tn: int  # answerable & answered
    precision: float | None
    recall: float | None
    f1: float | None
    false_answer_rate: float | None
    over_abstention_rate: float | None
    n_answerable_with_evidence: int
    over_abstention_with_evidence: float | None
    n_answerable_without_evidence: int
    over_abstention_without_evidence: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_abstained(prediction: Mapping[str, Any]) -> bool:
    if prediction.get("error") is not None:
        return False
    parsed = prediction.get("parsed")
    if isinstance(parsed, Mapping) and "abstained" in parsed:
        return bool(parsed["abstained"])
    return bool(parse_output(prediction.get("raw_output")).abstained)


def evaluate_abstention_batch(
    predictions: Sequence[Mapping[str, Any]],
    eval_by_qid: Mapping[str, Mapping[str, Any]],
    contexts_by_qid: Mapping[str, Mapping[str, Any]] | None = None,
) -> AbstentionSummary:
    """Compute abstention confusion matrix, rates, and evidence_in_context split (§11.4)."""
    tp = fn = fp = tn = 0
    with_ev_total = with_ev_abstained = 0
    without_ev_total = without_ev_abstained = 0

    for pred in predictions:
        qid = str(pred.get("qid", ""))
        eval_item = eval_by_qid.get(qid)
        if eval_item is None:
            continue

        answerable = bool(eval_item.get("answerable", True))
        abstained = _is_abstained(pred)

        if not answerable:
            if abstained:
                tp += 1
            else:
                fn += 1
        else:
            if abstained:
                fp += 1
            else:
                tn += 1

            if contexts_by_qid and qid in contexts_by_qid:
                ev = contexts_by_qid[qid].get("evidence_in_context")
                if ev is True:
                    with_ev_total += 1
                    if abstained:
                        with_ev_abstained += 1
                elif ev is False:
                    without_ev_total += 1
                    if abstained:
                        without_ev_abstained += 1

    n_eval = tp + fn + fp + tn
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else None
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1: float | None = round(2 * precision * recall / (precision + recall), 4)
    else:
        f1 = 0.0 if (precision is not None and recall is not None) else None

    false_answer_rate = round(fn / (tp + fn), 4) if (tp + fn) > 0 else None
    over_abstention_rate = round(fp / (fp + tn), 4) if (fp + tn) > 0 else None
    over_with_ev = (
        round(with_ev_abstained / with_ev_total, 4) if with_ev_total > 0 else None
    )
    over_without_ev = (
        round(without_ev_abstained / without_ev_total, 4) if without_ev_total > 0 else None
    )

    return AbstentionSummary(
        n_evaluated=n_eval,
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        false_answer_rate=false_answer_rate,
        over_abstention_rate=over_abstention_rate,
        n_answerable_with_evidence=with_ev_total,
        over_abstention_with_evidence=over_with_ev,
        n_answerable_without_evidence=without_ev_total,
        over_abstention_without_evidence=over_without_ev,
    )
