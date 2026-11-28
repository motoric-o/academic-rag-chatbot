"""Citation quality scorer: precision, recall, and invalid citations.

Owner: Member B (see docs/project_spec.md §11.3).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from regqa.generation.parser import parse_output


@dataclass(frozen=True)
class CitationScore:
    """Per-prediction citation quality result."""

    qid: str
    cited_ids: list[int]
    valid_cited_ids: list[int]
    invalid_cited_ids: list[int]
    has_invalid_citation: bool
    precision: float | None
    recall: float | None
    cited_section_ids: list[str]
    gold_section_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_cited_ids(prediction: Mapping[str, Any]) -> list[int]:
    if prediction.get("error") is not None:
        return []
    parsed = prediction.get("parsed")
    if isinstance(parsed, Mapping) and "cited_ids" in parsed:
        return [int(cid) for cid in parsed["cited_ids"]]
    res = parse_output(prediction.get("raw_output"))
    return list(res.cited_ids)


def score_citations(
    prediction: Mapping[str, Any],
    eval_item: Mapping[str, Any],
    context_chunks: Sequence[Mapping[str, Any]],
    *,
    top_k_context: int = 4,
) -> CitationScore:
    """Score citations for a single prediction against fed context chunks (§11.3)."""
    qid = str(prediction.get("qid") or eval_item.get("id", ""))
    answerable = bool(eval_item.get("answerable", True))
    gold_sections = sorted(
        {str(span["section_id"]) for span in (eval_item.get("gold_spans") or [])}
    )
    gold_set = set(gold_sections)

    fed_chunks = list(context_chunks)[:top_k_context]
    cited_ids = _extract_cited_ids(prediction)

    valid_ids = [cid for cid in cited_ids if 1 <= cid <= len(fed_chunks)]
    invalid_ids = [cid for cid in cited_ids if cid < 1 or cid > len(fed_chunks)]

    cited_sections_set: set[str] = set()
    relevant_cited_passages = 0
    for cid in valid_ids:
        chunk_sections = {str(s) for s in (fed_chunks[cid - 1].get("section_ids") or [])}
        cited_sections_set |= chunk_sections
        if chunk_sections & gold_set:
            relevant_cited_passages += 1

    precision: float | None = (
        round(relevant_cited_passages / len(cited_ids), 4) if cited_ids else None
    )
    if answerable and gold_set:
        recall: float | None = round(len(cited_sections_set & gold_set) / len(gold_set), 4)
    else:
        recall = None

    return CitationScore(
        qid=qid,
        cited_ids=cited_ids,
        valid_cited_ids=valid_ids,
        invalid_cited_ids=invalid_ids,
        has_invalid_citation=bool(invalid_ids),
        precision=precision,
        recall=recall,
        cited_section_ids=sorted(cited_sections_set),
        gold_section_ids=gold_sections,
    )


def evaluate_citation_batch(
    predictions: Sequence[Mapping[str, Any]],
    eval_by_qid: Mapping[str, Mapping[str, Any]],
    contexts_by_qid: Mapping[str, Mapping[str, Any]],
    *,
    top_k_context: int = 4,
) -> dict[str, Any]:
    """Aggregate citation precision, recall, and invalid-citation rates across predictions."""
    scores: list[CitationScore] = []
    for pred in predictions:
        qid = str(pred.get("qid", ""))
        eval_item = eval_by_qid.get(qid)
        ctx_row = contexts_by_qid.get(qid)
        if eval_item is None or ctx_row is None:
            continue
        chunks = ctx_row.get("chunks") or []
        scores.append(
            score_citations(pred, eval_item, chunks, top_k_context=top_k_context)
        )

    prec_vals = [s.precision for s in scores if s.precision is not None]
    rec_vals = [s.recall for s in scores if s.recall is not None]
    n_eval = len(scores)
    invalid_preds = sum(s.has_invalid_citation for s in scores)
    total_invalid = sum(len(s.invalid_cited_ids) for s in scores)

    return {
        "n_evaluated": n_eval,
        "citation_precision": (
            round(sum(prec_vals) / len(prec_vals), 4) if prec_vals else None
        ),
        "citation_recall": (
            round(sum(rec_vals) / len(rec_vals), 4) if rec_vals else None
        ),
        "invalid_citation_rate": round(invalid_preds / n_eval, 4) if n_eval else 0.0,
        "total_invalid_citations": total_invalid,
        "details": [s.to_dict() for s in scores],
    }
