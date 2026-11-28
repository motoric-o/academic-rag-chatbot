"""Recall and reciprocal rank using frozen gold section and paragraph references."""

from statistics import mean

from regqa.evaluation.validate_eval import EvalQuestion
from regqa.retrieval.chunking import Chunk

RECALL_AT = (1, 3, 4, 5)
MRR_AT = 10
Gold = set[tuple[str, str | None]]


def gold_ids(question: EvalQuestion) -> Gold:
    return {(span.section_id, span.paragraph) for span in question.gold_spans}


def found(chunks: list[Chunk], gold: Gold) -> Gold:
    return {
        (section_id, paragraph)
        for chunk in chunks
        for section_id, paragraph in gold
        if section_id in chunk["section_ids"]
        and (
            paragraph is None or "paragraph_ids" not in chunk or paragraph in chunk["paragraph_ids"]
        )
    }


def recall_at(chunks: list[Chunk], gold: Gold, k: int) -> float:
    return len(found(chunks[:k], gold)) / len(gold)


def retrieval_metrics(
    rows: list[tuple[EvalQuestion, list[Chunk]]],
) -> dict[str, int | float | None]:
    targets = [(gold_ids(question), chunks) for question, chunks in rows if question.answerable]
    metrics: dict[str, int | float | None] = {
        "questions": len(rows),
        "answerable": len(targets),
    }
    for k in RECALL_AT:
        metrics[f"recall@{k}"] = (
            mean(recall_at(chunks, gold, k) for gold, chunks in targets) if targets else None
        )
    metrics[f"mrr@{MRR_AT}"] = (
        mean(
            next(
                (1 / rank for rank, chunk in enumerate(chunks[:MRR_AT], 1) if found([chunk], gold)),
                0.0,
            )
            for gold, chunks in targets
        )
        if targets
        else None
    )
    return metrics
