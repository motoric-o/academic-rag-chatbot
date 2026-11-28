"""Fuse complete BM25 and FAISS rankings with reciprocal rank fusion."""

from typing import TYPE_CHECKING, NamedTuple

from regqa.retrieval.bm25 import bm25_ranks
from regqa.retrieval.chunking import Chunk
from regqa.retrieval.dense import dense_ranks

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class Ranking(NamedTuple):
    chunks: list[Chunk]
    scores: list[float]


def hybrid_rankings(
    model: "SentenceTransformer", chunks: list[Chunk], questions: list[str], rrf_k: int
) -> list[Ranking]:
    texts = [chunk["text"] for chunk in chunks]
    sparse = bm25_ranks(texts, questions)
    dense = dense_ranks(model, texts, questions)
    results = []
    for first, second in zip(sparse, dense, strict=True):
        scores = [0.0] * len(chunks)
        for order in (first, second):
            for rank, index in enumerate(order, 1):
                scores[index] += 1 / (rrf_k + rank)
        ranked = sorted(range(len(chunks)), key=lambda index: (-scores[index], index))
        results.append(Ranking([chunks[i] for i in ranked], [scores[i] for i in ranked]))
    return results
