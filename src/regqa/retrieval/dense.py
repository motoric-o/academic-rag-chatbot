"""Exact cosine similarity using a CPU FAISS inner-product index."""

from typing import TYPE_CHECKING

import faiss
import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def dense_ranks(
    model: "SentenceTransformer", texts: list[str], queries: list[str]
) -> list[list[int]]:
    documents = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    embeddings = model.encode(queries, normalize_embeddings=True, convert_to_numpy=True)
    documents = np.ascontiguousarray(documents, dtype=np.float32)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if not np.isfinite(documents).all() or not np.isfinite(embeddings).all():
        raise ValueError("Dense embedding returned a non-finite value")
    index = faiss.IndexFlatIP(documents.shape[1])
    index.add(documents)
    scores, ids = index.search(embeddings, index.ntotal)
    return [
        ranked[np.lexsort((ranked, -values))].tolist()
        for values, ranked in zip(scores, ids, strict=True)
    ]
