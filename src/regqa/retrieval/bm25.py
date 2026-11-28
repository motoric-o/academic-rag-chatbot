"""BM25 ranks with deterministic tie order."""

import re

import numpy as np
from rank_bm25 import BM25Okapi

WORD = re.compile(r"\w+")


def bm25_ranks(texts: list[str], queries: list[str]) -> list[list[int]]:
    index = BM25Okapi([WORD.findall(text.casefold()) for text in texts])
    return [
        np.argsort(-index.get_scores(WORD.findall(query.casefold())), kind="stable").tolist()
        for query in queries
    ]
