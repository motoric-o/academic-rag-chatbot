"""CPU retrieval engine shared by calibration and final cache generation."""

from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path

from regqa.evaluation.validate_eval import EvalQuestion
from regqa.ingest.structure import Corpus
from regqa.retrieval.chunking import ChunkLimitError, make_chunks
from regqa.retrieval.hybrid import Ranking, hybrid_rankings
from regqa.retrieval.settings import Settings


class RetrievalEngine:
    def __init__(self, settings: Settings) -> None:
        cache = Path(".tmp/huggingface").resolve()
        os.environ.setdefault("HF_HOME", str(cache))
        os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
        os.environ.setdefault("HF_XET_CACHE", str(cache / "xet"))
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer

        self.settings = settings
        self.encoder = SentenceTransformer(
            settings.embedding_model,
            revision=settings.embedding_revision,
            device="cpu",
            model_kwargs={"use_safetensors": True},
        )
        budget_tokenizer = AutoTokenizer.from_pretrained(
            settings.budget_tokenizer, revision=settings.budget_tokenizer_revision
        )
        maximum = self.encoder.max_seq_length
        if budget_tokenizer is None or maximum is None or maximum < 1:
            raise ValueError("Model or budget tokenizer is missing token limits")
        self.maximum_chunk_tokens = maximum
        self.budget_tokenizer = budget_tokenizer
        self.versions = {
            name: version(name)
            for name in (
                "faiss-cpu",
                "rank-bm25",
                "sentence-transformers",
                "torch",
                "transformers",
                "tokenizers",
                "numpy",
            )
        }

    def count(self, text: str) -> int:
        return len(self.encoder.tokenizer.encode(text, truncation=False))

    def rank(self, corpus: Corpus, questions: list[EvalQuestion], size: int) -> list[Ranking]:
        maximum = self.maximum_chunk_tokens
        if size > maximum:
            raise ChunkLimitError(f"{size} tokens exceeds the embedding model limit of {maximum}")
        chunks = make_chunks(
            corpus["sections"],
            self.count,
            limit=size,
            overlap_ratio=self.settings.chunking.overlap_ratio,
            header_template=self.settings.chunking.header_template,
        )
        if len(chunks) < self.settings.retrieval.cache_k:
            raise ChunkLimitError("Corpus produced fewer chunks than the required cache size")
        queries = [q.question for q in questions]
        if any(self.count(query) > maximum for query in queries):
            raise ValueError("An eval question exceeds the embedding model token limit")
        return hybrid_rankings(self.encoder, chunks, queries, self.settings.retrieval.rrf_k)
