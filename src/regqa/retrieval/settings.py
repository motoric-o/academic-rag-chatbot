"""Load the frozen inputs and shared configuration for retrieval."""

import argparse
import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from regqa.evaluation.validate_eval import (
    DEV_COUNT,
    TEST_COUNT,
    EvalQuestion,
    split_sha256,
    validate_eval,
)
from regqa.ingest.artifacts import read_corpus, sha256
from regqa.ingest.structure import Corpus


class Chunking(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidates: list[int] = Field(min_length=1)
    max_chunk_tokens: int | None
    overlap_ratio: float = Field(ge=0, lt=1)
    header_template: str


class Retrieval(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["hybrid_rrf"]
    rrf_k: int = Field(gt=0)
    faiss_index: Literal["IndexFlatIP"]
    normalize_embeddings: Literal[True]
    cache_k: Literal[5]
    top_k_context: Literal[4]
    prompt_token_budget: int = Field(gt=0)


class Calibration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bootstrap_samples: int = Field(gt=0)
    seed: int = Field(ge=0)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    embedding_model: str = Field(min_length=1)
    embedding_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    chunk_tokenizer: str
    budget_tokenizer: str = Field(min_length=1)
    budget_tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    chunking: Chunking
    retrieval: Retrieval
    calibration: Calibration

    @model_validator(mode="after")
    def validate_grid(self) -> "Settings":
        sizes = self.chunking.candidates
        if self.chunk_tokenizer != self.embedding_model:
            raise ValueError("Chunk tokenizer must match the pinned embedding model")
        if sorted(set(sizes)) != sizes or min(sizes) <= 0:
            raise ValueError("Chunk candidates must be increasing unique positive integers")
        chosen = self.chunking.max_chunk_tokens
        if chosen is not None and chosen not in sizes:
            raise ValueError("Frozen chunk size must be one of the candidates")
        header = self.chunking.header_template.format(chapter="1", article="1.1")
        if not re.fullmatch(r"\[.+\]", header):
            raise ValueError("Chunk header template must produce a bracketed header")
        return self

    def fingerprint(self) -> str:
        values = self.model_dump()
        values["chunking"]["max_chunk_tokens"] = None
        return sha256(json.dumps(values, sort_keys=True).encode("utf-8"))


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/retrieval.yaml"))
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--eval", type=Path)
    parser.add_argument("--prompt-template", type=Path, default=Path("configs/prompts/rag_v1.yaml"))
    parser.add_argument("--synthetic", action="store_true", help="Run the labeled Week 1 fixtures")


def load_inputs(
    args: argparse.Namespace, *, dev_only: bool
) -> tuple[Corpus, list[EvalQuestion], dict[str, str]]:
    corpus_path = args.corpus or Path(
        "tests/fixtures/mini_corpus.json" if args.synthetic else "data/source_en/corpus.json"
    )
    eval_path = args.eval or Path(
        "tests/fixtures/mini_eval.jsonl" if args.synthetic else "data/eval/eval_set.jsonl"
    )
    content = corpus_path.read_bytes()
    digest = sha256(content)
    corpus = read_corpus(content, language="en", synthetic=args.synthetic)
    if not args.synthetic:
        expected = corpus_path.with_suffix(".sha256").read_text(encoding="ascii").strip()
        if digest != expected:
            raise ValueError(f"Frozen corpus checksum mismatch: {corpus_path}")
    questions = validate_eval(
        eval_path,
        section_ids={s["section_id"] for s in corpus["sections"]},
        synthetic=args.synthetic,
        split="dev" if dev_only else None,
    )
    dev = [q for q in questions if q.split == "dev"]
    test = [q for q in questions if q.split == "test"]
    if not any(q.answerable for q in dev):
        raise ValueError("Dev split must contain answerable questions")
    if not args.synthetic and len(dev) != DEV_COUNT:
        raise ValueError(f"Expected {DEV_COUNT} dev questions, found {len(dev)}")
    if not args.synthetic and test and len(test) != TEST_COUNT:
        raise ValueError(f"Expected {TEST_COUNT} locked test questions, found {len(test)}")
    if not args.synthetic and test:
        expected = eval_path.with_suffix(".test.sha256").read_text(encoding="ascii").strip()
        if split_sha256(test) != expected:
            raise ValueError(f"Locked test checksum mismatch: {eval_path}")
    return (
        corpus,
        questions,
        {
            "corpus_sha256": digest,
            "dev_sha256": split_sha256(dev),
            "prompt_sha256": sha256(args.prompt_template.read_bytes()),
        },
    )


def load_settings(path: Path) -> Settings:
    return Settings.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def require_synthetic_output(path: Path) -> None:
    if not path.resolve().is_relative_to(Path(".tmp").resolve()):
        raise ValueError("Synthetic output must stay inside .tmp/")
