"""Build frozen retrieval and development oracle caches with the dev-selected configuration."""

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from regqa.evaluation.retrieval_metrics import Gold, found, gold_ids, retrieval_metrics
from regqa.evaluation.validate_eval import FIXTURE, EvalQuestion
from regqa.ingest.artifacts import publish
from regqa.retrieval.chunking import Chunk, Passage
from regqa.retrieval.hybrid import Ranking
from regqa.retrieval.runtime import RetrievalEngine
from regqa.retrieval.settings import (
    add_input_arguments,
    load_inputs,
    load_settings,
    require_synthetic_output,
)


def oracle_chunks(chunks: list[Chunk], gold: Gold, k: int) -> list[Chunk]:
    remaining = gold.copy()
    matches = [chunk for chunk in chunks if found([chunk], gold)]
    if found(matches, gold) != gold:
        raise ValueError("A dev gold span is not present in the chunked corpus")
    selected = []
    while matches and remaining and len(selected) < k:
        index = max(range(len(matches)), key=lambda i: len(found([matches[i]], remaining)))
        chunk = matches.pop(index)
        selected.append(chunk)
        remaining -= found([chunk], remaining)
    return (selected + matches)[:k]


def cache_row(
    question: EvalQuestion, chunks: list[Chunk], scores: list[float], *, top_k: int
) -> dict[str, Any]:
    passages: list[Passage] = [
        {"chunk_id": c["chunk_id"], "section_ids": c["section_ids"], "text": c["text"]}
        for c in chunks
    ]
    row: dict[str, Any] = {"qid": question.id, "chunks": passages, "scores": scores}
    if question.answerable:
        row["evidence_in_context"] = bool(found(chunks[:top_k], gold_ids(question)))
    return row


def jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def cache_files(
    questions: list[EvalQuestion], rankings: list[Ranking], *, k: int, top_k: int, synthetic: bool
) -> dict[str, bytes]:
    retrieved = []
    oracle = []
    pairs = list(zip(questions, rankings, strict=True))
    for question, ranking in pairs:
        retrieved.append(cache_row(question, ranking.chunks[:k], ranking.scores[:k], top_k=top_k))
        if question.split == "dev":
            try:
                chunks = oracle_chunks(ranking.chunks, gold_ids(question), k)
            except ValueError as error:
                raise ValueError(f"{question.id}: {error}") from None
            oracle.append(cache_row(question, chunks, [1.0] * len(chunks), top_k=top_k))
    metrics = {
        split: retrieval_metrics([(q, r.chunks) for q, r in pairs if q.split == split])
        for split in dict.fromkeys(q.split for q in questions)
    }
    return {
        "contexts.jsonl": jsonl(retrieved),
        "contexts_oracle.jsonl": jsonl(oracle),
        "retrieval_metrics.json": (
            json.dumps(
                {
                    "dataset": FIXTURE if synthetic else "eval",
                    "dev_used_for_chunk_selection": True,
                    **metrics,
                },
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    default = Path(".tmp/retrieval-smoke" if args.synthetic else "data/cache")
    output = args.output_dir or default
    try:
        if args.synthetic:
            require_synthetic_output(output)
        if (output / "contexts.jsonl").exists():
            raise FileExistsError(f"Output already exists: {output / 'contexts.jsonl'}")
        settings = load_settings(args.config)
        corpus, questions, input_hashes = load_inputs(args, dev_only=False)
        report_path = args.calibration or default / "chunk_calibration.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        size = settings.chunking.max_chunk_tokens
        if size is None or size != report.get("chosen_max_chunk_tokens"):
            raise ValueError("Set max_chunk_tokens to the successful dev calibration choice first")
        if (
            report.get("dataset") != (FIXTURE if args.synthetic else "dev")
            or report.get("settings_sha256") != settings.fingerprint()
            or report.get("inputs") != input_hashes
        ):
            raise ValueError("Frozen corpus, dev split, prompt or settings differ from calibration")
        engine = RetrievalEngine(settings)
        if report.get("library_versions") != engine.versions:
            raise ValueError("Installed library versions differ from calibration")
        dev = [q for q in questions if q.split == "dev"]
        test = [q for q in questions if q.split == "test"]
        # Keep dev inference in the same batches as the original calibration.
        rankings = engine.rank(corpus, dev, size)
        if test:
            rankings.extend(engine.rank(corpus, test, size))
        questions = dev + test
        files = cache_files(
            questions,
            rankings,
            k=settings.retrieval.cache_k,
            top_k=settings.retrieval.top_k_context,
            synthetic=args.synthetic,
        )
        output.mkdir(parents=True, exist_ok=True)
        publish(output, files)
        print(f"Frozen retrieval and dev oracle cache for {len(questions)} questions -> {output}")
    except (OSError, ValueError, RuntimeError, ImportError, AttributeError, yaml.YAMLError) as e:
        parser.exit(1, f"{e}\n")


if __name__ == "__main__":
    main()
