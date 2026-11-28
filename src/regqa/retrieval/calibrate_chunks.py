"""Choose a token bound by paired dev recall under the Qwen prompt budget."""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

import numpy as np
import yaml
from numpy.typing import NDArray

from regqa.evaluation.retrieval_metrics import gold_ids, recall_at, retrieval_metrics
from regqa.evaluation.validate_eval import FIXTURE, EvalQuestion
from regqa.generation.prompt_builder import PromptBuilder
from regqa.ingest.artifacts import publish
from regqa.ingest.structure import Corpus
from regqa.retrieval.chunking import ChunkLimitError
from regqa.retrieval.runtime import RetrievalEngine
from regqa.retrieval.settings import (
    add_input_arguments,
    load_inputs,
    load_settings,
    require_synthetic_output,
)


def select_size(
    recalls: dict[int, list[float]], *, seed: int, samples: int
) -> tuple[int, dict[int, list[float]]]:
    best = max(recalls, key=lambda size: (mean(recalls[size]), -size))
    baseline = np.asarray(recalls[best], dtype=np.float64)
    n = len(baseline)
    draws: NDArray[np.int64] = np.random.default_rng(seed).integers(n, size=(samples, n))
    intervals = {}
    for size, scores in recalls.items():
        paired = (baseline - np.asarray(scores))[draws].mean(axis=1)
        intervals[size] = np.percentile(paired, [2.5, 97.5]).tolist()
    return min(size for size, (low, high) in intervals.items() if low <= 0 <= high), intervals


def calibrate(
    engine: RetrievalEngine, corpus: Corpus, dev: list[EvalQuestion], prompt: PromptBuilder
) -> tuple[int | None, dict[int, dict]]:
    settings = engine.settings
    limit = settings.retrieval.prompt_token_budget
    top_k = settings.retrieval.top_k_context
    reports: dict[int, dict] = {}
    recalls: dict[int, list[float]] = {}
    for size in settings.chunking.candidates:
        try:
            rankings = engine.rank(corpus, dev, size)
        except ChunkLimitError as error:
            reports[size] = {"eligible": False, "reason": str(error)}
            continue
        pairs = list(zip(dev, (r.chunks for r in rankings), strict=True))
        prompt_tokens, context_tokens = [], []
        for question, ranked in pairs:
            chunks = [dict(chunk) for chunk in ranked[:top_k]]
            context_tokens.append(
                len(
                    engine.budget_tokenizer.encode(
                        prompt.format_passages(chunks), add_special_tokens=False, truncation=False
                    )
                )
            )
            messages = prompt.build_rag_messages(question.question, chunks)
            prompt_tokens.append(
                len(
                    engine.budget_tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        return_dict=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
            )
        p95 = float(np.percentile(prompt_tokens, 95))
        report = reports[size] = {
            **retrieval_metrics(pairs),
            "mean_top4_context_tokens": mean(context_tokens),
            "p95_prompt_tokens": p95,
            "eligible": p95 <= limit,
        }
        if p95 > limit:
            report["reason"] = f"p95 prompt exceeds {limit} tokens"
            continue
        recalls[size] = [
            recall_at(chunks, gold_ids(q), top_k) for q, chunks in pairs if q.answerable
        ]
    if not recalls:
        return None, reports
    selected, intervals = select_size(
        recalls, seed=settings.calibration.seed, samples=settings.calibration.bootstrap_samples
    )
    for size, interval in intervals.items():
        reports[size]["paired_recall4_gap_ci95"] = interval
    return selected, reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(
        ".tmp/retrieval-smoke/chunk_calibration.json"
        if args.synthetic
        else "data/cache/chunk_calibration.json"
    )
    try:
        if args.synthetic:
            require_synthetic_output(output)
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        settings = load_settings(args.config)
        if not args.synthetic and settings.chunking.max_chunk_tokens is not None:
            raise ValueError("Chunk size is already frozen in the retrieval configuration")
        corpus, dev, input_hashes = load_inputs(args, dev_only=True)
        engine = RetrievalEngine(settings)
        selected, candidates = calibrate(engine, corpus, dev, PromptBuilder(args.prompt_template))
        report = {
            "dataset": FIXTURE if args.synthetic else "dev",
            "date": datetime.now(UTC).isoformat(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "embedding_model": settings.embedding_model,
            "embedding_revision": settings.embedding_revision,
            "faiss_index": settings.retrieval.faiss_index,
            "normalize_embeddings": settings.retrieval.normalize_embeddings,
            "prompt_token_budget": settings.retrieval.prompt_token_budget,
            "settings_sha256": settings.fingerprint(),
            "inputs": input_hashes,
            "library_versions": engine.versions,
            "dev_questions": len(dev),
            "chosen_max_chunk_tokens": selected,
            "candidates": candidates,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        publish(output.parent, {output.name: (json.dumps(report, indent=2) + "\n").encode()})
        if selected is None:
            raise ValueError(f"No candidate meets the constraints. See {output}")
        print(f"Dev calibration chose {selected} tokens -> {output}")
    except (
        OSError,
        ValueError,
        RuntimeError,
        ImportError,
        KeyError,
        yaml.YAMLError,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
