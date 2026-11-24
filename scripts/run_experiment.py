"""CLI entrypoint for resumable experiment runs.

Owner: Member C (Francisco Valentino; see docs/project_spec.md §4, §10 & §13).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from regqa.generation.prompt_builder import PromptBuilder
from regqa.models.base import GenerationConfig, MockAdapter, ModelAdapter
from regqa.models.hosted_openai_compat import HostedOpenAICompatAdapter
from regqa.pipeline import RAGPipeline
from regqa.utils.io import append_jsonl, read_jsonl, read_yaml, write_json
from regqa.utils.logging import get_logger
from regqa.utils.seeds import seed_everything

logger = get_logger("regqa.experiment")


def load_adapter(model_cfg: dict[str, Any], force_mock: bool = False) -> ModelAdapter:
    """Load adapter based on model configuration and runtime environment."""
    provider = model_cfg.get("provider", "mock")
    model_name = model_cfg.get("name", "unknown")
    precision = model_cfg.get("precision", "hosted")

    if force_mock or provider == "mock":
        return MockAdapter(model=model_name, precision=precision)

    if provider in ("local", "local_hf"):
        try:
            from regqa.models.local_hf import LocalHFAdapter, LocalHFConfig

            cfg = LocalHFConfig(
                model_name=model_name,
                model_id=model_cfg.get("model_id", ""),
                precision=precision,
                target_hardware=model_cfg.get("target_hardware"),
            )
            return LocalHFAdapter(cfg)
        except Exception as e:
            logger.warning(f"Could not load LocalHFAdapter ({e}); falling back to MockAdapter.")
            return MockAdapter(model=model_name, precision=precision)

    if provider == "hosted_openai_compat":
        return HostedOpenAICompatAdapter(
            model_id=model_cfg.get("model_id", "openai/gpt-oss-20b"),
            base_url=model_cfg.get("base_url", "https://api.groq.com/openai/v1"),
            api_key_env=model_cfg.get("api_key_env", "GROQ_API_KEY"),
            precision=precision,
            rate_limits=model_cfg.get("rate_limits"),
            reasoning_effort=model_cfg.get("generation", {}).get("reasoning_effort", "low"),
        )

    return MockAdapter(model=model_name, precision=precision)


def run_experiment(
    config_path: str | Path,
    run_id: str | None = None,
    split_override: str | None = None,
    force_mock: bool = False,
    output_base_dir: str | Path = "runs",
) -> None:
    """Execute experiment run across configured models."""
    exp_cfg = read_yaml(config_path)
    base_cfg = read_yaml("configs/base.yaml")
    seed_everything(base_cfg.get("seed", 42))

    exp_name = exp_cfg.get("name", Path(config_path).stem)
    split = split_override or exp_cfg.get("split", "dev")
    repeats = int(exp_cfg.get("repeats", base_cfg.get("generation", {}).get("repeats", 1)))
    use_context = exp_cfg.get("use_context", True)

    # Generate timestamped run_id if not supplied
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Load eval set
    eval_path = Path(base_cfg.get("eval_set_path", "data/eval/eval_set.jsonl"))
    if not eval_path.is_file():
        eval_path = Path("tests/fixtures/mini_eval.jsonl")
    all_questions = read_jsonl(eval_path)
    questions = [q for q in all_questions if q.get("split", "dev") == split] or all_questions

    logger.info(f"Loaded {len(questions)} questions for split='{split}' from {eval_path}")

    # Load retrieval cache if RAG mode
    contexts_by_qid: dict[str, list[dict[str, Any]]] = {}
    if use_context:
        cache_path = Path(base_cfg.get("contexts_cache_path", "data/cache/contexts.jsonl"))
        if cache_path.is_file():
            cache_records = read_jsonl(cache_path)
            for r in cache_records:
                contexts_by_qid[r["qid"]] = r.get("chunks", [])

    prompt_builder = PromptBuilder()

    for model_cfg_path in exp_cfg.get("models", []):
        m_cfg = read_yaml(model_cfg_path)
        model_name = m_cfg.get("name", Path(model_cfg_path).stem)
        provider = "local" if "local" in m_cfg.get("provider", "") else "hosted"

        output_dir = Path(output_base_dir) / provider / exp_name / model_name / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_file = output_dir / "predictions.jsonl"

        # Check existing predictions for resumability
        existing = read_jsonl(pred_file)
        completed_keys = {(r.get("qid"), r.get("repeat_idx", 0)) for r in existing}

        logger.info(f"Starting model '{model_name}' (resumable: {len(completed_keys)} already done)")
        adapter = load_adapter(m_cfg, force_mock=force_mock)
        pipeline = RAGPipeline(adapter=adapter, prompt_builder=prompt_builder)

        for rep in range(repeats):
            for q in questions:
                qid = q.get("id") or q.get("qid", "q_unknown")
                if (qid, rep) in completed_keys:
                    continue

                q_text = q.get("question", "")
                chunks = contexts_by_qid.get(qid, [
                    {"text": "Sample regulation context passage.", "header": "[Chapter 1 > Article 1]"}
                ])

                out = pipeline.run_query(
                    question=q_text,
                    context_chunks=chunks if use_context else None,
                    is_closed_book=not use_context,
                )

                rec = out.to_prediction_record(qid=qid, run_id=run_id)
                rec["repeat_idx"] = rep
                append_jsonl(pred_file, rec)
                completed_keys.add((qid, rep))

        summary = {
            "exp_name": exp_name,
            "model_name": model_name,
            "run_id": run_id,
            "split": split,
            "total_questions": len(questions),
            "repeats": repeats,
            "total_records": len(completed_keys),
            "output_file": str(pred_file),
        }
        write_json(output_dir / "summary.json", summary)
        logger.info(f"Finished {model_name}. Summary saved to {output_dir / 'summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG Experiment CLI")
    parser.add_argument("--config", type=str, default="configs/experiments/e3_local_vs_hosted.yaml")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--mock", action="store_true", help="Force MockAdapter execution")
    args = parser.parse_args()

    run_experiment(
        config_path=args.config,
        run_id=args.run_id,
        split_override=args.split,
        force_mock=args.mock,
    )


if __name__ == "__main__":
    main()
