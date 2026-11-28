"""CLI entrypoint for resumable experiment runs.

Owner: Member C (Francisco Valentino; see docs/project_spec.md §4, §10 & §13).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from regqa.generation.prompt_builder import PromptBuilder
from regqa.models.base import GenerationConfig, MockAdapter, ModelAdapter
from regqa.models.hosted_openai_compat import HostedOpenAICompatAdapter
from regqa.models.rate_limiter import RateLimitExceededError
from regqa.pipeline import RAGPipeline
from regqa.utils.io import append_jsonl, read_jsonl, read_yaml, write_json
from regqa.utils.logging import get_logger
from regqa.utils.seeds import seed_everything

logger = get_logger("regqa.experiment")


def load_adapter(
    model_cfg: dict[str, Any],
    *,
    model_cfg_path: str | Path | None = None,
    force_mock: bool = False,
) -> ModelAdapter:
    """Load adapter based on model configuration and runtime environment."""
    provider = model_cfg.get("provider", "mock")
    model_name = model_cfg.get("name", "unknown")
    precision = model_cfg.get("precision", "hosted")

    if force_mock or provider == "mock":
        return MockAdapter(model=model_name, precision=precision)

    if provider in ("local", "local_hf"):
        from regqa.models.local_hf import LocalHFAdapter, LocalHFConfig

        if model_cfg_path is not None:
            cfg = LocalHFConfig.from_yaml(model_cfg_path)
        else:
            gen = model_cfg.get("generation") or {}
            cfg = LocalHFConfig(
                name=model_name,
                hf_model_id=str(
                    model_cfg.get("hf_model_id") or model_cfg.get("model_id", "")
                ),
                precision=precision,
                revision=model_cfg.get("revision"),
                quantization=model_cfg.get("quantization"),
                enable_thinking=bool(gen.get("enable_thinking", False)),
                target_hardware=model_cfg.get("target_hardware"),
                default_generation=dict(gen) if gen else None,
            )
        return LocalHFAdapter(cfg)

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


def _build_gen_config(
    base_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
) -> GenerationConfig:
    base_gen = base_cfg.get("generation") or {}
    model_gen = model_cfg.get("generation") or {}
    merged = {**base_gen, **model_gen}
    return GenerationConfig(
        temperature=float(merged.get("temperature", 0.0)),
        do_sample=bool(merged.get("do_sample", False)),
        max_new_tokens=int(merged.get("max_new_tokens", 256)),
        enable_thinking=bool(merged.get("enable_thinking", False)),
    )


def _load_top_k_context(retrieval_cfg_path: str | Path = "configs/retrieval.yaml") -> int:
    path = Path(retrieval_cfg_path)
    if path.is_file():
        data = read_yaml(path)
        retrieval = data.get("retrieval") or {}
        if "top_k_context" in retrieval:
            return int(retrieval["top_k_context"])
    return 4


def run_experiment(
    config_path: str | Path,
    run_id: str | None = None,
    split_override: str | None = None,
    force_mock: bool = False,
    output_base_dir: str | Path = "runs",
    contexts_cache_path: str | Path | None = None,
    top_k_context: int | None = None,
) -> None:
    """Execute experiment run across configured models."""
    exp_cfg = read_yaml(config_path)
    base_cfg = read_yaml("configs/base.yaml")
    seed_everything(base_cfg.get("seed", 42))

    exp_name = exp_cfg.get("name", Path(config_path).stem)
    split = split_override or exp_cfg.get("split", "dev")
    repeats = int(exp_cfg.get("repeats", base_cfg.get("generation", {}).get("repeats", 1)))
    use_context = bool(exp_cfg.get("use_context", True))
    k_context = top_k_context if top_k_context is not None else _load_top_k_context()

    # Generate timestamped run_id if not supplied
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    # Load eval set
    eval_path = Path(base_cfg.get("eval_set_path", "data/eval/eval_set.jsonl"))
    if not eval_path.is_file():
        if not force_mock:
            raise FileNotFoundError(
                f"Eval set not found: {eval_path}. Pass --mock to run on synthetic fixtures."
            )
        eval_path = Path("tests/fixtures/mini_eval.jsonl")
    all_questions = read_jsonl(eval_path)
    questions = [q for q in all_questions if q.get("split", "dev") == split] or all_questions

    logger.info(f"Loaded {len(questions)} questions for split='{split}' from {eval_path}")

    # Load retrieval cache if RAG mode (§6: cache holds 5, feed top_k_context=4)
    contexts_by_qid: dict[str, list[dict[str, Any]]] = {}
    if use_context:
        cache_path = Path(
            contexts_cache_path
            if contexts_cache_path is not None
            else base_cfg.get("contexts_cache_path", "data/cache/contexts.jsonl")
        )
        if cache_path.is_file():
            for r in read_jsonl(cache_path):
                contexts_by_qid[str(r["qid"])] = list(r.get("chunks", []))[:k_context]
        elif not force_mock:
            raise FileNotFoundError(
                f"Retrieval cache not found: {cache_path}. Build cache first or pass --mock."
            )

    prompt_builder = PromptBuilder()

    for model_cfg_path in exp_cfg.get("models", []):
        m_cfg = read_yaml(model_cfg_path)
        model_name = m_cfg.get("name", Path(model_cfg_path).stem)
        provider = "local" if "local" in m_cfg.get("provider", "") else "hosted"

        output_dir = Path(output_base_dir) / provider / exp_name / model_name / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_file = output_dir / "predictions.jsonl"

        # Check existing predictions for resumability (only count error-free rows)
        existing = read_jsonl(pred_file)
        completed_keys = {
            (r.get("qid"), r.get("repeat_idx", 0))
            for r in existing
            if r.get("error") is None
        }

        logger.info(
            f"Starting model '{model_name}' (resumable: {len(completed_keys)} already done)"
        )
        adapter = load_adapter(
            m_cfg,
            model_cfg_path=model_cfg_path,
            force_mock=force_mock,
        )
        gen_config = _build_gen_config(base_cfg, m_cfg)
        pipeline = RAGPipeline(
            adapter=adapter,
            prompt_builder=prompt_builder,
            gen_config=gen_config,
        )

        stopped_for_quota = False
        try:
            for rep in range(repeats):
                if stopped_for_quota:
                    break
                for q in questions:
                    qid = str(q.get("id") or q.get("qid", "q_unknown"))
                    if (qid, rep) in completed_keys:
                        continue

                    q_text = q.get("question", "")
                    if use_context:
                        if qid in contexts_by_qid:
                            chunks = contexts_by_qid[qid]
                        elif force_mock:
                            chunks = [
                                {
                                    "text": "Sample regulation context passage.",
                                    "header": "[Chapter 1 > Article 1]",
                                }
                            ]
                        else:
                            raise KeyError(
                                f"Question ID '{qid}' missing from retrieval cache {cache_path}"
                            )
                    else:
                        chunks = None

                    try:
                        out = pipeline.run_query(
                            question=q_text,
                            context_chunks=chunks,
                            is_closed_book=not use_context,
                        )
                    except RateLimitExceededError as exc:
                        logger.warning(
                            f"Daily quota exhausted for '{model_name}' ({exc}); "
                            "stopping cleanly so run can resume later."
                        )
                        stopped_for_quota = True
                        break

                    rec = out.to_prediction_record(qid=qid, run_id=run_id)
                    rec["repeat_idx"] = rep
                    append_jsonl(pred_file, rec)
                    if rec.get("error") is None:
                        completed_keys.add((qid, rep))
        finally:
            unload = getattr(adapter, "unload", None)
            if callable(unload):
                unload()

        summary = {
            "exp_name": exp_name,
            "model_name": model_name,
            "run_id": run_id,
            "split": split,
            "total_questions": len(questions),
            "repeats": repeats,
            "top_k_context": k_context if use_context else 0,
            "total_records": len(completed_keys),
            "stopped_for_quota": stopped_for_quota,
            "output_file": str(pred_file),
        }
        write_json(output_dir / "summary.json", summary)
        logger.info(f"Finished {model_name}. Summary saved to {output_dir / 'summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG Experiment CLI")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/e3_local_vs_hosted.yaml",
    )
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
