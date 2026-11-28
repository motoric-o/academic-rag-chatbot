"""CLI entrypoint for generating result tables and summary JSON (`make report`).

Owner: Member B (see docs/project_spec.md §4, §11, §13).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from regqa.evaluation.report import build_full_report, format_report_markdown, write_report


def _discover_predictions(runs_dir: Path) -> list[Path]:
    if not runs_dir.is_dir():
        return []
    return sorted(runs_dir.rglob("predictions.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        help=(
            "One or more predictions.jsonl files "
            "(auto-discovers runs/**/predictions.jsonl if omitted)"
        ),
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=Path("data/eval/eval_set.jsonl"),
        help="Evaluation JSONL path (default: data/eval/eval_set.jsonl)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use tests/fixtures/mini_eval.jsonl when --eval-set is not present",
    )
    parser.add_argument(
        "--contexts",
        type=Path,
        default=Path("data/cache/contexts.jsonl"),
        help="Optional retrieval cache JSONL for citation & evidence-split metrics",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write report_summary.json and report_tables.md",
    )
    parser.add_argument("--top-k-context", type=int, default=4)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pred_paths = list(args.predictions) if args.predictions else _discover_predictions(Path("runs"))
    if not pred_paths:
        parser.exit(1, "No predictions.jsonl files found. Pass --predictions <path>.\n")

    eval_path = args.eval_set
    if not eval_path.is_file():
        if args.synthetic:
            eval_path = Path("tests/fixtures/mini_eval.jsonl")
        else:
            parser.exit(1, f"Eval set not found: {eval_path} (pass --synthetic for fixtures)\n")

    contexts_path = args.contexts if args.contexts and args.contexts.is_file() else None
    report = build_full_report(
        pred_paths,
        eval_path,
        contexts_path=contexts_path,
        top_k_context=args.top_k_context,
        n_resamples=args.resamples,
        seed=args.seed,
    )
    json_path, md_path = write_report(report, args.output_dir)
    print(format_report_markdown(report))
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
