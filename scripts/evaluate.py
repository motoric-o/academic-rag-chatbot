"""CLI entrypoint for evaluation pipeline.

Owner: Member D (spec §4, §11).

Usage:
    python -m scripts.evaluate --predictions runs/local/e1/predictions.jsonl
    python -m scripts.evaluate --predictions runs/local/e1/predictions.jsonl --faithfulness
    python -m scripts.evaluate --create-rating-sheet --predictions runs/local/e1/predictions.jsonl
    python -m scripts.evaluate --agreement results/ratings_A.csv results/ratings_B.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_eval_set(path: str | Path | None = None) -> dict[str, dict]:
    """Load eval set indexed by qid."""
    if path is None:
        path = PROJECT_ROOT / "data" / "eval" / "eval_set.jsonl"
    items = load_jsonl(path)
    return {item["id"]: item for item in items}


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation metrics on predictions."""
    from regqa.analysis.error_taxonomy import (
        classify_errors,
        compute_error_taxonomy,
    )
    from regqa.evaluation.abstention import evaluate_abstention_batch
    from regqa.evaluation.citation import evaluate_citation_batch
    from regqa.evaluation.faithfulness import evaluate_faithfulness_batch
    from regqa.evaluation.keyfact import evaluate_keyfact_batch

    predictions = load_jsonl(args.predictions)
    print(f"Loaded {len(predictions)} predictions from {args.predictions}")

    # Load eval set for key-fact and error analysis
    eval_set = load_eval_set(args.eval_set)

    # --- Deterministic metrics (keyfact, abstention, optional citation) ---
    kf = evaluate_keyfact_batch(predictions, eval_set)
    print("\n=== Key-Fact Correctness ===")
    print(f"  Strict correctness:  {kf['strict_correctness']:.1%} (n={kf['n_evaluated']})")
    print(f"  Soft key-fact rec:   {kf['soft_keyfact_recall']:.1%}")
    print(f"  Format valid rate:   {kf['format_ok_rate']:.1%}")

    contexts_by_qid = None
    if args.contexts and Path(args.contexts).is_file():
        contexts_by_qid = {str(r["qid"]): r for r in load_jsonl(args.contexts)}
        cit = evaluate_citation_batch(predictions, eval_set, contexts_by_qid)
        print("\n=== Citation Quality ===")
        print(f"  Citation precision:  {cit['citation_precision']}")
        print(f"  Citation recall:     {cit['citation_recall']}")
        print(f"  Invalid cit. rate:   {cit['invalid_citation_rate']:.1%}")

    abst = evaluate_abstention_batch(predictions, eval_set, contexts_by_qid)
    print("\n=== Abstention ===")
    print(f"  Abstention recall:   {abst.recall}")
    print(f"  False-answer rate:   {abst.false_answer_rate}")
    print(f"  Over-abstention:     {abst.over_abstention_rate}")

    # --- Error taxonomy ---
    annotations = []
    for pred in predictions:
        qid = pred.get("qid", "")
        eval_item = eval_set.get(qid, {})
        ann = classify_errors(pred, eval_item)
        annotations.append(ann)

    taxonomy = compute_error_taxonomy(annotations)
    print("\n=== Error Taxonomy ===")
    print(f"  Total predictions:   {taxonomy['total_predictions']}")
    print(f"  With errors:         {taxonomy['total_with_errors']}")
    print(f"  Error rate:          {taxonomy['error_rate']:.1%}")
    print("  Category breakdown:")
    for cat, count in taxonomy["category_counts"].items():
        rate = taxonomy["category_rates"][cat]
        if count > 0:
            print(f"    {cat:25s}  {count:4d}  ({rate:.1%})")

    # --- Faithfulness (optional, slow) ---
    if args.faithfulness:
        print("\n=== Faithfulness (NLI) ===")
        faith_results = evaluate_faithfulness_batch(
            predictions,
            threshold=args.threshold,
            use_model=not args.no_model,
        )
        print(f"  Mean faithfulness:   {faith_results['mean_faithfulness']:.3f}")
        print(f"  Hallucination rate:  {faith_results['hallucination_rate']:.3f}")
        print(f"  Num predictions:     {faith_results['num_predictions']}")

        # Save detailed results
        output_path = Path(args.predictions).parent / "faithfulness_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(faith_results, f, indent=2, ensure_ascii=False)
        print(f"  Detailed results:    {output_path}")


def cmd_create_rating_sheet(args: argparse.Namespace) -> None:
    """Create a human rating CSV template."""
    from regqa.evaluation.judge import create_rating_sheet_template

    predictions = load_jsonl(args.predictions)
    output = Path(args.output or "results/human_rating_template.csv")

    path = create_rating_sheet_template(output, predictions)
    print(f"Created rating sheet template: {path}")
    print("  Columns: qid, model, precision, answer, rating, rater, notes")
    print("  Fill in 'rating' (0/1/2), 'rater' (your initial), and optional 'notes'.")


def cmd_agreement(args: argparse.Namespace) -> None:
    """Compute inter-rater agreement between two rating CSVs."""
    from regqa.evaluation.judge import compute_agreement

    result = compute_agreement(args.file_a, args.file_b)
    print("=== Inter-Rater Agreement ===")
    print(f"  Overlapping items: {result['n_overlap']}")
    print(f"  Exact agreement:   {result['agreement_pct']:.1f}%")
    print(f"  Cohen's kappa:     {result['kappa']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluation pipeline for Academic RAG Chatbot (Member D)",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- evaluate ---
    p_eval = subparsers.add_parser("evaluate", help="Run evaluation metrics")
    p_eval.add_argument(
        "--predictions", required=True, help="Path to predictions.jsonl"
    )
    p_eval.add_argument(
        "--eval-set", default=None, help="Path to eval_set.jsonl (default: data/eval/)"
    )
    p_eval.add_argument(
        "--contexts", default=None, help="Optional retrieval cache contexts.jsonl path"
    )
    p_eval.add_argument(
        "--faithfulness",
        action="store_true",
        help="Also run NLI faithfulness check (slow, needs transformers)",
    )
    p_eval.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="NLI entailment threshold (default: 0.5)",
    )
    p_eval.add_argument(
        "--no-model",
        action="store_true",
        help="Skip loading NLI model (placeholder mode)",
    )

    # --- create-rating-sheet ---
    p_sheet = subparsers.add_parser(
        "create-rating-sheet", help="Create human rating CSV template"
    )
    p_sheet.add_argument(
        "--predictions", required=True, help="Path to predictions.jsonl"
    )
    p_sheet.add_argument(
        "--output", default=None, help="Output CSV path"
    )

    # --- agreement ---
    p_agree = subparsers.add_parser(
        "agreement", help="Compute inter-rater agreement (Cohen's kappa)"
    )
    p_agree.add_argument("file_a", help="First rater's CSV file")
    p_agree.add_argument("file_b", help="Second rater's CSV file")

    args = parser.parse_args()

    if args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "create-rating-sheet":
        cmd_create_rating_sheet(args)
    elif args.command == "agreement":
        cmd_agreement(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
