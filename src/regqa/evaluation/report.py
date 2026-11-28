"""Automated evaluation report & Markdown/JSON table generator.

Owner: Member B (see docs/project_spec.md §11.7, §13, Milestone M6/M8).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from regqa.evaluation.abstention import evaluate_abstention_batch
from regqa.evaluation.citation import evaluate_citation_batch
from regqa.evaluation.efficiency import (
    compute_efficiency_summary,
    format_efficiency_markdown,
)
from regqa.evaluation.keyfact import evaluate_keyfact_batch
from regqa.evaluation.stats import bootstrap_ci, compare_systems_pairwise

# Exploratory demo-suitability thresholds (spec §11.7; operational choices, not results)
SUCCESS_TARGETS: dict[str, tuple[str, float]] = {
    "strict_correctness": (">=", 0.80),
    "faithfulness": (">=", 0.90),
    "format_ok_rate": (">=", 0.95),
    "abstention_recall": (">=", 0.80),
    "over_abstention_rate": ("<=", 0.10),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def check_success_targets(metrics: Mapping[str, float | None]) -> dict[str, dict[str, Any]]:
    """Evaluate exploratory §11.7 demo-suitability targets."""
    checks: dict[str, dict[str, Any]] = {}
    for name, (op, threshold) in SUCCESS_TARGETS.items():
        val = metrics.get(name)
        if val is None:
            passed: bool | None = None
        elif op == ">=":
            passed = val >= threshold
        else:
            passed = val <= threshold
        checks[name] = {
            "value": val,
            "operator": op,
            "target": threshold,
            "meets_target": passed,
        }
    return checks


def evaluate_run(
    predictions_path: str | Path,
    eval_by_qid: Mapping[str, Mapping[str, Any]],
    contexts_by_qid: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    top_k_context: int = 4,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute all deterministic + efficiency metrics for a single predictions.jsonl file."""
    pred_path = Path(predictions_path)
    predictions = _read_jsonl(pred_path)
    if not predictions:
        raise ValueError(f"{pred_path}: predictions file is empty")

    model_name = str(predictions[0].get("model", pred_path.parent.name))
    precision = str(predictions[0].get("precision", "unknown"))
    system_id = f"{model_name} ({precision})"

    kf = evaluate_keyfact_batch(predictions, eval_by_qid)
    abst = evaluate_abstention_batch(predictions, eval_by_qid, contexts_by_qid)
    cit = (
        evaluate_citation_batch(
            predictions, eval_by_qid, contexts_by_qid, top_k_context=top_k_context
        )
        if contexts_by_qid
        else None
    )
    eff = compute_efficiency_summary(predictions, model=model_name, precision=precision)

    # Bootstrap CI on per-question strict correctness
    strict_vec = list(kf["per_question_mean"].values())
    strict_ci = (
        bootstrap_ci(strict_vec, n_resamples=n_resamples, seed=seed).to_dict()
        if strict_vec
        else None
    )

    # Optional NLI faithfulness sidecar (written by scripts/evaluate.py)
    faithfulness_val: float | None = None
    faith_sidecar = pred_path.parent / "faithfulness_results.json"
    if faith_sidecar.is_file():
        faith_data = json.loads(faith_sidecar.read_text(encoding="utf-8"))
        if faith_data.get("mean_faithfulness") is not None:
            faithfulness_val = round(float(faith_data["mean_faithfulness"]), 4)

    targets = check_success_targets(
        {
            "strict_correctness": kf["strict_correctness"],
            "faithfulness": faithfulness_val,
            "format_ok_rate": kf["format_ok_rate"],
            "abstention_recall": abst.recall,
            "over_abstention_rate": abst.over_abstention_rate,
        }
    )

    return {
        "system_id": system_id,
        "model": model_name,
        "precision": precision,
        "predictions_path": str(pred_path),
        "n_predictions": len(predictions),
        "keyfact": {k: v for k, v in kf.items() if k != "details"},
        "strict_correctness_ci": strict_ci,
        "citation": {k: v for k, v in cit.items() if k != "details"} if cit else None,
        "abstention": abst.to_dict(),
        "efficiency": eff.to_dict(),
        "faithfulness": faithfulness_val,
        "success_targets": targets,
    }


def _fmt_pct(val: float | None) -> str:
    return "n/a" if val is None else f"{val * 100:.1f}%"


def format_report_markdown(report: Mapping[str, Any]) -> str:
    """Render headline, abstention, efficiency, and pairwise statistics tables in Markdown."""
    systems: list[dict[str, Any]] = list(report["systems"])
    lines: list[str] = [
        "# Evaluation Report",
        "",
        f"- **Generated (UTC):** {report['provenance']['timestamp_utc']}",
        f"- **Git commit:** `{report['provenance']['git_commit'] or 'unknown'}`",
        f"- **Eval set:** `{report['provenance']['eval_path']}` "
        f"(SHA-256: `{report['provenance']['eval_sha256'] or 'n/a'}`)",
        "",
        "## Headline Quality Metrics",
        "",
        "| System | n | Coverage | Strict Correctness (95% CI) | Soft Recall | "
        "Format OK | Cit. Prec | Cit. Rec | Invalid Cit. |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]

    for s in systems:
        kf = s["keyfact"]
        ci = s["strict_correctness_ci"]
        if ci:
            strict_str = (
                f"{_fmt_pct(ci['mean'])} "
                f"[{_fmt_pct(ci['ci_low'])}, {_fmt_pct(ci['ci_high'])}]"
            )
        else:
            strict_str = _fmt_pct(kf["strict_correctness"])
        cit = s["citation"] or {}
        lines.append(
            f"| {s['system_id']} | {kf['n_unique_questions']} | {_fmt_pct(kf['coverage'])} | "
            f"{strict_str} | {_fmt_pct(kf['soft_keyfact_recall'])} | "
            f"{_fmt_pct(kf['format_ok_rate'])} | {_fmt_pct(cit.get('citation_precision'))} | "
            f"{_fmt_pct(cit.get('citation_recall'))} | "
            f"{_fmt_pct(cit.get('invalid_citation_rate'))} |"
        )

    lines.extend(
        [
            "",
            "## Abstention & Evidence Sufficiency (§11.4)",
            "",
            "| System | Abst. Prec | Abst. Rec | Abst. F1 | False-Answer | "
            "Over-Abst. (All) | Over-Abst. (Evid=True) | Over-Abst. (Evid=False) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for s in systems:
        a = s["abstention"]
        lines.append(
            f"| {s['system_id']} | {_fmt_pct(a['precision'])} | {_fmt_pct(a['recall'])} | "
            f"{_fmt_pct(a['f1'])} | {_fmt_pct(a['false_answer_rate'])} | "
            f"{_fmt_pct(a['over_abstention_rate'])} | "
            f"{_fmt_pct(a['over_abstention_with_evidence'])} | "
            f"{_fmt_pct(a['over_abstention_without_evidence'])} |"
        )

    from regqa.evaluation.efficiency import EfficiencySummary

    eff_objs = [EfficiencySummary(**s["efficiency"]) for s in systems]
    lines.extend(["", "## Efficiency (§11.5)", "", format_efficiency_markdown(eff_objs)])

    pairwise: list[dict[str, Any]] = list(report.get("pairwise_stats") or [])
    if pairwise:
        lines.extend(
            [
                "",
                "## Pairwise Comparisons (Exact McNemar + Holm Correction & Flips, §11.6)",
                "",
                "| System A | System B | Diff (B - A) | 95% CI | McNemar p | Holm p | "
                "C->W | W->C | Status |",
                "|---|---|---:|---|---:|---:|---:|---:|---|",
            ]
        )
        for p in pairwise:
            b = p["paired_bootstrap"]
            m = p["mcnemar"]
            f = p["flips"]
            status = "inconclusive" if b["is_inconclusive"] else "significant_ci"
            lines.append(
                f"| {p['system_a']} | {p['system_b']} | {b['mean_diff']:+.4f} | "
                f"[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] | {m['p_value']:.4f} | "
                f"{m['p_value_holm']:.4f} | {f['correct_to_wrong']} | "
                f"{f['wrong_to_correct']} | {status} |"
            )

    return "\n".join(lines) + "\n"


def build_full_report(
    prediction_paths: Sequence[str | Path],
    eval_path: str | Path,
    *,
    contexts_path: str | Path | None = None,
    top_k_context: int = 4,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Build complete evaluation report JSON + Markdown tables across systems."""
    eval_p = Path(eval_path)
    eval_rows = _read_jsonl(eval_p)
    eval_by_qid = {str(r["id"]): r for r in eval_rows}

    contexts_by_qid: dict[str, dict[str, Any]] | None = None
    if contexts_path is not None and Path(contexts_path).is_file():
        contexts_by_qid = {str(r["qid"]): r for r in _read_jsonl(Path(contexts_path))}

    systems = [
        evaluate_run(
            path,
            eval_by_qid,
            contexts_by_qid,
            top_k_context=top_k_context,
            n_resamples=n_resamples,
            seed=seed,
        )
        for path in prediction_paths
    ]

    binary_map = {
        s["system_id"]: s["keyfact"]["per_question_binary"] for s in systems
    }
    pairwise = (
        compare_systems_pairwise(binary_map, n_resamples=n_resamples, seed=seed)
        if len(systems) >= 2
        else []
    )

    return {
        "provenance": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "git_commit": _git_commit(),
            "eval_path": str(eval_p),
            "eval_sha256": _file_sha256(eval_p),
            "contexts_path": str(contexts_path) if contexts_path else None,
            "n_resamples": n_resamples,
            "seed": seed,
        },
        "systems": systems,
        "pairwise_stats": pairwise,
    }


def write_report(report: Mapping[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Write report_summary.json and report_tables.md to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "report_summary.json"
    md_path = out / "report_tables.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(format_report_markdown(report), encoding="utf-8")
    return json_path, md_path
