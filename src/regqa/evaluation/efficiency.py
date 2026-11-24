"""Local efficiency metrics: VRAM, load time, p50/p95 latency, tokens/sec.

Owner: Member A (see docs/project_spec.md §0.2, §0.4, §11.5).
Note: Local latency must be reported ONLY from the RTX 3070 machine; never mix
Colab/Kaggle timings with 3070 timings.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from regqa.models.base import GenerationResult


@dataclass(frozen=True)
class EfficiencySummary:
    """Aggregated efficiency metrics for a single model/precision run."""

    model: str
    precision: str
    hardware_label: str
    n_samples: int
    load_time_s: float | None
    peak_vram_mb: float | None
    latency_p50_s: float | None
    latency_p95_s: float | None
    tokens_per_sec: float | None
    mean_tokens_in: float
    mean_tokens_out: float
    is_reference_hardware: bool = False
    latency_status: str = "measured"
    error_count: int = 0
    warmup_excluded: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_hardware_label(hardware: Mapping[str, Any] | None) -> str:
    """Extract a normalized hardware label from a prediction's hardware dict."""
    if not hardware:
        return "unknown"
    gpu_name = hardware.get("gpu_name")
    if gpu_name:
        return str(gpu_name).strip()
    device = hardware.get("device")
    if device:
        return str(device).strip()
    return "unknown"


def is_reference_rtx3070(hardware: Mapping[str, Any] | None) -> bool:
    """Return True iff the record was measured on the reference RTX 3070 GPU (§0.4, §11.5)."""
    if not hardware:
        return False
    device = str(hardware.get("device", "")).strip().lower()
    gpu_name = str(hardware.get("gpu_name") or "").strip().lower()
    return device == "cuda" and "3070" in gpu_name


def _normalize_record(record: Mapping[str, Any] | GenerationResult) -> dict[str, Any]:
    if isinstance(record, GenerationResult):
        return {
            "model": record.model,
            "precision": record.precision,
            "latency_s": record.latency_s,
            "tokens_in": record.tokens_in,
            "tokens_out": record.tokens_out,
            "peak_vram_mb": record.peak_vram_mb,
            "hardware": record.hardware,
            "error": record.error,
        }
    return dict(record)


def compute_efficiency_summary(
    records: Sequence[Mapping[str, Any] | GenerationResult],
    *,
    model: str | None = None,
    precision: str | None = None,
    load_time_s: float | None = None,
    warmup_count: int = 0,
    enforce_reference_gpu_for_latency: bool = True,
) -> EfficiencySummary:
    """Aggregate VRAM, latency, and throughput metrics from run records.

    Args:
        records: Sequence of GenerationResult objects or predictions.jsonl dicts.
        model: Optional model name override.
        precision: Optional precision override.
        load_time_s: Optional model load time in seconds.
        warmup_count: Number of leading warm-up records to exclude (§11.5).
        enforce_reference_gpu_for_latency: If True (default), suppress latency/tok-s
            for non-RTX-3070 or mixed hardware runs so timings are never mixed (§0.4).
    """
    if warmup_count < 0:
        raise ValueError("warmup_count must be >= 0")

    normalized = [_normalize_record(r) for r in records]
    trimmed = normalized[warmup_count:] if warmup_count else normalized

    first = trimmed[0] if trimmed else (normalized[0] if normalized else {})
    resolved_model = model or str(first.get("model", "unknown"))
    resolved_precision = precision or str(first.get("precision", "unknown"))

    valid = [r for r in trimmed if r.get("error") is None]
    error_count = len(trimmed) - len(valid)

    hw_labels = {extract_hardware_label(r.get("hardware")) for r in valid}
    if not valid:
        hw_label = extract_hardware_label(first.get("hardware")) if first else "unknown"
        return EfficiencySummary(
            model=resolved_model,
            precision=resolved_precision,
            hardware_label=hw_label,
            n_samples=0,
            load_time_s=round(load_time_s, 4) if load_time_s is not None else None,
            peak_vram_mb=None,
            latency_p50_s=None,
            latency_p95_s=None,
            tokens_per_sec=None,
            mean_tokens_in=0.0,
            mean_tokens_out=0.0,
            is_reference_hardware=False,
            latency_status="no_valid_samples",
            error_count=error_count,
            warmup_excluded=min(warmup_count, len(normalized)),
        )

    mixed_hardware = len(hw_labels) > 1
    hw_label = "mixed" if mixed_hardware else next(iter(hw_labels))
    is_ref = (not mixed_hardware) and all(is_reference_rtx3070(r.get("hardware")) for r in valid)

    vram_values = [
        float(r["peak_vram_mb"]) for r in valid if r.get("peak_vram_mb") is not None
    ]
    peak_vram_mb = round(max(vram_values), 2) if vram_values else None

    tokens_in = [int(r.get("tokens_in", 0)) for r in valid]
    tokens_out = [int(r.get("tokens_out", 0)) for r in valid]
    mean_tokens_in = round(float(np.mean(tokens_in)), 2)
    mean_tokens_out = round(float(np.mean(tokens_out)), 2)

    if mixed_hardware:
        latency_p50_s = None
        latency_p95_s = None
        tokens_per_sec = None
        latency_status = "mixed_hardware_rejected"
    elif enforce_reference_gpu_for_latency and not is_ref:
        latency_p50_s = None
        latency_p95_s = None
        tokens_per_sec = None
        latency_status = "not_measured_non_reference_gpu"
    else:
        latencies = [float(r["latency_s"]) for r in valid]
        total_latency = float(sum(latencies))
        latency_p50_s = round(float(np.percentile(latencies, 50)), 4)
        latency_p95_s = round(float(np.percentile(latencies, 95)), 4)
        tokens_per_sec = (
            round(float(sum(tokens_out)) / total_latency, 2) if total_latency > 0 else None
        )
        latency_status = "measured"

    return EfficiencySummary(
        model=resolved_model,
        precision=resolved_precision,
        hardware_label=hw_label,
        n_samples=len(valid),
        load_time_s=round(load_time_s, 4) if load_time_s is not None else None,
        peak_vram_mb=peak_vram_mb,
        latency_p50_s=latency_p50_s,
        latency_p95_s=latency_p95_s,
        tokens_per_sec=tokens_per_sec,
        mean_tokens_in=mean_tokens_in,
        mean_tokens_out=mean_tokens_out,
        is_reference_hardware=is_ref,
        latency_status=latency_status,
        error_count=error_count,
        warmup_excluded=min(warmup_count, len(normalized)),
    )


def load_predictions_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a predictions.jsonl file into a list of dicts."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def summarize_predictions_jsonl(
    path: str | Path,
    *,
    warmup_count: int = 0,
    enforce_reference_gpu_for_latency: bool = True,
) -> list[EfficiencySummary]:
    """Group a predictions.jsonl file by (model, precision, hardware) and summarize."""
    records = load_predictions_jsonl(path)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        key = (
            str(rec.get("model", "unknown")),
            str(rec.get("precision", "unknown")),
            extract_hardware_label(rec.get("hardware")),
        )
        grouped[key].append(rec)

    summaries: list[EfficiencySummary] = []
    for (model_name, precision_name, _hw), group_records in sorted(grouped.items()):
        summaries.append(
            compute_efficiency_summary(
                group_records,
                model=model_name,
                precision=precision_name,
                warmup_count=warmup_count,
                enforce_reference_gpu_for_latency=enforce_reference_gpu_for_latency,
            )
        )
    return summaries


def _fmt_opt(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "not measured"
    return f"{value:.2f}{suffix}"


def format_efficiency_markdown(summaries: Sequence[EfficiencySummary]) -> str:
    """Format efficiency summaries as a Markdown table (reporting 'not measured' when omitted)."""
    header = (
        "| Model | Precision | Hardware | n | Load (s) | Peak VRAM (MB) | "
        "p50 (s) | p95 (s) | tok/s | Mean In | Mean Out | Errors |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [header]
    for s in summaries:
        rows.append(
            f"| {s.model} | {s.precision} | {s.hardware_label} | {s.n_samples} | "
            f"{_fmt_opt(s.load_time_s)} | {_fmt_opt(s.peak_vram_mb)} | "
            f"{_fmt_opt(s.latency_p50_s)} | {_fmt_opt(s.latency_p95_s)} | "
            f"{_fmt_opt(s.tokens_per_sec)} | {s.mean_tokens_in:.1f} | "
            f"{s.mean_tokens_out:.1f} | {s.error_count} |"
        )
    return "\n".join(rows)


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize local efficiency from predictions.jsonl."
    )
    parser.add_argument("predictions", type=Path, help="Path to predictions.jsonl")
    parser.add_argument("--warmup", type=int, default=0, help="Leading warm-up records to exclude")
    parser.add_argument(
        "--allow-non-3070-latency",
        action="store_true",
        help="Include latency even when hardware is not the reference RTX 3070",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    summaries = summarize_predictions_jsonl(
        args.predictions,
        warmup_count=args.warmup,
        enforce_reference_gpu_for_latency=not args.allow_non_3070_latency,
    )
    print(format_efficiency_markdown(summaries))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps([s.to_dict() for s in summaries], indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    _cli()
