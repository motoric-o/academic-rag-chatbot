"""Scaling fit: correctness vs log10(parameters) with bootstrap CI.

Owner: Member A (see docs/project_spec.md §0.2, §11.6).
Note: With only 4 model sizes, report slope + bootstrap CI and do NOT claim a 'scaling law'.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScalingPoint:
    """A single model evaluation point for log-linear scaling analysis."""

    model: str
    params_b: float
    metric_value: float
    active_params_b: float | None = None
    precision: str | None = None
    condition: str | None = None

    def param_for_axis(self, axis: str) -> float:
        if axis == "active":
            val = self.active_params_b if self.active_params_b is not None else self.params_b
        elif axis == "total":
            val = self.params_b
        else:
            raise ValueError(f"Unsupported param_axis {axis!r}; expected 'total' or 'active'")
        if val <= 0:
            raise ValueError(f"Parameter count must be > 0, got {val} for {self.model}")
        return float(val)


@dataclass(frozen=True)
class ScalingFitResult:
    """Empirical OLS slope of metric vs. log10(parameters) with bootstrap CI."""

    metric_name: str
    param_axis: str
    condition: str
    n_points: int
    slope: float
    intercept: float
    r_squared: float
    slope_ci_low: float | None
    slope_ci_high: float | None
    ci_level: float
    n_resamples: int
    seed: int
    is_inconclusive: bool
    caveat: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ols_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = slope * x + intercept and return (slope, intercept, r_squared)."""
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    dx = x - x_mean
    dy = y - y_mean
    denom = float(np.sum(dx * dx))
    if denom <= 0.0:
        raise ValueError("All x values are identical; cannot fit a slope")
    slope = float(np.sum(dx * dy) / denom)
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum(dy * dy))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 1.0
    return slope, intercept, r_squared


def _make_caveat(n_points: int) -> str:
    return (
        f"Empirical log10(params) slope over {n_points} model sizes only — "
        "do not interpret or claim as a general scaling law (spec §11.6)."
    )


def fit_scaling_slope(
    points: Sequence[ScalingPoint],
    *,
    metric_name: str,
    param_axis: str = "total",
    condition: str = "L-uniform",
    n_resamples: int = 2000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> ScalingFitResult:
    """Fit OLS slope vs. log10(params) with point-level bootstrap CI."""
    if len(points) < 2:
        raise ValueError("At least 2 points are required to fit a scaling slope")
    if n_resamples < 1000:
        raise ValueError("n_resamples must be >= 1000 per spec §11.6")

    x = np.array([np.log10(p.param_for_axis(param_axis)) for p in points], dtype=float)
    y = np.array([float(p.metric_value) for p in points], dtype=float)
    slope, intercept, r_squared = _ols_fit(x, y)

    rng = np.random.default_rng(seed)
    boot_slopes: list[float] = []
    n = len(points)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        bx = x[idx]
        if float(np.std(bx)) == 0.0:
            continue
        b_slope, _, _ = _ols_fit(bx, y[idx])
        boot_slopes.append(b_slope)

    alpha = (1.0 - ci_level) / 2.0
    if boot_slopes:
        ci_low = round(float(np.quantile(boot_slopes, alpha)), 6)
        ci_high = round(float(np.quantile(boot_slopes, 1.0 - alpha)), 6)
        inconclusive = ci_low <= 0.0 <= ci_high
    else:
        ci_low = ci_high = None
        inconclusive = True

    return ScalingFitResult(
        metric_name=metric_name,
        param_axis=param_axis,
        condition=condition,
        n_points=n,
        slope=round(slope, 6),
        intercept=round(intercept, 6),
        r_squared=round(r_squared, 6),
        slope_ci_low=ci_low,
        slope_ci_high=ci_high,
        ci_level=ci_level,
        n_resamples=n_resamples,
        seed=seed,
        is_inconclusive=inconclusive,
        caveat=_make_caveat(n),
    )


def fit_scaling_slope_from_per_question(
    per_model_scores: Sequence[tuple[str, float, Sequence[float]]],
    *,
    metric_name: str,
    param_axis: str = "total",
    condition: str = "L-uniform",
    n_resamples: int = 2000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> ScalingFitResult:
    """Paired question-level bootstrap for OLS slope vs. log10(params) (§11.6).

    Each tuple in per_model_scores is (model_name, params_b, aligned_question_scores).
    """
    if len(per_model_scores) < 2:
        raise ValueError("At least 2 models are required to fit a scaling slope")
    if n_resamples < 1000:
        raise ValueError("n_resamples must be >= 1000 per spec §11.6")

    lengths = {len(scores) for _, _, scores in per_model_scores}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError("All models must have identical, non-empty aligned question score lists")

    n_questions = next(iter(lengths))
    x = np.array([np.log10(float(params)) for _, params, _ in per_model_scores], dtype=float)
    matrix = np.array([list(scores) for _, _, scores in per_model_scores], dtype=float)
    y = np.mean(matrix, axis=1)
    slope, intercept, r_squared = _ols_fit(x, y)

    rng = np.random.default_rng(seed)
    boot_slopes = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        q_idx = rng.integers(0, n_questions, size=n_questions)
        by = np.mean(matrix[:, q_idx], axis=1)
        boot_slopes[i], _, _ = _ols_fit(x, by)

    alpha = (1.0 - ci_level) / 2.0
    ci_low = round(float(np.quantile(boot_slopes, alpha)), 6)
    ci_high = round(float(np.quantile(boot_slopes, 1.0 - alpha)), 6)
    inconclusive = ci_low <= 0.0 <= ci_high

    return ScalingFitResult(
        metric_name=metric_name,
        param_axis=param_axis,
        condition=condition,
        n_points=len(per_model_scores),
        slope=round(slope, 6),
        intercept=round(intercept, 6),
        r_squared=round(r_squared, 6),
        slope_ci_low=ci_low,
        slope_ci_high=ci_high,
        ci_level=ci_level,
        n_resamples=n_resamples,
        seed=seed,
        is_inconclusive=inconclusive,
        caveat=_make_caveat(len(per_model_scores)),
    )


def plot_scaling_fit(
    points: Sequence[ScalingPoint],
    fit: ScalingFitResult,
    output_path: str | Path,
) -> Path:
    """Save a log10(parameters) scaling plot with fitted slope and caveat."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [np.log10(p.param_for_axis(fit.param_axis)) for p in points]
    ys = [p.metric_value for p in points]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(xs, ys, s=55, zorder=3, label="Observed models")
    for p, x_val, y_val in zip(points, xs, ys, strict=True):
        ax.annotate(p.model, (x_val, y_val), textcoords="offset points", xytext=(5, 5), fontsize=8)

    line_x = np.linspace(min(xs), max(xs), 100)
    line_y = fit.slope * line_x + fit.intercept
    ci_label = (
        f"slope={fit.slope:.3f} (95% CI [{fit.slope_ci_low:.3f}, {fit.slope_ci_high:.3f}])"
        if fit.slope_ci_low is not None and fit.slope_ci_high is not None
        else f"slope={fit.slope:.3f}"
    )
    ax.plot(line_x, line_y, linestyle="--", linewidth=1.5, label=ci_label)

    ax.set_xlabel(f"log10({fit.param_axis} parameters in B)")
    ax.set_ylabel(fit.metric_name)
    ax.set_title(f"{fit.metric_name} vs. log10(params) [{fit.condition}]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.text(0.5, 0.01, fit.caveat, ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Fit metric vs. log10(params) with bootstrap CI.")
    parser.add_argument("--input", type=Path, required=True, help="JSON list of point dicts")
    parser.add_argument("--metric", required=True, help="Metric name label")
    parser.add_argument("--axis", choices=["total", "active"], default="total")
    parser.add_argument("--condition", default="L-uniform")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument("--plot", type=Path, help="Optional PNG plot output path")
    args = parser.parse_args()

    raw: list[Mapping[str, Any]] = json.loads(args.input.read_text(encoding="utf-8"))
    points = [
        ScalingPoint(
            model=str(item["model"]),
            params_b=float(item["params_b"]),
            metric_value=float(item["metric_value"]),
            active_params_b=(
                float(item["active_params_b"]) if item.get("active_params_b") is not None else None
            ),
            precision=item.get("precision"),
            condition=item.get("condition"),
        )
        for item in raw
    ]
    result = fit_scaling_slope(
        points,
        metric_name=args.metric,
        param_axis=args.axis,
        condition=args.condition,
        n_resamples=args.resamples,
        seed=args.seed,
    )
    print(json.dumps(result.to_dict(), indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    if args.plot:
        plot_scaling_fit(points, result, args.plot)


if __name__ == "__main__":
    _cli()
