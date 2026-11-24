"""Unit tests for Member A's scaling fit & bootstrap CI module (§11.6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from regqa.analysis.scaling_fit import (
    ScalingPoint,
    fit_scaling_slope,
    fit_scaling_slope_from_per_question,
    plot_scaling_fit,
)


def test_fit_scaling_slope_exact_log_linear(tmp_path: Path) -> None:
    # Exact y = 0.2 * log10(params_b) + 0.5
    points = [
        ScalingPoint(model="qwen3-0.6b", params_b=0.1, metric_value=0.3),
        ScalingPoint(model="qwen3-1.7b", params_b=1.0, metric_value=0.5),
        ScalingPoint(model="qwen3-4b", params_b=10.0, metric_value=0.7),
        ScalingPoint(model="qwen3-8b", params_b=100.0, metric_value=0.9),
    ]
    fit = fit_scaling_slope(
        points,
        metric_name="strict_correctness",
        n_resamples=1000,
        seed=42,
    )
    assert fit.slope == pytest.approx(0.2, abs=1e-5)
    assert fit.intercept == pytest.approx(0.5, abs=1e-5)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-5)
    assert fit.is_inconclusive is False
    assert "not interpret or claim as a general scaling law" in fit.caveat

    plot_path = plot_scaling_fit(points, fit, tmp_path / "scaling.png")
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0


def test_paired_per_question_bootstrap_reproducible_and_inconclusive() -> None:
    # Flat scores across sizes -> slope 0, CI includes 0 -> inconclusive
    flat = [
        ("m1", 0.6, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        ("m2", 1.7, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        ("m3", 4.0, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        ("m4", 8.2, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
    ]
    fit_a = fit_scaling_slope_from_per_question(
        flat, metric_name="strict_correctness", n_resamples=1000, seed=7
    )
    fit_b = fit_scaling_slope_from_per_question(
        flat, metric_name="strict_correctness", n_resamples=1000, seed=7
    )
    assert fit_a == fit_b
    assert fit_a.slope == pytest.approx(0.0)
    assert fit_a.is_inconclusive is True


def test_fit_scaling_slope_validates_inputs() -> None:
    with pytest.raises(ValueError, match="n_resamples must be >= 1000"):
        fit_scaling_slope(
            [ScalingPoint("a", 1.0, 0.5), ScalingPoint("b", 2.0, 0.6)],
            metric_name="m",
            n_resamples=100,
        )
