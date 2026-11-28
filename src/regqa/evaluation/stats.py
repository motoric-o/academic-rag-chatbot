"""Statistical analysis: 95% bootstrap CIs, exact McNemar + Holm correction, flip analysis.

Owner: Member B (see docs/project_spec.md §11.6).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np
from scipy.stats import binomtest


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval for a headline metric (§11.6)."""

    mean: float
    ci_low: float
    ci_high: float
    n: int
    ci_level: float
    n_resamples: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Paired bootstrap CI for the per-question difference (system B - system A)."""

    mean_a: float
    mean_b: float
    mean_diff: float
    ci_low: float
    ci_high: float
    n_paired: int
    is_inconclusive: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class McNemarResult:
    """Exact two-sided McNemar test on paired binary outcomes (§11.6)."""

    n_paired: int
    n11: int
    n10_correct_to_wrong: int
    n01_wrong_to_correct: int
    n00: int
    p_value: float
    p_value_holm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FlipSummary:
    """Per-question flip analysis between two systems vs. repeat instability (§11.6)."""

    n_paired: int
    correct_to_wrong: int
    wrong_to_correct: int
    net_gain: int
    flip_rate: float
    correct_to_wrong_qids: list[str]
    wrong_to_correct_qids: list[str]
    repeat_instability_a: float | None = None
    repeat_instability_b: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Compute a deterministic percentile bootstrap CI (>= 1,000 resamples, §11.6)."""
    if n_resamples < 1000:
        raise ValueError("n_resamples must be >= 1000 per spec §11.6")
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("values must be non-empty")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = np.mean(arr[idx], axis=1)
    alpha = (1.0 - ci_level) / 2.0
    return BootstrapCI(
        mean=round(float(np.mean(arr)), 4),
        ci_low=round(float(np.quantile(means, alpha)), 4),
        ci_high=round(float(np.quantile(means, 1.0 - alpha)), 4),
        n=int(arr.size),
        ci_level=ci_level,
        n_resamples=n_resamples,
        seed=seed,
    )


def paired_bootstrap_diff(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_resamples: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> PairedBootstrapResult:
    """Paired bootstrap CI for mean(scores_b - scores_a)."""
    if len(scores_a) != len(scores_b) or not scores_a:
        raise ValueError("scores_a and scores_b must have identical non-empty lengths")
    if n_resamples < 1000:
        raise ValueError("n_resamples must be >= 1000 per spec §11.6")

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diffs = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_resamples, diffs.size))
    boot_diffs = np.mean(diffs[idx], axis=1)
    alpha = (1.0 - ci_level) / 2.0
    ci_low = round(float(np.quantile(boot_diffs, alpha)), 4)
    ci_high = round(float(np.quantile(boot_diffs, 1.0 - alpha)), 4)
    return PairedBootstrapResult(
        mean_a=round(float(np.mean(a)), 4),
        mean_b=round(float(np.mean(b)), 4),
        mean_diff=round(float(np.mean(diffs)), 4),
        ci_low=ci_low,
        ci_high=ci_high,
        n_paired=int(diffs.size),
        is_inconclusive=ci_low <= 0.0 <= ci_high,
    )


def exact_mcnemar(
    binary_a: Sequence[int | bool],
    binary_b: Sequence[int | bool],
) -> McNemarResult:
    """Exact two-sided McNemar test via binomial test on discordant pairs (§11.6)."""
    if len(binary_a) != len(binary_b) or not binary_a:
        raise ValueError("binary_a and binary_b must have identical non-empty lengths")

    n11 = n10 = n01 = n00 = 0
    for a, b in zip(binary_a, binary_b, strict=True):
        ba, bb = bool(a), bool(b)
        if ba and bb:
            n11 += 1
        elif ba and not bb:
            n10 += 1
        elif not ba and bb:
            n01 += 1
        else:
            n00 += 1

    discordant = n10 + n01
    if discordant == 0:
        p_val = 1.0
    else:
        p_val = float(binomtest(min(n10, n01), discordant, p=0.5, alternative="two-sided").pvalue)

    return McNemarResult(
        n_paired=len(binary_a),
        n11=n11,
        n10_correct_to_wrong=n10,
        n01_wrong_to_correct=n01,
        n00=n00,
        p_value=round(p_val, 6),
    )


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (§11.6)."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        adj = min(1.0, (m - rank) * float(p_val))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = round(running_max, 6)
    return adjusted


def _repeat_instability(repeat_matrix: Sequence[Sequence[int | bool]] | None) -> float | None:
    if not repeat_matrix:
        return None
    multi = [row for row in repeat_matrix if len(row) >= 2]
    if not multi:
        return None
    unstable = sum(1 for row in multi if len({bool(x) for x in row}) > 1)
    return round(unstable / len(multi), 4)


def flip_analysis(
    binary_a: Sequence[int | bool],
    binary_b: Sequence[int | bool],
    *,
    qids: Sequence[str] | None = None,
    repeat_matrix_a: Sequence[Sequence[int | bool]] | None = None,
    repeat_matrix_b: Sequence[Sequence[int | bool]] | None = None,
) -> FlipSummary:
    """Count per-question correct->wrong and wrong->correct flips between two systems."""
    if len(binary_a) != len(binary_b) or not binary_a:
        raise ValueError("binary_a and binary_b must have identical non-empty lengths")
    labels = list(qids) if qids is not None else [str(i) for i in range(len(binary_a))]
    if len(labels) != len(binary_a):
        raise ValueError("qids length must match binary_a length")

    c2w: list[str] = []
    w2c: list[str] = []
    for qid, a, b in zip(labels, binary_a, binary_b, strict=True):
        ba, bb = bool(a), bool(b)
        if ba and not bb:
            c2w.append(qid)
        elif not ba and bb:
            w2c.append(qid)

    n = len(binary_a)
    return FlipSummary(
        n_paired=n,
        correct_to_wrong=len(c2w),
        wrong_to_correct=len(w2c),
        net_gain=len(w2c) - len(c2w),
        flip_rate=round((len(c2w) + len(w2c)) / n, 4),
        correct_to_wrong_qids=c2w,
        wrong_to_correct_qids=w2c,
        repeat_instability_a=_repeat_instability(repeat_matrix_a),
        repeat_instability_b=_repeat_instability(repeat_matrix_b),
    )


def compare_systems_pairwise(
    system_scores: Mapping[str, Mapping[str, int | bool]],
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run paired bootstrap, exact McNemar + Holm correction, and flip analysis for all pairs."""
    names = sorted(system_scores)
    raw_pairs: list[dict[str, Any]] = []
    p_values: list[float] = []

    for sys_a, sys_b in combinations(names, 2):
        common_qids = sorted(set(system_scores[sys_a]) & set(system_scores[sys_b]))
        if not common_qids:
            continue
        vec_a = [int(bool(system_scores[sys_a][q])) for q in common_qids]
        vec_b = [int(bool(system_scores[sys_b][q])) for q in common_qids]

        boot = paired_bootstrap_diff(vec_a, vec_b, n_resamples=n_resamples, seed=seed)
        mcn = exact_mcnemar(vec_a, vec_b)
        flips = flip_analysis(vec_a, vec_b, qids=common_qids)
        p_values.append(mcn.p_value)
        raw_pairs.append(
            {
                "system_a": sys_a,
                "system_b": sys_b,
                "paired_bootstrap": boot.to_dict(),
                "mcnemar": mcn.to_dict(),
                "flips": flips.to_dict(),
            }
        )

    adjusted = holm_correction(p_values)
    for pair, p_holm in zip(raw_pairs, adjusted, strict=True):
        pair["mcnemar"]["p_value_holm"] = p_holm
    return raw_pairs
