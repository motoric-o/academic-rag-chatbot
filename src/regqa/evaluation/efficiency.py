"""Local efficiency metrics: VRAM, load time, p50/p95 latency, tokens/sec.

Owner: Member A (see docs/project_spec.md §0.2, §0.4, §11.5).
Note: Local latency must be reported ONLY from the RTX 3070 machine; never mix
Colab/Kaggle timings with 3070 timings.
"""

from __future__ import annotations

from dataclasses import dataclass


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
