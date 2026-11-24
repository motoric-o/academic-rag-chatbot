"""Unit tests for Member A's efficiency metrics and smoke-suite runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from regqa.evaluation.efficiency import (
    compute_efficiency_summary,
    format_efficiency_markdown,
    is_reference_rtx3070,
    summarize_predictions_jsonl,
)
from regqa.models.base import GenerationConfig, GenerationResult
from regqa.models.local_hf import LocalHFAdapter, LocalHFConfig, run_smoke_suite

REPO_ROOT = Path(__file__).resolve().parents[2]
RTX_3070_HW = {"device": "cuda", "gpu_name": "NVIDIA GeForce RTX 3070"}
COLAB_T4_HW = {"device": "cuda", "gpu_name": "Tesla T4"}


def test_is_reference_rtx3070() -> None:
    assert is_reference_rtx3070(RTX_3070_HW) is True
    assert is_reference_rtx3070(COLAB_T4_HW) is False
    assert is_reference_rtx3070({"device": "cpu", "gpu_name": None}) is False


def test_compute_efficiency_summary_hand_computed() -> None:
    records = [
        # Warm-up record (excluded)
        GenerationResult(
            raw_output="warmup",
            tokens_in=50,
            tokens_out=10,
            latency_s=9.99,
            model="qwen3-0.6b-4bit",
            precision="4bit",
            hardware=RTX_3070_HW,
            peak_vram_mb=900.0,
        ),
        GenerationResult(
            raw_output="a",
            tokens_in=100,
            tokens_out=20,
            latency_s=1.0,
            model="qwen3-0.6b-4bit",
            precision="4bit",
            hardware=RTX_3070_HW,
            peak_vram_mb=1200.0,
        ),
        GenerationResult(
            raw_output="b",
            tokens_in=200,
            tokens_out=40,
            latency_s=2.0,
            model="qwen3-0.6b-4bit",
            precision="4bit",
            hardware=RTX_3070_HW,
            peak_vram_mb=1400.0,
        ),
        # Errored record (excluded from latency, counted in error_count)
        GenerationResult(
            raw_output="",
            tokens_in=200,
            tokens_out=0,
            latency_s=0.5,
            model="qwen3-0.6b-4bit",
            precision="4bit",
            hardware=RTX_3070_HW,
            peak_vram_mb=7900.0,
            error="OOM: CUDA out of memory",
        ),
    ]

    summary = compute_efficiency_summary(records, load_time_s=3.25, warmup_count=1)
    assert summary.n_samples == 2
    assert summary.error_count == 1
    assert summary.warmup_excluded == 1
    assert summary.is_reference_hardware is True
    assert summary.latency_status == "measured"
    assert summary.load_time_s == 3.25
    assert summary.peak_vram_mb == 1400.0
    assert summary.latency_p50_s == pytest.approx(1.5)
    assert summary.latency_p95_s == pytest.approx(1.95)
    assert summary.tokens_per_sec == pytest.approx(60.0 / 3.0)
    assert summary.mean_tokens_in == pytest.approx(150.0)
    assert summary.mean_tokens_out == pytest.approx(30.0)


def test_non_reference_and_mixed_hardware_suppress_latency() -> None:
    t4_records = [
        {
            "model": "qwen3-8b-8bit",
            "precision": "8bit",
            "latency_s": 2.5,
            "tokens_in": 300,
            "tokens_out": 50,
            "peak_vram_mb": 9800.0,
            "hardware": COLAB_T4_HW,
            "error": None,
        }
    ]
    t4_summary = compute_efficiency_summary(t4_records)
    assert t4_summary.is_reference_hardware is False
    assert t4_summary.latency_status == "not_measured_non_reference_gpu"
    assert t4_summary.latency_p50_s is None
    assert t4_summary.latency_p95_s is None
    assert t4_summary.tokens_per_sec is None
    assert t4_summary.peak_vram_mb == 9800.0
    assert "not measured" in format_efficiency_markdown([t4_summary])

    mixed = [
        {**t4_records[0], "hardware": RTX_3070_HW},
        t4_records[0],
    ]
    mixed_summary = compute_efficiency_summary(mixed)
    assert mixed_summary.hardware_label == "mixed"
    assert mixed_summary.latency_status == "mixed_hardware_rejected"
    assert mixed_summary.latency_p50_s is None


def test_summarize_predictions_jsonl_groups_by_model(tmp_path: Path) -> None:
    pred_file = tmp_path / "predictions.jsonl"
    rows = [
        {
            "qid": "q1",
            "model": "qwen3-0.6b-4bit",
            "precision": "4bit",
            "latency_s": 0.5,
            "tokens_in": 100,
            "tokens_out": 20,
            "peak_vram_mb": 1100.0,
            "hardware": RTX_3070_HW,
            "error": None,
        },
        {
            "qid": "q1",
            "model": "qwen3-1.7b-4bit",
            "precision": "4bit",
            "latency_s": 1.0,
            "tokens_in": 100,
            "tokens_out": 25,
            "peak_vram_mb": 2100.0,
            "hardware": RTX_3070_HW,
            "error": None,
        },
    ]
    pred_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    summaries = summarize_predictions_jsonl(pred_file)
    assert [s.model for s in summaries] == ["qwen3-0.6b-4bit", "qwen3-1.7b-4bit"]


def test_run_smoke_suite_writes_contract_predictions(tmp_path: Path) -> None:
    cfg = LocalHFConfig.from_yaml(
        REPO_ROOT / "configs" / "models" / "local_qwen3_0.6b_4bit.yaml"
    )

    class _StubAdapter(LocalHFAdapter):
        def __init__(self, config: LocalHFConfig) -> None:
            super().__init__(config)
            self._model = object()
            self._tokenizer = object()
            self.load_time_s = 1.23
            self.resolved_revision = "rev_smoke_123"
            self.warmup_calls = 0

        def warmup(self, gen_config: GenerationConfig | None = None) -> None:
            self.warmup_calls += 1

        def generate(
            self,
            messages: list[dict[str, str]],
            gen_config: GenerationConfig,
        ) -> GenerationResult:
            return GenerationResult(
                raw_output="Up to three consecutive terms. Sources: [1]",
                tokens_in=120,
                tokens_out=10,
                latency_s=0.25,
                model=self.config.name,
                precision=self.config.precision,
                hardware=RTX_3070_HW,
                peak_vram_mb=1350.0,
            )

    stub = _StubAdapter(cfg)
    result: dict[str, Any] = run_smoke_suite(
        cfg,
        corpus_path=REPO_ROOT / "tests" / "fixtures" / "mini_corpus.json",
        eval_path=REPO_ROOT / "tests" / "fixtures" / "mini_eval.jsonl",
        repeats=2,
        max_questions=3,
        output_dir=tmp_path,
        adapter=stub,
    )

    assert stub.warmup_calls == 1
    pred_lines = Path(result["predictions_path"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(pred_lines) == 6
    sample = json.loads(pred_lines[0])
    required_keys = {
        "qid",
        "model",
        "precision",
        "run_id",
        "prompt_hash",
        "raw_output",
        "parsed",
        "tokens_in",
        "tokens_out",
        "latency_s",
        "hardware",
        "timestamp",
    }
    assert required_keys.issubset(sample.keys())
    assert sample["parsed"]["format_ok"] is True
    assert result["summary"]["n_samples"] == 6
    assert result["summary"]["is_reference_hardware"] is True
    assert result["summary"]["resolved_revision"] == "rev_smoke_123"
