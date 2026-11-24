"""Unit tests for Member C Week 2 deliverables (RateLimiter, Pipeline, CLI runner, HardwareInfo).

Owner: Member C (Francisco Valentino; see docs/project_spec.md §10 & §13).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from regqa.generation.parser import parse_output
from regqa.models.base import GenerationConfig, MockAdapter
from regqa.models.hosted_openai_compat import HostedOpenAICompatAdapter
from regqa.models.rate_limiter import PersistentRateLimiter, RateLimitExceededError
from regqa.pipeline import RAGPipeline
from regqa.utils.hardware_info import collect_hardware_info
from regqa.utils.io import compute_string_sha256, read_json, write_json
from regqa.utils.logging import get_logger
from regqa.utils.seeds import seed_everything
from scripts.run_experiment import run_experiment


# -----------------------------------------------------------------------------
# Rate Limiter Tests with Simulated Clock (§10, §13)
# -----------------------------------------------------------------------------
def test_rate_limiter_simulated_clock() -> None:
    current_time = 1000.0

    def mock_time() -> float:
        return current_time

    def mock_sleep(seconds: float) -> None:
        nonlocal current_time
        current_time += seconds

    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / "rate_state.json"
        limiter = PersistentRateLimiter(
            rpm=60,
            tpm=6000,
            rpd=10,
            tpd=10000,
            state_path=state_file,
            time_fn=mock_time,
            sleep_fn=mock_sleep,
        )

        # First acquisition should succeed immediately
        limiter.acquire(estimated_tokens=500)
        assert limiter.req_tokens == 59.0
        assert limiter.token_tokens == 5500.0

        limiter.record_usage(actual_tokens_in=300, actual_tokens_out=100)
        assert limiter.daily_requests_used == 1
        assert limiter.daily_tokens_used == 400

        # Advance time by 30 seconds and test refill
        current_time += 30.0
        limiter._refill_buckets()
        assert limiter.req_tokens > 59.0


def test_rate_limiter_handle_429() -> None:
    current_time = 1000.0

    def mock_time() -> float:
        return current_time

    def mock_sleep(seconds: float) -> None:
        nonlocal current_time
        current_time += seconds

    limiter = PersistentRateLimiter(
        rpm=10,
        tpm=1000,
        time_fn=mock_time,
        sleep_fn=mock_sleep,
    )
    limiter.handle_429(retry_after_s=3.0)
    assert current_time == 1003.0


# -----------------------------------------------------------------------------
# Hardware Info & Utils Tests (§13)
# -----------------------------------------------------------------------------
def test_hardware_info_collector() -> None:
    info = collect_hardware_info(device="cpu", extra={"test_key": "val"})
    assert "platform" in info
    assert info["device"] in ("cpu", "cuda")
    assert info["test_key"] == "val"


def test_seed_everything() -> None:
    seed_everything(123)
    # Ensure no exceptions raised


def test_string_sha256() -> None:
    h = compute_string_sha256("test_string")
    assert len(h) == 64


# -----------------------------------------------------------------------------
# RAGPipeline Tests (§0.3, §13)
# -----------------------------------------------------------------------------
def test_rag_pipeline_execution() -> None:
    adapter = MockAdapter(model="mock-rag", canned_response="Test answer.\nSources: [1]")
    pipeline = RAGPipeline(adapter=adapter)

    out = pipeline.run_query(
        question="What is the rule?",
        context_chunks=[{"text": "Sample passage", "header": "[Chapter 1 > Article 1]"}],
    )
    assert out.question == "What is the rule?"
    assert out.parsed.format_ok is True
    assert out.parsed.cited_ids == [1]

    rec = out.to_prediction_record(qid="q001", run_id="test_run")
    assert rec["qid"] == "q001"
    assert rec["model"] == "mock-rag"
    assert rec["run_id"] == "test_run"


# -----------------------------------------------------------------------------
# Resumable CLI Runner Tests (§4, §10, §13)
# -----------------------------------------------------------------------------
def test_run_experiment_mock() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        run_experiment(
            config_path="configs/experiments/e3_local_vs_hosted.yaml",
            run_id="smoke_test_run",
            split_override="dev",
            force_mock=True,
            output_base_dir=tmpdir,
        )

        summary_file = (
            Path(tmpdir)
            / "hosted"
            / "e3_local_vs_hosted"
            / "hosted_groq_gpt_oss_20b"
            / "smoke_test_run"
            / "summary.json"
        )
        assert summary_file.is_file()
        summary = read_json(summary_file)
        assert summary["run_id"] == "smoke_test_run"
        assert summary["total_records"] > 0
