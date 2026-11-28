"""Unit tests for Member C Week 2 deliverables (RateLimiter, Pipeline, CLI runner, HardwareInfo).

Owner: Member C (Francisco Valentino; see docs/project_spec.md §10 & §13).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_experiment import load_adapter, run_experiment  # noqa: E402

from regqa.models.base import MockAdapter  # noqa: E402
from regqa.models.local_hf import LocalHFAdapter  # noqa: E402
from regqa.models.rate_limiter import PersistentRateLimiter  # noqa: E402
from regqa.pipeline import RAGPipeline  # noqa: E402
from regqa.utils.hardware_info import collect_hardware_info  # noqa: E402
from regqa.utils.io import compute_string_sha256, read_json, read_yaml  # noqa: E402
from regqa.utils.seeds import seed_everything  # noqa: E402


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
def test_load_adapter_constructs_local_hf_from_yaml() -> None:
    cfg_path = REPO_ROOT / "configs" / "models" / "local_qwen3_0.6b_4bit.yaml"
    m_cfg = read_yaml(cfg_path)
    adapter = load_adapter(m_cfg, model_cfg_path=cfg_path, force_mock=False)
    assert isinstance(adapter, LocalHFAdapter)
    assert adapter.config.name == "qwen3-0.6b-4bit"
    assert adapter.config.hf_model_id == "Qwen/Qwen3-0.6B"
    assert adapter.config.precision == "4bit"


def test_run_experiment_mock_and_top_k_slice(tmp_path: Path) -> None:
    cache_path = tmp_path / "contexts.jsonl"
    chunks_5 = [
        {"chunk_id": f"c{i}", "section_ids": ["1.1"], "text": f"Passage {i}"}
        for i in range(1, 6)
    ]
    cache_path.write_text(
        json.dumps({"qid": "fixture_01", "chunks": chunks_5}) + "\n",
        encoding="utf-8",
    )

    run_experiment(
        config_path="configs/experiments/e3_local_vs_hosted.yaml",
        run_id="smoke_test_run",
        split_override="dev",
        force_mock=True,
        output_base_dir=tmp_path,
        contexts_cache_path=cache_path,
        top_k_context=4,
    )

    summary_file = (
        tmp_path
        / "hosted"
        / "e3_local_vs_hosted"
        / "hosted_groq_gpt_oss_20b"
        / "smoke_test_run"
        / "summary.json"
    )
    assert summary_file.is_file()
    summary = read_json(summary_file)
    assert summary["run_id"] == "smoke_test_run"
    assert summary["top_k_context"] == 4
    assert summary["total_records"] > 0
