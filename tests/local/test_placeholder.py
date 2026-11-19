"""Smoke tests for Member A's local-model setup and shared adapter contract."""

from regqa.models.base import GenerationConfig, MockAdapter
from regqa.models.local_hf import LocalHFAdapter, LocalHFConfig


def test_mock_adapter_contract() -> None:
    adapter = MockAdapter(canned_response="Two semesters. Sources: [1]")
    result = adapter.generate(
        messages=[{"role": "user", "content": "How many semesters?"}],
        gen_config=GenerationConfig(),
    )
    assert result.raw_output == "Two semesters. Sources: [1]"
    assert result.tokens_in > 0
    assert result.tokens_out > 0
    assert result.error is None


def test_local_hf_adapter_cpu_importable() -> None:
    cfg = LocalHFConfig(
        name="qwen3-0.6b-4bit",
        hf_model_id="Qwen/Qwen3-0.6B",
        precision="4bit",
    )
    adapter = LocalHFAdapter(cfg)
    assert adapter.config.precision == "4bit"
    assert adapter.config.enable_thinking is False
