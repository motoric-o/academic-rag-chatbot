"""CPU-safe unit tests for Member A's LocalHFAdapter and local model YAML configs."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from regqa.models.base import GenerationConfig
from regqa.models.local_hf import (
    LocalHFAdapter,
    LocalHFConfig,
    collect_local_hardware_info,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_CONFIGS = sorted((REPO_ROOT / "configs" / "models").glob("local_*.yaml"))


class _FakeTensor:
    def __init__(self, data: list[int] | list[list[int]], device: str = "cpu") -> None:
        self.data = data
        self.device = device
        if data and isinstance(data[0], list):
            self.shape = (len(data), len(data[0]))
        else:
            self.shape = (len(data),)

    def to(self, device: Any) -> _FakeTensor:
        return _FakeTensor(self.data, device=str(device))

    def __getitem__(self, idx: Any) -> _FakeTensor:
        sliced = self.data[idx]
        if isinstance(sliced, list):
            return _FakeTensor(sliced, device=self.device)
        return _FakeTensor([sliced], device=self.device)


class _FakeTokenizer:
    def __init__(self, *, support_enable_thinking: bool = True) -> None:
        self.pad_token_id: int | None = None
        self.eos_token_id = 2
        self.support_enable_thinking = support_enable_thinking
        self.chat_template_calls: list[dict[str, Any]] = []
        self.decoded_slices: list[list[int]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> str:
        if not self.support_enable_thinking and "enable_thinking" in kwargs:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        self.chat_template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                **kwargs,
            }
        )
        return "FORMATTED_PROMPT"

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, _FakeTensor]:
        assert return_tensors == "pt"
        return {
            "input_ids": _FakeTensor([[10, 11, 12, 13]]),
            "attention_mask": _FakeTensor([[1, 1, 1, 1]]),
        }

    def decode(self, token_ids: _FakeTensor, skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens is True
        self.decoded_slices.append(token_ids.data)  # type: ignore[arg-type]
        return "  Minimum GPA is 2.0. Sources: [1]  "


class _FakeModel:
    def __init__(
        self,
        *,
        device: str = "cuda:0",
        commit_hash: str = "abc123def",
        raise_on_generate: BaseException | None = None,
    ) -> None:
        self.device = device
        self.config = SimpleNamespace(_commit_hash=commit_hash)
        self.eval_called = False
        self.generate_calls: list[dict[str, Any]] = []
        self._raise_on_generate = raise_on_generate

    def eval(self) -> _FakeModel:
        self.eval_called = True
        return self

    def generate(self, **kwargs: Any) -> _FakeTensor:
        self.generate_calls.append(kwargs)
        if self._raise_on_generate is not None:
            raise self._raise_on_generate
        # 4 prompt tokens + 3 generated tokens
        return _FakeTensor([[10, 11, 12, 13, 101, 102, 103]], device=self.device)


def _make_fake_runtime(
    *,
    cuda_available: bool = True,
    support_enable_thinking: bool = True,
    raise_on_generate: BaseException | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    state: dict[str, Any] = {
        "sync_count": 0,
        "reset_peak_count": 0,
        "empty_cache_count": 0,
        "bnb_calls": [],
        "tokenizer_from_pretrained": [],
        "model_from_pretrained": [],
    }

    class _FakeCudaOOM(RuntimeError):
        pass

    def _sync() -> None:
        state["sync_count"] += 1

    def _reset_peak() -> None:
        state["reset_peak_count"] += 1

    def _empty_cache() -> None:
        state["empty_cache_count"] += 1

    fake_cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        synchronize=_sync,
        reset_peak_memory_stats=_reset_peak,
        max_memory_allocated=lambda: 1536 * 1024 * 1024,  # 1536 MB
        empty_cache=_empty_cache,
        get_device_name=lambda _idx: "NVIDIA GeForce RTX 3070",
        get_device_properties=lambda _idx: SimpleNamespace(total_memory=8192 * 1024 * 1024),
        OutOfMemoryError=_FakeCudaOOM,
    )
    fake_torch = SimpleNamespace(
        __version__="2.6.0+cu124",
        version=SimpleNamespace(cuda="12.4"),
        float16="torch.float16",
        bfloat16="torch.bfloat16",
        float32="torch.float32",
        cuda=fake_cuda,
        inference_mode=lambda: nullcontext(),
    )

    fake_tokenizer = _FakeTokenizer(support_enable_thinking=support_enable_thinking)
    fake_model = _FakeModel(
        device="cuda:0" if cuda_available else "cpu",
        raise_on_generate=raise_on_generate,
    )
    state["tokenizer"] = fake_tokenizer
    state["model"] = fake_model

    class _FakeBnBConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            state["bnb_calls"].append(kwargs)

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> _FakeTokenizer:
            state["tokenizer_from_pretrained"].append((model_id, kwargs))
            return fake_tokenizer

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> _FakeModel:
            state["model_from_pretrained"].append((model_id, kwargs))
            return fake_model

    fake_transformers = SimpleNamespace(
        AutoTokenizer=_FakeAutoTokenizer,
        AutoModelForCausalLM=_FakeAutoModel,
        BitsAndBytesConfig=_FakeBnBConfig,
    )
    return fake_torch, fake_transformers, state


def test_all_local_yaml_configs_load_cleanly() -> None:
    assert len(LOCAL_MODEL_CONFIGS) == 10
    for path in LOCAL_MODEL_CONFIGS:
        cfg = LocalHFConfig.from_yaml(path)
        assert cfg.hf_model_id.startswith("Qwen/Qwen3-")
        assert cfg.precision in {"fp16", "8bit", "4bit"}
        assert cfg.enable_thinking is False
        gen_cfg = cfg.default_generation_config()
        assert gen_cfg.do_sample is False
        assert gen_cfg.temperature == 0.0
        assert gen_cfg.max_new_tokens == 256


def test_local_hf_config_validation_and_aliases() -> None:
    cfg = LocalHFConfig(name="m", hf_model_id="Qwen/Qwen3-0.6B", precision="4-bit")
    assert cfg.precision == "4bit"
    assert cfg.quantization_method == "bitsandbytes_nf4"

    with pytest.raises(ValueError, match="Unsupported precision"):
        LocalHFConfig(name="m", hf_model_id="Qwen/Qwen3-0.6B", precision="int2")

    with pytest.raises(ValueError, match="fp16 precision must not specify"):
        LocalHFConfig(
            name="m",
            hf_model_id="Qwen/Qwen3-0.6B",
            precision="fp16",
            quantization={"method": "bitsandbytes_nf4"},
        )

    with pytest.raises(ValueError, match="Unsupported bnb_4bit_quant_type"):
        LocalHFConfig(
            name="m",
            hf_model_id="Qwen/Qwen3-0.6B",
            precision="4bit",
            quantization={"method": "bitsandbytes_nf4", "bnb_4bit_quant_type": "invalid"},
        )


def test_load_fp16_8bit_4bit_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_versions = {
        "bitsandbytes": "0.45.0",
        "accelerate": "1.2.0",
        "transformers": "5.23.0",
    }
    monkeypatch.setattr(
        "regqa.models.local_hf._optional_version",
        lambda name: fake_versions.get(name),
    )

    # 1. fp16 on CUDA
    fake_torch, fake_transformers, state = _make_fake_runtime(cuda_available=True)
    adapter = LocalHFAdapter(
        LocalHFConfig(
            name="qwen3-0.6b-fp16",
            hf_model_id="Qwen/Qwen3-0.6B",
            precision="fp16",
            revision="rev1",
        )
    )
    monkeypatch.setattr(adapter, "_import_runtime", lambda: (fake_torch, fake_transformers))
    load_time = adapter.load()
    assert load_time >= 0.0
    assert adapter.load_peak_vram_mb == 1536.0
    assert state["model"].eval_called is True
    _, model_kwargs = state["model_from_pretrained"][0]
    assert model_kwargs["dtype"] == "torch.float16"
    assert model_kwargs["revision"] == "rev1"
    assert "quantization_config" not in model_kwargs

    # 2. 8bit on CUDA
    fake_torch, fake_transformers, state = _make_fake_runtime(cuda_available=True)
    adapter8 = LocalHFAdapter(
        LocalHFConfig(
            name="qwen3-1.7b-8bit",
            hf_model_id="Qwen/Qwen3-1.7B",
            precision="8bit",
            quantization={"method": "bitsandbytes_int8"},
        )
    )
    monkeypatch.setattr(adapter8, "_import_runtime", lambda: (fake_torch, fake_transformers))
    adapter8.load()
    assert state["bnb_calls"] == [{"load_in_8bit": True}]

    # 3. 4bit NF4 on CUDA
    fake_torch, fake_transformers, state = _make_fake_runtime(cuda_available=True)
    adapter4 = LocalHFAdapter(
        LocalHFConfig.from_yaml(REPO_ROOT / "configs" / "models" / "local_qwen3_0.6b_4bit.yaml")
    )
    monkeypatch.setattr(adapter4, "_import_runtime", lambda: (fake_torch, fake_transformers))
    adapter4.load()
    assert state["bnb_calls"] == [
        {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "torch.float16",
        }
    ]


def test_quantized_load_fails_clearly_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch, fake_transformers, _ = _make_fake_runtime(cuda_available=False)
    adapter = LocalHFAdapter(
        LocalHFConfig(name="qwen3-0.6b-4bit", hf_model_id="Qwen/Qwen3-0.6B", precision="4bit")
    )
    monkeypatch.setattr(adapter, "_import_runtime", lambda: (fake_torch, fake_transformers))
    with pytest.raises(RuntimeError, match="requires CUDA"):
        adapter.load()


def test_generate_greedy_thinking_disabled_and_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch, fake_transformers, state = _make_fake_runtime(cuda_available=True)
    adapter = LocalHFAdapter(
        LocalHFConfig(name="qwen3-0.6b-fp16", hf_model_id="Qwen/Qwen3-0.6B", precision="fp16")
    )
    monkeypatch.setattr(adapter, "_import_runtime", lambda: (fake_torch, fake_transformers))

    result = adapter.generate(
        messages=[
            {"role": "system", "content": "Be strict."},
            {"role": "user", "content": "Minimum GPA?"},
        ],
        gen_config=GenerationConfig(max_new_tokens=64, do_sample=False, temperature=0.0),
    )

    assert result.error is None
    assert result.raw_output == "Minimum GPA is 2.0. Sources: [1]"
    assert result.tokens_in == 4
    assert result.tokens_out == 3
    assert result.peak_vram_mb == 1536.0
    assert result.hardware["device"] == "cuda"
    assert result.hardware["revision"] == "abc123def"

    assert state["tokenizer"].chat_template_calls[0]["enable_thinking"] is False
    gen_call = state["model"].generate_calls[0]
    assert gen_call["do_sample"] is False
    assert gen_call["temperature"] is None
    assert gen_call["top_p"] is None
    assert gen_call["top_k"] is None
    assert gen_call["max_new_tokens"] == 64
    assert state["tokenizer"].decoded_slices == [[101, 102, 103]]


def test_generate_handles_oom_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch, fake_transformers, state = _make_fake_runtime(cuda_available=True)
    oom_exc = fake_torch.cuda.OutOfMemoryError("CUDA out of memory")
    state["model"]._raise_on_generate = oom_exc

    adapter = LocalHFAdapter(
        LocalHFConfig(name="qwen3-4b-fp16", hf_model_id="Qwen/Qwen3-4B", precision="fp16")
    )
    monkeypatch.setattr(adapter, "_import_runtime", lambda: (fake_torch, fake_transformers))

    result = adapter.generate(
        messages=[{"role": "user", "content": "Hello"}],
        gen_config=GenerationConfig(),
    )
    assert result.error is not None
    assert result.error.startswith("OOM:")
    assert result.raw_output == ""
    assert result.tokens_out == 0
    assert state["empty_cache_count"] >= 1


def test_collect_local_hardware_info_cpu_safe() -> None:
    info = collect_local_hardware_info(hf_model_id="Qwen/Qwen3-0.6B")
    assert info["device"] in {"cpu", "cuda"}
    assert info["hf_model_id"] == "Qwen/Qwen3-0.6B"
    assert "platform" in info
