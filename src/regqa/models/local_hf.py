"""Local Hugging Face + bitsandbytes adapter (Qwen3 family).

Owner: Member A (see docs/project_spec.md §0.2, §7.1, §7.4).
Guards CUDA imports/calls so this module can be imported safely on CPU-only machines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from regqa.models.base import GenerationConfig, GenerationResult, ModelAdapter


@dataclass(frozen=True)
class LocalHFConfig:
    """Configuration loaded from configs/models/local_*.yaml."""

    name: str
    hf_model_id: str
    precision: str  # "fp16" | "8bit" | "4bit"
    revision: str | None = None
    quantization: dict[str, Any] | None = None
    enable_thinking: bool = False


class LocalHFAdapter(ModelAdapter):
    """Local Hugging Face Transformers adapter with optional bitsandbytes quantization."""

    def __init__(self, config: LocalHFConfig) -> None:
        self.config = config
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> float:
        """Load tokenizer and model weights, returning load time in seconds.

        TODO (Member A): implement transformers + bitsandbytes loading,
        pin revision hash, and record peak VRAM on CUDA devices.
        """
        raise NotImplementedError("Member A: implement LocalHFAdapter.load()")

    def generate(
        self,
        messages: list[dict[str, str]],
        gen_config: GenerationConfig,
    ) -> GenerationResult:
        """Generate a response using chat template with thinking disabled by default.

        TODO (Member A): implement deterministic greedy decoding, token counting,
        latency timing, and CUDA peak VRAM tracking.
        """
        raise NotImplementedError("Member A: implement LocalHFAdapter.generate()")
