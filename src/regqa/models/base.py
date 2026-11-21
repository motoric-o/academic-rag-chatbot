"""Cross-scope ModelAdapter contract and MockAdapter.

Owner: Member C (defines contract; see docs/project_spec.md §0.3 & §7).
Consumers: Member A (local_hf.py), Member C (hosted_openai_compat.py).
Do not change this interface unilaterally; log changes in docs/CONTRACT_CHANGES.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
    """Model-agnostic generation parameters."""

    temperature: float = 0.0
    do_sample: bool = False
    max_new_tokens: int = 256
    enable_thinking: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Standard output returned by every ModelAdapter.generate() call."""

    raw_output: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    model: str
    precision: str
    hardware: dict[str, Any] = field(default_factory=dict)
    peak_vram_mb: float | None = None
    error: str | None = None


class ModelAdapter(ABC):
    """Abstract interface shared by local and hosted model adapters."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        gen_config: GenerationConfig,
    ) -> GenerationResult:
        """Run a single generation call from chat messages."""


class MockAdapter(ModelAdapter):
    """Deterministic CPU-only mock adapter for unit tests and offline development."""

    def __init__(
        self,
        model: str = "mock-model",
        precision: str = "mock",
        canned_response: str = "NOT_FOUND",
        responses_by_question: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.precision = precision
        self.canned_response = canned_response
        self.responses_by_question = responses_by_question or {}

    def generate(
        self,
        messages: list[dict[str, str]],
        gen_config: GenerationConfig,
    ) -> GenerationResult:
        prompt_tokens = sum(len(m.get("content", "").split()) for m in messages)
        user_msg = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        
        output = self.canned_response
        for q, resp in self.responses_by_question.items():
            if q.lower() in user_msg.lower():
                output = resp
                break

        return GenerationResult(
            raw_output=output,
            tokens_in=prompt_tokens,
            tokens_out=len(output.split()),
            latency_s=0.001,
            model=self.model,
            precision=self.precision,
            hardware={"device": "cpu", "mock": True},
        )
