"""Hosted OpenAI-compatible adapter for Groq and cloud endpoints.

Owner: Member C (Francisco Valentino; see docs/project_spec.md §7.2).
"""

from __future__ import annotations

import os
import time
from typing import Any

from regqa.models.base import GenerationConfig, GenerationResult, ModelAdapter
from regqa.models.rate_limiter import PersistentRateLimiter
from regqa.utils.hardware_info import collect_hardware_info
from regqa.utils.logging import get_logger

logger = get_logger("regqa.models.hosted")


class HostedOpenAICompatAdapter(ModelAdapter):
    """Adapter for hosted open-weight models via Groq's OpenAI-compatible API."""

    def __init__(
        self,
        model_id: str = "openai/gpt-oss-20b",
        base_url: str = "https://api.groq.com/openai/v1",
        api_key_env: str = "GROQ_API_KEY",
        api_key: str | None = None,
        precision: str = "hosted",
        rate_limits: dict[str, int] | None = None,
        reasoning_effort: str = "low",
        state_dir: str = "data/cache",
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url
        self.precision = precision
        self.reasoning_effort = reasoning_effort
        self.api_key = api_key or os.getenv(api_key_env, "")

        # Configure rate limiter per model
        rl = rate_limits or {"rpm": 30, "tpm": 8000, "rpd": 1000, "tpd": 200000}
        sanitized_id = model_id.replace("/", "_")
        state_path = f"{state_dir}/rate_limit_{sanitized_id}.json"
        self.rate_limiter = PersistentRateLimiter(
            rpm=rl.get("rpm", 30),
            tpm=rl.get("tpm", 8000),
            rpd=rl.get("rpd", 1000),
            tpd=rl.get("tpd", 200000),
            state_path=state_path,
        )

        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    f"API key missing for hosted model '{self.model_id}'. "
                    f"Please set GROQ_API_KEY in .env or pass api_key."
                )
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=60.0,
            )
        return self._client

    def generate(
        self,
        messages: list[dict[str, str]],
        gen_config: GenerationConfig,
    ) -> GenerationResult:
        """Call Groq API with rate limiting, retry backoff, and telemetry capture."""
        estimated_prompt_tokens = sum(len(m.get("content", "").split()) for m in messages) * 2
        estimated_tokens = estimated_prompt_tokens + gen_config.max_new_tokens

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                self.rate_limiter.acquire(estimated_tokens)
                client = self._get_client()

                kwargs: dict[str, Any] = {
                    "model": self.model_id,
                    "messages": messages,
                    "temperature": gen_config.temperature,
                    "max_tokens": gen_config.max_new_tokens,
                }
                # For models with reasoning support (like gpt-oss)
                if "gpt-oss" in self.model_id.lower() and self.reasoning_effort:
                    kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}

                start_time = time.perf_counter()
                response = client.chat.completions.create(**kwargs)
                latency = time.perf_counter() - start_time

                choice = response.choices[0]
                raw_output = choice.message.content or ""
                usage = response.usage
                tokens_in = usage.prompt_tokens if usage else estimated_prompt_tokens
                tokens_out = usage.completion_tokens if usage else len(raw_output.split())

                self.rate_limiter.record_usage(tokens_in, tokens_out)

                return GenerationResult(
                    raw_output=raw_output,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    latency_s=round(latency, 4),
                    model=self.model_id,
                    precision=self.precision,
                    hardware=collect_hardware_info(
                        device="hosted",
                        extra={"provider": "groq", "base_url": self.base_url},
                    ),
                    peak_vram_mb=None,
                    error=None,
                )

            except Exception as e:
                err_str = str(e)
                last_error = err_str
                # Check for rate limit 429
                if "429" in err_str or "rate_limit" in err_str.lower():
                    logger.warning(f"Groq 429 rate limit hit for {self.model_id}. Backing off... (attempt {attempt+1})")
                    self.rate_limiter.handle_429(retry_after_s=5.0 * (attempt + 1))
                else:
                    logger.warning(f"Error calling {self.model_id}: {err_str} (attempt {attempt+1})")
                    time.sleep(2.0 * (attempt + 1))

        # If all retries failed
        return GenerationResult(
            raw_output="",
            tokens_in=0,
            tokens_out=0,
            latency_s=0.0,
            model=self.model_id,
            precision=self.precision,
            hardware=collect_hardware_info(device="hosted"),
            error=last_error or "Unknown failure after retries",
        )
