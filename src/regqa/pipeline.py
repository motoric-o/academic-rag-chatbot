"""End-to-end experiment execution pipeline.

Owner: Member C (Francisco Valentino; see docs/project_spec.md §0.2, §0.3 & §13).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from regqa.generation.parser import ParsedOutput, parse_output
from regqa.generation.prompt_builder import PromptBuilder
from regqa.models.base import GenerationConfig, GenerationResult, ModelAdapter


@dataclass
class PipelineOutput:
    """Consolidated output of a single pipeline execution."""

    question: str
    prompt_hash: str
    gen_result: GenerationResult
    parsed: ParsedOutput
    timestamp: str

    def to_prediction_record(self, qid: str, run_id: str) -> dict[str, Any]:
        """Convert to the standard predictions.jsonl schema specified in §13."""
        return {
            "qid": qid,
            "model": self.gen_result.model,
            "precision": self.gen_result.precision,
            "run_id": run_id,
            "prompt_hash": self.prompt_hash,
            "raw_output": self.gen_result.raw_output,
            "parsed": self.parsed.to_dict(),
            "tokens_in": self.gen_result.tokens_in,
            "tokens_out": self.gen_result.tokens_out,
            "latency_s": self.gen_result.latency_s,
            "peak_vram_mb": self.gen_result.peak_vram_mb,
            "hardware": self.gen_result.hardware,
            "error": self.gen_result.error,
            "timestamp": self.timestamp,
        }


class RAGPipeline:
    """Orchestrates prompt formatting, adapter inference, and output parsing."""

    def __init__(
        self,
        adapter: ModelAdapter,
        prompt_builder: PromptBuilder | None = None,
        gen_config: GenerationConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.gen_config = gen_config or GenerationConfig(temperature=0.0, max_new_tokens=256)

    def run_query(
        self,
        question: str,
        context_chunks: list[dict[str, Any]] | None = None,
        is_closed_book: bool = False,
    ) -> PipelineOutput:
        """Run a single query through prompt building, model generation, and parsing."""
        if is_closed_book or context_chunks is None:
            messages = self.prompt_builder.build_closed_book_messages(question)
        else:
            messages = self.prompt_builder.build_rag_messages(question, context_chunks)

        prompt_hash = self.prompt_builder.compute_prompt_hash(messages)
        gen_result = self.adapter.generate(messages, self.gen_config)

        if gen_result.error:
            parsed = parse_output(None)
        else:
            parsed = parse_output(gen_result.raw_output)

        now_str = datetime.now(UTC).isoformat()
        return PipelineOutput(
            question=question,
            prompt_hash=prompt_hash,
            gen_result=gen_result,
            parsed=parsed,
            timestamp=now_str,
        )
