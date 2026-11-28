"""Pydantic request and response schemas (OpenAPI contract).

Owner: Member C (Francisco Valentino; see docs/project_spec.md §12.1).
Consumers: Member D (Frontend API types via openapi.json), Member A (Compare UI).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field("ok", description="Status string")
    mode: str = Field(
        "replay", description="Current operating mode (replay, live-hosted, live-local)"
    )


class ModelInfo(BaseModel):
    """Metadata describing a supported model configuration."""

    id: str = Field(..., description="Unique model identifier, e.g. 'qwen3-0.6b'")
    name: str = Field(..., description="Human-readable display name")
    family: str = Field(..., description="Architecture family, e.g. 'qwen3', 'gpt-oss', 'mock'")
    params_total_b: float | None = Field(None, description="Total parameters in billions")
    params_active_b: float | None = Field(
        None, description="Active parameters per token in billions (for MoE)"
    )
    precision: str = Field("4-bit", description="Quantization precision (4-bit, 8-bit, fp16, mock)")
    provider: str = Field("local", description="Provider location ('local', 'groq', 'mock')")
    available_modes: list[str] = Field(
        default_factory=lambda: ["replay", "live-hosted", "live-local"],
        description="Supported operational modes",
    )


class ModelsListResponse(BaseModel):
    """List of available models."""

    models: list[ModelInfo]


class SourceCitation(BaseModel):
    """Citation details linked to a context passage and original regulation section."""

    passage_idx: int = Field(..., description="1-indexed passage number in prompt context [n]")
    section_id: str = Field(..., description="Canonical section identifier, e.g. 'pasal_12'")
    chapter: str = Field(..., description="Chapter label, e.g. 'Chapter 3'")
    article: str = Field(..., description="Article label, e.g. 'Article 12'")
    text: str = Field(..., description="Passage or paragraph text")


class ApiError(BaseModel):
    """Structured error information for failures."""

    code: str = Field(
        ...,
        description=(
            "Error code: 'QUOTA_EXHAUSTED', 'MODEL_UNAVAILABLE', "
            "'TIMEOUT', 'FORMAT_FAILURE', 'GENERAL_ERROR'"
        ),
    )
    message: str = Field(..., description="Human-readable error explanation")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional context or telemetry"
    )


class AskRequest(BaseModel):
    """Request payload for /api/ask."""

    question: str = Field(..., min_length=1, description="Question text")
    model_id: str = Field("qwen3-0.6b", description="Model ID to query")
    precision: str = Field("4-bit", description="Model precision (4-bit, 8-bit, fp16)")
    top_k: int = Field(4, ge=1, le=10, description="Number of context passages to feed")
    mode: str = Field("replay", description="Execution mode: 'replay', 'live-hosted', 'live-local'")


class AskResponse(BaseModel):
    """Response payload for /api/ask."""

    question: str
    model: str
    precision: str
    mode: str
    answer_text: str
    cited_sources: list[SourceCitation] = Field(default_factory=list)
    abstained: bool = False
    format_ok: bool = True
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    peak_vram_mb: float | None = None
    raw_output: str = ""
    error: ApiError | None = None


class CompareRequest(BaseModel):
    """Request payload for side-by-side comparison (/api/compare)."""

    question: str = Field(..., min_length=1, description="Question text submitted to all models")
    model_ids: list[str] = Field(
        ..., min_length=2, max_length=3, description="2 or 3 model IDs to compare"
    )
    precision: str = Field("4-bit", description="Default precision to use across models")
    mode: str = Field("replay", description="Execution mode")


class CompareResponse(BaseModel):
    """Response payload for side-by-side comparison (/api/compare)."""

    question: str
    results: list[AskResponse]


class SectionDetailResponse(BaseModel):
    """Details of a specific regulation section for the Sources panel."""

    section_id: str
    chapter: str
    article: str
    title: str = ""
    text: str
