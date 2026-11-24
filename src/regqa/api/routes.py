"""API routes definition for Academic Regulation QA.

Owner: Member C (Francisco Valentino; see docs/project_spec.md §12.1).
"""

from __future__ import annotations

import os
from fastapi import APIRouter, HTTPException, status

from regqa.api.replay import ReplayService
from regqa.api.schemas import (
    AskRequest,
    AskResponse,
    CompareRequest,
    CompareResponse,
    HealthResponse,
    ModelInfo,
    ModelsListResponse,
    SectionDetailResponse,
)

from regqa.generation.parser import parse_output
from regqa.generation.prompt_builder import PromptBuilder
from regqa.models.base import GenerationConfig
from regqa.models.hosted_openai_compat import HostedOpenAICompatAdapter

router = APIRouter(prefix="/api", tags=["Regulation QA"])
replay_service = ReplayService()
prompt_builder = PromptBuilder()

# Available models catalog (spec §7 & §12.1)
SUPPORTED_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="qwen3-0.6b",
        name="Qwen3 0.6B (Local)",
        family="qwen3",
        params_total_b=0.6,
        params_active_b=0.6,
        precision="4-bit",
        provider="local",
        available_modes=["replay", "live-local"],
    ),
    ModelInfo(
        id="qwen3-1.7b",
        name="Qwen3 1.7B (Local)",
        family="qwen3",
        params_total_b=1.7,
        params_active_b=1.7,
        precision="4-bit",
        provider="local",
        available_modes=["replay", "live-local"],
    ),
    ModelInfo(
        id="qwen3-4b",
        name="Qwen3 4B (Local)",
        family="qwen3",
        params_total_b=4.0,
        params_active_b=4.0,
        precision="4-bit",
        provider="local",
        available_modes=["replay", "live-local"],
    ),
    ModelInfo(
        id="qwen3-8b",
        name="Qwen3 8B (Local)",
        family="qwen3",
        params_total_b=8.2,
        params_active_b=8.2,
        precision="4-bit",
        provider="local",
        available_modes=["replay", "live-local"],
    ),
    ModelInfo(
        id="openai/gpt-oss-20b",
        name="GPT-OSS 20B (Groq Hosted MoE)",
        family="gpt-oss",
        params_total_b=20.91,
        params_active_b=3.61,
        precision="MXFP4",
        provider="groq",
        available_modes=["replay", "live-hosted"],
    ),
    ModelInfo(
        id="openai/gpt-oss-120b",
        name="GPT-OSS 120B (Groq Hosted MoE)",
        family="gpt-oss",
        params_total_b=116.83,
        params_active_b=5.13,
        precision="MXFP4",
        provider="groq",
        available_modes=["replay", "live-hosted"],
    ),
    ModelInfo(
        id="qwen/qwen3.8-27b",
        name="Qwen 27B Preview (Groq Hosted)",
        family="qwen3",
        params_total_b=27.0,
        params_active_b=27.0,
        precision="4-bit",
        provider="groq",
        available_modes=["replay", "live-hosted"],
    ),
    ModelInfo(
        id="mock-model",
        name="Mock Adapter (Testing)",
        family="mock",
        params_total_b=0.01,
        params_active_b=0.01,
        precision="mock",
        provider="mock",
        available_modes=["replay"],
    ),
]


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return system status and current active mode."""
    mode = os.getenv("REGQA_API_MODE", "replay")
    return HealthResponse(status="ok", mode=mode)


@router.get("/models", response_model=ModelsListResponse)
def get_models() -> ModelsListResponse:
    """Return catalog of supported models, parameter scales, and hosting options."""
    return ModelsListResponse(models=SUPPORTED_MODELS)


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """Query the regulation assistant with a question."""
    mode = os.getenv("REGQA_API_MODE", request.mode)

    if mode == "live-hosted":
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            return AskResponse(
                question=request.question,
                model=request.model_id,
                precision=request.precision,
                mode=mode,
                answer_text="",
                error={
                    "code": "API_KEY_MISSING",
                    "message": "GROQ_API_KEY is not set in environment or .env file.",
                    "details": {"mode": mode},
                },
            )
        try:
            adapter = HostedOpenAICompatAdapter(
                model_id=request.model_id,
                api_key=api_key,
                precision=request.precision,
            )
            # Fetch sample context for demo query
            canned_sec = replay_service.get_section("pasal_12")
            chunks = [{"text": canned_sec["text"], "header": f"[{canned_sec['chapter']} > {canned_sec['article']}]"}] if canned_sec else []
            messages = prompt_builder.build_rag_messages(request.question, chunks)
            gen_res = adapter.generate(messages, GenerationConfig(temperature=0.0, max_new_tokens=256))

            if gen_res.error:
                return AskResponse(
                    question=request.question,
                    model=request.model_id,
                    precision=request.precision,
                    mode=mode,
                    answer_text="",
                    error={"code": "GENERATION_ERROR", "message": gen_res.error},
                )

            parsed = parse_output(gen_res.raw_output)
            return AskResponse(
                question=request.question,
                model=request.model_id,
                precision=request.precision,
                mode=mode,
                answer_text=parsed.answer_text,
                cited_sources=[],
                abstained=parsed.abstained,
                format_ok=parsed.format_ok,
                tokens_in=gen_res.tokens_in,
                tokens_out=gen_res.tokens_out,
                latency_s=gen_res.latency_s,
                peak_vram_mb=gen_res.peak_vram_mb,
                raw_output=gen_res.raw_output,
            )
        except Exception as e:
            return AskResponse(
                question=request.question,
                model=request.model_id,
                precision=request.precision,
                mode=mode,
                answer_text="",
                error={"code": "HOSTED_ERROR", "message": str(e)},
            )

    # Default fallback: replay mode
    return replay_service.replay_query(
        question=request.question,
        model_id=request.model_id,
        precision=request.precision,
        top_k=request.top_k,
    )


@router.post("/compare", response_model=CompareResponse)
def compare_models(request: CompareRequest) -> CompareResponse:
    """Compare answers from 2 to 3 models for the same question."""
    results: list[AskResponse] = []
    for model_id in request.model_ids:
        res = replay_service.replay_query(
            question=request.question,
            model_id=model_id,
            precision=request.precision,
        )
        results.append(res)
    return CompareResponse(question=request.question, results=results)


@router.get("/sections/{section_id}", response_model=SectionDetailResponse)
def get_section(section_id: str) -> SectionDetailResponse:
    """Retrieve regulation section details for the Sources panel in frontend."""
    sec = replay_service.get_section(section_id)
    if not sec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Section '{section_id}' not found in cached corpus.",
        )
    return SectionDetailResponse(
        section_id=sec["section_id"],
        chapter=sec["chapter"],
        article=sec["article"],
        title=sec.get("title", ""),
        text=sec["text"],
    )
