// Owner: Member A (see docs/project_spec.md §0.2 & §12.2)
// Compare API client calling POST /api/compare with automatic canned replay fallback.

import type { AskResponse, CompareRequest, CompareResponse } from "../../api";

export interface CompareClientResult {
  data: CompareResponse;
  isFallback: boolean;
}

function isMockMode(): boolean {
  try {
    const globalProcess = (
      globalThis as unknown as { process?: { env?: { NODE_ENV?: string } } }
    ).process;
    if (globalProcess?.env?.NODE_ENV === "test") {
      return true;
    }
    const env = import.meta.env;
    if (
      env?.MODE === "test" ||
      env?.VITE_USE_MOCK === "true" ||
      env?.VITE_USE_MOCK === "1"
    ) {
      return true;
    }
  } catch {
    // ignore
  }
  return false;
}

const SYNTHETIC_CITATION = {
  passage_idx: 1,
  section_id: "1.1",
  chapter: "Chapter 1",
  article: "Article 1.1 — Academic Leave",
  text:
    "Students may take up to three consecutive terms of academic leave. " +
    "A request starts with the student portal, then needs an advisor's approval.",
};

export function buildFallbackCompareResponse(
  req: CompareRequest,
): CompareResponse {
  const lowerQ = req.question.toLowerCase();
  const isUnanswerable =
    lowerQ.includes("fee") || lowerQ.includes("not_found");
  const isFormatDemo = lowerQ.includes("format");
  const isErrorDemo = lowerQ.includes("quota") || lowerQ.includes("error");

  const results: AskResponse[] = req.model_ids.map((modelId, idx) => {
    const isHosted =
      modelId.includes("gpt-oss") || modelId.includes("qwen3.8-27b");
    const precision = isHosted ? "MXFP4" : (req.precision ?? "4-bit");
    const peakVram = isHosted ? null : 1180 + idx * 920;
    const latency = Number((0.45 + idx * 0.32).toFixed(2));

    if (isErrorDemo && idx === 0) {
      return {
        question: req.question,
        model: modelId,
        precision,
        mode: req.mode ?? "replay",
        answer_text: "",
        cited_sources: [],
        abstained: false,
        format_ok: false,
        tokens_in: 480,
        tokens_out: 0,
        latency_s: 0.12,
        peak_vram_mb: peakVram,
        raw_output: "",
        error: {
          code: "QUOTA_EXHAUSTED",
          message: "Daily free-tier token quota reached for this model.",
          details: { fixture: "SYNTHETIC_FIXTURE" },
        },
      };
    }

    if (isUnanswerable) {
      return {
        question: req.question,
        model: modelId,
        precision,
        mode: req.mode ?? "replay",
        answer_text: "NOT_FOUND",
        cited_sources: [],
        abstained: true,
        format_ok: true,
        tokens_in: 480,
        tokens_out: 3,
        latency_s: latency,
        peak_vram_mb: peakVram,
        raw_output: "NOT_FOUND",
        error: null,
      };
    }

    if (isFormatDemo && idx === 0) {
      return {
        question: req.question,
        model: modelId,
        precision,
        mode: req.mode ?? "replay",
        answer_text: "Up to three consecutive terms of academic leave.",
        cited_sources: [],
        abstained: false,
        format_ok: false,
        tokens_in: 480,
        tokens_out: 18,
        latency_s: latency,
        peak_vram_mb: peakVram,
        raw_output: "Up to three consecutive terms of academic leave. (missing Sources line)",
        error: null,
      };
    }

    return {
      question: req.question,
      model: modelId,
      precision,
      mode: req.mode ?? "replay",
      answer_text:
        "Students may take up to three consecutive terms of academic leave via the student portal with advisor approval.",
      cited_sources: [SYNTHETIC_CITATION],
      abstained: false,
      format_ok: true,
      tokens_in: 480,
      tokens_out: 24 + idx * 4,
      latency_s: latency,
      peak_vram_mb: peakVram,
      raw_output:
        "Students may take up to three consecutive terms of academic leave via the student portal with advisor approval.\n\nSources: [1]",
      error: null,
    };
  });

  return { question: req.question, results };
}

export async function compareModels(
  req: CompareRequest,
): Promise<CompareClientResult> {
  if (isMockMode()) {
    return { data: buildFallbackCompareResponse(req), isFallback: true };
  }

  try {
    const res = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!res.ok) {
      throw new Error(`API ${res.status}`);
    }
    const data = (await res.json()) as CompareResponse;
    return { data, isFallback: false };
  } catch {
    return { data: buildFallbackCompareResponse(req), isFallback: true };
  }
}
