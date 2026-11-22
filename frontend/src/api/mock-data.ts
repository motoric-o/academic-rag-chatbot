/**
 * Canned / mock JSON data for frontend development without a backend.
 * Based on Member B's mini_corpus.json and Member C's OpenAPI schema.
 *
 * Owner: Member D (spec §0.2 — frontend lead).
 * NOTE: All data here is SYNTHETIC_FIXTURE for development only.
 */

import type {
  AskResponse,
  ModelInfo,
  SectionDetailResponse,
} from "./types";

/* ------------------------------------------------------------------ */
/*  Mock models catalog (from Member C's routes.py SUPPORTED_MODELS)   */
/* ------------------------------------------------------------------ */

export const MOCK_MODELS: ModelInfo[] = [
  {
    id: "qwen3-0.6b",
    name: "Qwen3 0.6B (Local)",
    family: "qwen3",
    params_total_b: 0.6,
    params_active_b: 0.6,
    precision: "4-bit",
    provider: "local",
    available_modes: ["replay", "live-local"],
  },
  {
    id: "qwen3-1.7b",
    name: "Qwen3 1.7B (Local)",
    family: "qwen3",
    params_total_b: 1.7,
    params_active_b: 1.7,
    precision: "4-bit",
    provider: "local",
    available_modes: ["replay", "live-local"],
  },
  {
    id: "qwen3-4b",
    name: "Qwen3 4B (Local)",
    family: "qwen3",
    params_total_b: 4.0,
    params_active_b: 4.0,
    precision: "4-bit",
    provider: "local",
    available_modes: ["replay", "live-local"],
  },
  {
    id: "qwen3-8b",
    name: "Qwen3 8B (Local)",
    family: "qwen3",
    params_total_b: 8.2,
    params_active_b: 8.2,
    precision: "4-bit",
    provider: "local",
    available_modes: ["replay", "live-local"],
  },
  {
    id: "openai/gpt-oss-20b",
    name: "GPT-OSS 20B (Groq Hosted MoE)",
    family: "gpt-oss",
    params_total_b: 20.91,
    params_active_b: 3.61,
    precision: "MXFP4",
    provider: "groq",
    available_modes: ["replay", "live-hosted"],
  },
  {
    id: "openai/gpt-oss-120b",
    name: "GPT-OSS 120B (Groq Hosted MoE)",
    family: "gpt-oss",
    params_total_b: 116.83,
    params_active_b: 5.13,
    precision: "MXFP4",
    provider: "groq",
    available_modes: ["replay", "live-hosted"],
  },
  {
    id: "mock-model",
    name: "Mock Adapter (Testing)",
    family: "mock",
    params_total_b: 0.01,
    params_active_b: 0.01,
    precision: "mock",
    provider: "mock",
    available_modes: ["replay"],
  },
];

/* ------------------------------------------------------------------ */
/*  Mock ask response (normal answer with citations)                   */
/* ------------------------------------------------------------------ */

export const MOCK_ASK_RESPONSE: AskResponse = {
  question: "",
  model: "qwen3-4b",
  precision: "4-bit",
  mode: "replay",
  answer_text:
    "Students may take up to three consecutive terms of academic leave. " +
    "A request starts with the student portal, then needs an advisor's approval.",
  cited_sources: [
    {
      passage_idx: 1,
      section_id: "1.1",
      chapter: "Chapter 1",
      article: "Article 1.1 — Academic Leave",
      text:
        "Students may take up to three consecutive terms of academic leave. " +
        "A request starts with the student portal, then needs an advisor's approval.",
    },
  ],
  abstained: false,
  format_ok: true,
  tokens_in: 480,
  tokens_out: 38,
  latency_s: 1.24,
  peak_vram_mb: 2048,
  raw_output:
    "Students may take up to three consecutive terms of academic leave. " +
    "A request starts with the student portal, then needs an advisor's approval.\n\nSources: [1]",
  error: null,
};

/** Mock response when the AI abstains (NOT_FOUND). */
export const MOCK_ABSTAIN_RESPONSE: AskResponse = {
  question: "What is the fee for applying for academic leave?",
  model: "qwen3-4b",
  precision: "4-bit",
  mode: "replay",
  answer_text: "NOT_FOUND",
  cited_sources: [],
  abstained: true,
  format_ok: true,
  tokens_in: 480,
  tokens_out: 3,
  latency_s: 0.42,
  peak_vram_mb: 2048,
  raw_output: "NOT_FOUND",
  error: null,
};

/* ------------------------------------------------------------------ */
/*  Mock sections (from Member B's mini_corpus.json)                   */
/* ------------------------------------------------------------------ */

export const MOCK_SECTIONS: Record<string, SectionDetailResponse> = {
  "1": {
    section_id: "1",
    chapter: "Chapter 1",
    article: "Article 1",
    title: "Example Campus Guide",
    text: "Example Campus provides the following student services.",
  },
  "1.1": {
    section_id: "1.1",
    chapter: "Chapter 1",
    article: "Article 1.1 — Academic Leave",
    title: "Academic Leave",
    text:
      "Students may take up to three consecutive terms of academic leave. " +
      "A request starts with the student portal, then needs an advisor's approval.",
  },
  "1.2": {
    section_id: "1.2",
    chapter: "Chapter 1",
    article: "Article 1.2 — Credit Transfer",
    title: "Credit Transfer",
    text:
      "A transfer request must include the course outline and an official transcript. " +
      "Only courses graded B or higher are eligible.",
  },
  "1.3": {
    section_id: "1.3",
    chapter: "Chapter 1",
    article: "Article 1.3 — Internship",
    title: "Internship",
    text:
      "Students need at least 84 credit units to register for an internship. " +
      "A workplace mentor must sign the weekly log.",
  },
  "2": {
    section_id: "2",
    chapter: "Chapter 2",
    article: "Article 2",
    title: "Example Campus Deadlines",
    text: "The following deadlines apply to internship administration.",
  },
  "2.1": {
    section_id: "2.1",
    chapter: "Chapter 2",
    article: "Article 2.1 — Registration and Reporting",
    title: "Registration and Reporting",
    text:
      "Internship registration closes on April 15. " +
      "The final report is due 14 days after the placement ends.",
  },
};
