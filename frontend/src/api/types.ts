/**
 * TypeScript types derived from Member C's OpenAPI schema (openapi.json).
 * Owner: Member D (consumer); Member C (producer).
 *
 * NOTE: In production these will be auto-generated from openapi.json.
 * This hand-written version is for the Week 1 mock phase only.
 */

/* ------------------------------------------------------------------ */
/*  Request types                                                      */
/* ------------------------------------------------------------------ */

export interface AskRequest {
  question: string;
  model_id?: string;
  precision?: string;
  top_k?: number;
  mode?: "replay" | "live-hosted" | "live-local";
}

export interface CompareRequest {
  question: string;
  model_ids: string[];
  precision?: string;
  mode?: "replay" | "live-hosted" | "live-local";
}

/* ------------------------------------------------------------------ */
/*  Response types                                                     */
/* ------------------------------------------------------------------ */

export interface HealthResponse {
  status: string;
  mode: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  family: string;
  params_total_b: number | null;
  params_active_b: number | null;
  precision: string;
  provider: string;
  available_modes: string[];
}

export interface ModelsListResponse {
  models: ModelInfo[];
}

export interface SourceCitation {
  passage_idx: number;
  section_id: string;
  chapter: string;
  article: string;
  text: string;
}

export interface ApiError {
  code:
    | "QUOTA_EXHAUSTED"
    | "MODEL_UNAVAILABLE"
    | "TIMEOUT"
    | "FORMAT_FAILURE"
    | "GENERAL_ERROR";
  message: string;
  details: Record<string, unknown>;
}

export interface AskResponse {
  question: string;
  model: string;
  precision: string;
  mode: string;
  answer_text: string;
  cited_sources: SourceCitation[];
  abstained: boolean;
  format_ok: boolean;
  tokens_in: number;
  tokens_out: number;
  latency_s: number;
  peak_vram_mb: number | null;
  raw_output: string;
  error: ApiError | null;
}

export interface CompareResponse {
  question: string;
  results: AskResponse[];
}

export interface SectionDetailResponse {
  section_id: string;
  chapter: string;
  article: string;
  title: string;
  text: string;
}
