/**
 * Public API barrel — re-exports types and client functions.
 * Owner: Member D (frontend lead).
 */

export type {
  AskRequest,
  AskResponse,
  CompareRequest,
  CompareResponse,
  HealthResponse,
  ModelInfo,
  ModelsListResponse,
  SectionDetailResponse,
  SourceCitation,
  ApiError,
} from "./types";

export { getHealth, getModels, askQuestion, getSection } from "./client";
