/**
 * API client — calls the FastAPI backend via the Vite proxy (/api).
 * Automatically falls back to canned mock data when running in test,
 * when VITE_USE_MOCK=true, or when the backend is unreachable.
 *
 * Owner: Member D (spec §12.2 — frontend lead).
 */

import type {
  AskRequest,
  AskResponse,
  HealthResponse,
  ModelInfo,
  ModelsListResponse,
  SectionDetailResponse,
} from "./types";
import {
  MOCK_ABSTAIN_RESPONSE,
  MOCK_ASK_RESPONSE,
  MOCK_MODELS,
  MOCK_SECTIONS,
} from "./mock-data";

function isMockEnabled(): boolean {
  try {
    const globalProcess = (globalThis as unknown as { process?: { env?: { NODE_ENV?: string } } }).process;
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

let forceMock = isMockEnabled();

export function setUseMock(val: boolean) {
  forceMock = val;
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  if (forceMock) return { status: "ok", mode: "replay" };
  try {
    return await fetchJson<HealthResponse>("/api/health");
  } catch {
    return { status: "ok", mode: "replay" };
  }
}

export async function getModels(): Promise<ModelInfo[]> {
  if (forceMock) return MOCK_MODELS;
  try {
    const data = await fetchJson<ModelsListResponse>("/api/models");
    return data.models;
  } catch {
    return MOCK_MODELS;
  }
}

export async function askQuestion(req: AskRequest): Promise<AskResponse> {
  if (forceMock) {
    await new Promise((r) => setTimeout(r, 50));
    if (
      req.question.toLowerCase().includes("fee") ||
      req.question.toLowerCase().includes("not_found")
    ) {
      return {
        ...MOCK_ABSTAIN_RESPONSE,
        question: req.question,
        model: req.model_id ?? "mock-model",
        precision: req.precision ?? "4-bit",
      };
    }
    return {
      ...MOCK_ASK_RESPONSE,
      question: req.question,
      model: req.model_id ?? "mock-model",
      precision: req.precision ?? "4-bit",
    };
  }

  try {
    return await fetchJson<AskResponse>("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
  } catch (err) {
    console.warn("Backend unavailable, falling back to mock response", err);
    return {
      ...MOCK_ASK_RESPONSE,
      question: req.question,
      model: req.model_id ?? "mock-model",
      precision: req.precision ?? "4-bit",
    };
  }
}

export async function getSection(
  sectionId: string,
): Promise<SectionDetailResponse | null> {
  if (forceMock) {
    return MOCK_SECTIONS[sectionId] ?? null;
  }
  try {
    return await fetchJson<SectionDetailResponse>(
      `/api/sections/${encodeURIComponent(sectionId)}`,
    );
  } catch {
    return MOCK_SECTIONS[sectionId] ?? null;
  }
}
