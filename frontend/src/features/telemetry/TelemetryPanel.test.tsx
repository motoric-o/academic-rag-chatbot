import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { AskResponse } from "../../api";
import { TelemetryPanel } from "./TelemetryPanel";

const SAMPLE_RESULTS: AskResponse[] = [
  {
    question: "Minimum GPA for Practical Work?",
    model: "qwen3-0.6b",
    precision: "4-bit",
    mode: "replay",
    answer_text: "Minimum cumulative GPA is 2.0.",
    cited_sources: [],
    abstained: false,
    format_ok: true,
    tokens_in: 400,
    tokens_out: 20,
    latency_s: 0.5,
    peak_vram_mb: 1200,
    raw_output: "Minimum cumulative GPA is 2.0. Sources: [1]",
    error: null,
  },
  {
    question: "Minimum GPA for Practical Work?",
    model: "openai/gpt-oss-20b",
    precision: "MXFP4",
    mode: "replay",
    answer_text: "Minimum cumulative GPA is 2.0.",
    cited_sources: [],
    abstained: false,
    format_ok: true,
    tokens_in: 400,
    tokens_out: 25,
    latency_s: 0.4,
    peak_vram_mb: null,
    raw_output: "Minimum cumulative GPA is 2.0. Sources: [1]",
    error: null,
  },
];

describe("TelemetryPanel", () => {
  it("renders latency, tokens, local VRAM, hosted N/A, and replay badge", () => {
    render(<TelemetryPanel results={SAMPLE_RESULTS} />);

    expect(screen.getByText("Replay data")).toBeInTheDocument();
    expect(screen.getByText("qwen3-0.6b")).toBeInTheDocument();
    expect(screen.getByText("1200 MB")).toBeInTheDocument();
    expect(screen.getByText("N/A (hosted)")).toBeInTheDocument();
    expect(screen.getByText("400 / 20")).toBeInTheDocument();
    expect(screen.getByText("40.0 tok/s")).toBeInTheDocument();
  });

  it("shows canned fixture badge when fallback is active", () => {
    render(<TelemetryPanel results={SAMPLE_RESULTS} isReplayFallback />);
    expect(screen.getByText("Replay (canned fixture)")).toBeInTheDocument();
  });
});
