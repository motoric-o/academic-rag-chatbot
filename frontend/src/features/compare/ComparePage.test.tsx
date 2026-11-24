import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ComparePage } from "./ComparePage";

describe("ComparePage", () => {
  it("renders heading, model selector, and enforces 2-3 model selection", async () => {
    render(<ComparePage />);

    expect(screen.getByText(/Model Compare/i)).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText(/Select 2–3 models \(2\/3 selected\)/i)).toBeInTheDocument();
    });

    // Adding a 3rd model should succeed
    const thirdBtn = screen.getByRole("button", { name: /Qwen3 1\.7B/i });
    fireEvent.click(thirdBtn);
    expect(screen.getByText(/Select 2–3 models \(3\/3 selected\)/i)).toBeInTheDocument();

    // 4th unselected model should be disabled
    const fourthBtn = screen.getByRole("button", { name: /Qwen3 8B/i });
    expect(fourthBtn).toBeDisabled();
  });

  it("runs comparison and renders grounded answers, citations, and telemetry", async () => {
    render(<ComparePage />);
    await waitFor(() => {
      expect(screen.getByText(/Select 2–3 models/i)).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /^Compare$/i }));

    await waitFor(() => {
      expect(screen.getAllByText("Grounded").length).toBe(2);
    });
    expect(screen.getByText(/Comparison Telemetry/i)).toBeInTheDocument();
    expect(screen.getByText(/Replay \(canned fixture\)/i)).toBeInTheDocument();
  });

  it("distinguishes abstention (NOT_FOUND), format failure, and service errors", async () => {
    render(<ComparePage />);
    await waitFor(() => {
      expect(screen.getByText(/Select 2–3 models/i)).toBeInTheDocument();
    });

    const input = screen.getByLabelText("Comparison question");

    // 1. Abstention
    fireEvent.change(input, { target: { value: "What is the fee for leave?" } });
    fireEvent.click(screen.getByRole("button", { name: /^Compare$/i }));
    await waitFor(() => {
      expect(screen.getAllByText(/Abstained \(NOT_FOUND\)/i).length).toBe(2);
    });

    // 2. Format failure
    fireEvent.change(input, { target: { value: "Check format output" } });
    fireEvent.click(screen.getByRole("button", { name: /^Compare$/i }));
    await waitFor(() => {
      expect(screen.getByText("Format Failure")).toBeInTheDocument();
    });

    // 3. Service error
    fireEvent.change(input, { target: { value: "Trigger quota error" } });
    fireEvent.click(screen.getByRole("button", { name: /^Compare$/i }));
    await waitFor(() => {
      expect(screen.getAllByText("QUOTA_EXHAUSTED").length).toBeGreaterThanOrEqual(1);
    });
  });
});
