/**
 * Tests for the ChatPage component (Member D).
 * Tests against mock data (no backend needed).
 */
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ChatPage } from "./ChatPage";

describe("ChatPage", () => {
  it("renders the question input and Ask button", () => {
    render(<ChatPage />);
    expect(
      screen.getByPlaceholderText(/Type your question/i),
    ).toBeInTheDocument();
    expect(screen.getByText("Ask")).toBeInTheDocument();
  });

  it("shows model and precision selectors", () => {
    render(<ChatPage />);
    expect(screen.getByText("Model:")).toBeInTheDocument();
    expect(screen.getByText("Precision:")).toBeInTheDocument();
  });

  it("shows replay mode badge by default", () => {
    render(<ChatPage />);
    expect(screen.getByText("replay")).toBeInTheDocument();
  });

  it("shows empty state prompt when no messages", () => {
    render(<ChatPage />);
    expect(
      screen.getByText(/Ask a question about academic regulations/i),
    ).toBeInTheDocument();
  });

  it("disables Ask button when input is empty", () => {
    render(<ChatPage />);
    const askBtn = screen.getByText("Ask");
    expect(askBtn).toBeDisabled();
  });

  it("enables Ask button when text is typed and submits question", async () => {
    render(<ChatPage />);
    const input = screen.getByPlaceholderText(/Type your question/i);
    const askBtn = screen.getByText("Ask");

    fireEvent.change(input, { target: { value: "How long is academic leave?" } });
    expect(askBtn).not.toBeDisabled();

    fireEvent.click(askBtn);

    // Should show user message
    expect(screen.getByText("How long is academic leave?")).toBeInTheDocument();

    // Input should be cleared
    expect((input as HTMLInputElement).value).toBe("");

    // Wait for mock response to resolve
    await waitFor(
      () => {
        expect(
          screen.getByText(/Students may take up to three consecutive terms/i),
        ).toBeInTheDocument();
      },
      { timeout: 3000 },
    );
  });
});
