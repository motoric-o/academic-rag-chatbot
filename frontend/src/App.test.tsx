/**
 * Tests for the App shell (Member D).
 * Verifies: title, disclaimer, navigation tabs, default page.
 */
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { App } from "./App";

describe("App shell", () => {
  it("renders title and regulation disclaimer", () => {
    render(<App />);
    expect(
      screen.getByText(/Academic Regulation QA — Model Scaling/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Informational only/i),
    ).toBeInTheDocument();
  });

  it("shows Chat and Compare navigation tabs", () => {
    render(<App />);
    expect(screen.getByText(/Chat/)).toBeInTheDocument();
    expect(screen.getByText(/Compare/)).toBeInTheDocument();
  });

  it("defaults to chat page with Ask button", () => {
    render(<App />);
    expect(screen.getByText("Ask")).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText(/Type your question/i),
    ).toBeInTheDocument();
  });

  it("switches to compare page when Compare tab is clicked", () => {
    render(<App />);
    fireEvent.click(screen.getByText(/Compare/));
    expect(screen.getByText(/Model Compare/i)).toBeInTheDocument();
  });

  it("switches back to chat page when Chat tab is clicked", () => {
    render(<App />);
    fireEvent.click(screen.getByText(/Compare/));
    fireEvent.click(screen.getByText(/Chat/));
    expect(screen.getByText("Ask")).toBeInTheDocument();
  });
});
