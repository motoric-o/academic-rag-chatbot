/**
 * Tests for SourcesPanel component (Member D).
 */
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SourcesPanel } from "./SourcesPanel";

describe("SourcesPanel", () => {
  it("renders loading state initially and then shows section details", async () => {
    const handleClose = vi.fn();
    render(
      <SourcesPanel
        sectionId="1.1"
        highlightText="consecutive terms"
        onClose={handleClose}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Academic Leave")).toBeInTheDocument();
      expect(screen.getByText("Chapter 1")).toBeInTheDocument();
      expect(screen.getByText(/Article 1.1/)).toBeInTheDocument();
    });

    // Check highlighted text is wrapped in <mark>
    const markEl = screen.getByText("consecutive terms");
    expect(markEl.tagName.toLowerCase()).toBe("mark");
  });

  it("calls onClose when close button is clicked", async () => {
    const handleClose = vi.fn();
    render(
      <SourcesPanel sectionId="1.1" onClose={handleClose} />,
    );

    const closeBtn = screen.getByLabelText("Close sources panel");
    fireEvent.click(closeBtn);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it("displays error when section is not found", async () => {
    const handleClose = vi.fn();
    render(
      <SourcesPanel sectionId="999.invalid" onClose={handleClose} />,
    );

    await waitFor(() => {
      expect(screen.getByText(/Section "999.invalid" not found/i)).toBeInTheDocument();
    });
  });
});
