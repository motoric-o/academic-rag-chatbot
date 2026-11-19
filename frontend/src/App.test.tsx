import { render, screen } from "@testing-library/react";
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
});
