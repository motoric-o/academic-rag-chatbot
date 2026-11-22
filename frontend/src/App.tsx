/**
 * App shell — layout, routing, and global disclaimer.
 *
 * Owner: Member D (spec §12.2 — app shell, routing, state management,
 * styling, Chat page, Sources panel, disclaimer, error/empty states).
 *
 * Pages:
 *   /        → Chat (Member D)
 *   /compare → Compare placeholder (Member A)
 */

import { useState } from "react";
import { ChatPage } from "./features/chat";
import { ComparePlaceholder } from "./features/compare";
import { TelemetryPlaceholder } from "./features/telemetry";

type Page = "chat" | "compare";

export function App() {
  const [page, setPage] = useState<Page>("chat");

  return (
    <div className="flex h-screen flex-col">
      {/* ============================================================ */}
      {/*  Header                                                       */}
      {/* ============================================================ */}
      <header className="border-b border-slate-200 bg-white px-6 py-3">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-slate-900">
              📖 Academic Regulation QA — Model Scaling
            </h1>
            <p className="text-xs text-slate-500">
              Fixed RAG pipeline comparison across local quantized Qwen3 and
              hosted open-weight models.
            </p>
          </div>

          {/* Nav tabs */}
          <nav className="flex gap-1">
            <NavButton
              label="💬 Chat"
              active={page === "chat"}
              onClick={() => setPage("chat")}
            />
            <NavButton
              label="🔀 Compare"
              active={page === "compare"}
              onClick={() => setPage("compare")}
            />
          </nav>
        </div>
      </header>

      {/* ============================================================ */}
      {/*  Main content                                                 */}
      {/* ============================================================ */}
      <main className="flex-1 overflow-hidden">
        {page === "chat" && <ChatPage />}
        {page === "compare" && (
          <div className="mx-auto w-full max-w-5xl space-y-6 p-6">
            <ComparePlaceholder />
            <TelemetryPlaceholder />
          </div>
        )}
      </main>

      {/* ============================================================ */}
      {/*  Footer / Disclaimer (spec §12.2)                            */}
      {/* ============================================================ */}
      <footer className="border-t border-slate-200 bg-white px-6 py-2 text-center text-[11px] text-slate-400">
        ⚠ Disclaimer: Informational only — always consult the official campus
        regulation document.
      </footer>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function NavButton({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
        active
          ? "bg-blue-50 text-blue-700"
          : "text-slate-500 hover:bg-slate-100 hover:text-slate-700"
      }`}
    >
      {label}
    </button>
  );
}
