import { ComparePlaceholder } from "./features/compare";
import { TelemetryPlaceholder } from "./features/telemetry";

export function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-slate-200 bg-white px-6 py-4">
        <h1 className="text-xl font-semibold text-slate-900">
          Academic Regulation QA — Model Scaling
        </h1>
        <p className="text-sm text-slate-600">
          Fixed RAG pipeline comparison across local quantized Qwen3 and hosted open-weight models.
        </p>
      </header>

      <main className="mx-auto w-full max-w-5xl flex-1 space-y-6 p-6">
        <section className="rounded-lg border border-slate-200 bg-white p-4">
          <h2 className="text-base font-medium">Chat &amp; Sources (Member D)</h2>
          <p className="mt-1 text-sm text-slate-600">
            Chat page and expandable Sources panel will live in{" "}
            <code>src/features/chat/</code> and <code>src/features/sources/</code>.
          </p>
        </section>

        <ComparePlaceholder />
        <TelemetryPlaceholder />
      </main>

      <footer className="border-t border-slate-200 bg-white px-6 py-3 text-xs text-slate-500">
        Disclaimer: Informational only — always consult the official campus regulation document.
      </footer>
    </div>
  );
}
