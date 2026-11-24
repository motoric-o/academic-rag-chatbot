// Owner: Member A (see docs/project_spec.md §0.2 & §12.2)
// Compare page: send the same question + identical cached context to 2-3 models side by side.

import { useEffect, useState } from "react";
import { getModels } from "../../api";
import type { AskResponse, ModelInfo } from "../../api";
import { TelemetryPanel } from "../telemetry";
import { compareModels } from "./compareClient";

type RunMode = "replay" | "live-hosted" | "live-local";

const SAMPLE_QUESTIONS = [
  "How long may academic leave last?",
  "What are the campus parking rules?",
  "Compare format compliance for academic leave rules",
];

export function ComparePage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([
    "qwen3-0.6b",
    "qwen3-4b",
  ]);
  const [question, setQuestion] = useState(SAMPLE_QUESTIONS[0]);
  const [precision, setPrecision] = useState("4-bit");
  const [mode, setMode] = useState<RunMode>("replay");
  const [results, setResults] = useState<AskResponse[]>([]);
  const [isFallback, setIsFallback] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    let active = true;
    getModels().then((list) => {
      if (!active) return;
      setModels(list);
      if (list.length >= 2) {
        setSelectedIds((prev) => {
          const valid = prev.filter((id) => list.some((m) => m.id === id));
          return valid.length >= 2 ? valid : [list[0].id, list[1].id];
        });
      }
    });
    return () => {
      active = false;
    };
  }, []);

  function toggleModel(id: string) {
    setSelectedIds((prev) => {
      if (prev.includes(id)) {
        if (prev.length <= 2) return prev; // Enforce minimum of 2 models (§12.1)
        return prev.filter((m) => m !== id);
      }
      if (prev.length >= 3) return prev; // Enforce maximum of 3 models (§12.1)
      return [...prev, id];
    });
  }

  async function runCompare(overrideQuestion?: string) {
    const q = (overrideQuestion ?? question).trim();
    if (!q || selectedIds.length < 2) return;
    setIsLoading(true);
    try {
      const response = await compareModels({
        question: q,
        model_ids: selectedIds,
        precision,
        mode,
      });
      setResults(response.data.results);
      setIsFallback(response.isFallback);
    } finally {
      setIsLoading(false);
    }
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    void runCompare();
  }

  const abstainedCount = results.filter((r) => r.abstained).length;
  const formatFailCount = results.filter((r) => !r.format_ok && !r.error).length;
  const errorCount = results.filter((r) => r.error !== null).length;
  const fastest =
    results.length > 0
      ? results.reduce((a, b) => (a.latency_s <= b.latency_s ? a : b))
      : null;

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Model Compare
            </h2>
            <p className="text-xs text-slate-500">
              Send the same question and identical cached context to 2–3 models side by side.
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <label className="flex items-center gap-1.5 text-slate-600">
              <span>Precision</span>
              <select
                aria-label="Precision"
                value={precision}
                onChange={(e) => setPrecision(e.target.value)}
                className="rounded border border-slate-300 bg-white px-2 py-1 text-xs text-slate-800"
              >
                <option value="4-bit">4-bit</option>
                <option value="8-bit">8-bit</option>
                <option value="fp16">fp16</option>
              </select>
            </label>
            <label className="flex items-center gap-1.5 text-slate-600">
              <span>Mode</span>
              <select
                aria-label="Execution mode"
                value={mode}
                onChange={(e) => setMode(e.target.value as RunMode)}
                className="rounded border border-slate-300 bg-white px-2 py-1 text-xs text-slate-800"
              >
                <option value="replay">replay</option>
                <option value="live-hosted">live-hosted</option>
                <option value="live-local">live-local</option>
              </select>
            </label>
          </div>
        </div>

        <div className="mt-3">
          <div className="mb-1.5 flex items-center justify-between text-xs text-slate-600">
            <span>Select 2–3 models ({selectedIds.length}/3 selected)</span>
          </div>
          <div className="flex flex-wrap gap-2" role="group" aria-label="Models to compare">
            {models.map((m) => {
              const checked = selectedIds.includes(m.id);
              const disabled =
                (!checked && selectedIds.length >= 3) ||
                (checked && selectedIds.length <= 2);
              const paramsLabel =
                m.params_total_b !== null
                  ? m.params_active_b !== null &&
                    m.params_active_b !== m.params_total_b
                    ? `${m.params_total_b}B (${m.params_active_b}B active)`
                    : `${m.params_total_b}B`
                  : "";

              return (
                <button
                  key={m.id}
                  type="button"
                  aria-pressed={checked}
                  disabled={disabled}
                  onClick={() => toggleModel(m.id)}
                  className={`rounded-md border px-2.5 py-1.5 text-left text-xs transition-colors ${
                    checked
                      ? "border-blue-500 bg-blue-50 text-blue-800"
                      : "border-slate-200 bg-slate-50 text-slate-700 hover:bg-slate-100"
                  } disabled:cursor-not-allowed disabled:opacity-50`}
                >
                  <span className="font-medium">{m.name}</span>
                  {paramsLabel && (
                    <span className="ml-1.5 text-[11px] text-slate-500">
                      {paramsLabel}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        <form onSubmit={handleSubmit} className="mt-4 flex flex-col gap-2 sm:flex-row">
          <input
            type="text"
            aria-label="Comparison question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Enter a regulation question to compare models..."
            className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-900 focus:border-blue-500 focus:outline-none"
          />
          <button
            type="submit"
            disabled={isLoading || !question.trim() || selectedIds.length < 2}
            className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {isLoading ? "Comparing..." : "Compare"}
          </button>
        </form>

        <div className="mt-2 flex flex-wrap gap-1.5">
          {SAMPLE_QUESTIONS.map((sample) => (
            <button
              key={sample}
              type="button"
              onClick={() => {
                setQuestion(sample);
                void runCompare(sample);
              }}
              className="rounded border border-slate-200 bg-slate-50 px-2 py-0.5 text-[11px] text-slate-600 hover:bg-slate-100"
            >
              {sample}
            </button>
          ))}
        </div>
      </section>

      {isFallback && results.length > 0 && (
        <div
          role="status"
          className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900"
        >
          Showing canned replay fixture data (backend API offline or mock mode enabled).
        </div>
      )}

      {results.length > 0 && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-2 text-xs text-slate-700">
            <span>
              <strong>Models:</strong> {results.length}
            </span>
            <span>
              <strong>Abstained:</strong> {abstainedCount}/{results.length}
            </span>
            {formatFailCount > 0 && (
              <span className="font-medium text-amber-800">
                Format failures: {formatFailCount}
              </span>
            )}
            {errorCount > 0 && (
              <span className="font-medium text-red-700">
                Service errors: {errorCount}
              </span>
            )}
            {fastest && (
              <span>
                <strong>Fastest:</strong> {fastest.model} ({fastest.latency_s.toFixed(2)}s)
              </span>
            )}
          </div>

          <div
            className={`grid gap-4 ${
              results.length === 3 ? "md:grid-cols-3" : "md:grid-cols-2"
            }`}
          >
            {results.map((res) => (
              <article
                key={`${res.model}-${res.precision}`}
                className="flex flex-col justify-between rounded-lg border border-slate-200 bg-white p-4"
              >
                <div>
                  <div className="flex items-start justify-between gap-2 border-b border-slate-100 pb-2">
                    <div>
                      <h3 className="text-sm font-semibold text-slate-900">
                        {res.model}
                      </h3>
                      <p className="text-[11px] text-slate-500">
                        {res.precision} · {res.mode}
                      </p>
                    </div>
                    {res.error ? (
                      <span className="rounded bg-red-50 px-2 py-0.5 text-[11px] font-medium text-red-700">
                        {res.error.code}
                      </span>
                    ) : !res.format_ok ? (
                      <span className="rounded bg-amber-50 px-2 py-0.5 text-[11px] font-medium text-amber-800">
                        Format Failure
                      </span>
                    ) : res.abstained ? (
                      <span className="rounded bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-700">
                        Abstained (NOT_FOUND)
                      </span>
                    ) : (
                      <span className="rounded bg-emerald-50 px-2 py-0.5 text-[11px] font-medium text-emerald-700">
                        Grounded
                      </span>
                    )}
                  </div>

                  <div className="mt-3 text-sm text-slate-800">
                    {res.error ? (
                      <div className="space-y-2 rounded border border-red-200 bg-red-50 p-2.5 text-xs text-red-800">
                        <p className="font-medium">{res.error.message}</p>
                        <button
                          type="button"
                          onClick={() => void runCompare()}
                          className="rounded border border-red-300 bg-white px-2 py-1 text-xs font-medium text-red-700 hover:bg-red-50"
                        >
                          Retry
                        </button>
                      </div>
                    ) : res.abstained ? (
                      <p className="rounded border border-slate-200 bg-slate-50 px-3 py-2 font-mono text-xs text-slate-700">
                        NOT_FOUND — Context does not contain the answer.
                      </p>
                    ) : (
                      <p className="whitespace-pre-wrap leading-relaxed">
                        {res.answer_text}
                      </p>
                    )}

                    {!res.error && !res.format_ok && res.raw_output && (
                      <details className="mt-2 rounded border border-amber-200 bg-amber-50 p-2 text-xs text-amber-900">
                        <summary className="cursor-pointer font-medium">
                          Inspect raw output
                        </summary>
                        <pre className="mt-1 whitespace-pre-wrap font-mono text-[11px]">
                          {res.raw_output}
                        </pre>
                      </details>
                    )}
                  </div>
                </div>

                {res.cited_sources.length > 0 && (
                  <div className="mt-4 border-t border-slate-100 pt-2">
                    <p className="text-[11px] font-medium text-slate-500">
                      Citations
                    </p>
                    <div className="mt-1 flex flex-wrap gap-1.5">
                      {res.cited_sources.map((src) => (
                        <span
                          key={`${src.passage_idx}-${src.section_id}`}
                          title={src.text}
                          className="rounded bg-blue-50 px-2 py-0.5 text-[11px] font-medium text-blue-700"
                        >
                          [{src.passage_idx}] §{src.section_id}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </article>
            ))}
          </div>

          <TelemetryPanel
            results={results}
            isReplayFallback={isFallback}
            isLoading={isLoading}
            title="Comparison Telemetry"
          />
        </div>
      )}
    </div>
  );
}
