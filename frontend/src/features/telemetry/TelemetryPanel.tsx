// Owner: Member A (see docs/project_spec.md §0.2 & §12.2)
// Telemetry panel: latency, tokens in/out, peak VRAM (local), and replay vs. live badge.

import type { AskResponse } from "../../api";

export interface TelemetryPanelProps {
  results?: AskResponse[];
  isReplayFallback?: boolean;
  isLoading?: boolean;
  title?: string;
}

const SAMPLE_REPLAY_TELEMETRY: AskResponse[] = [
  {
    question: "How long may academic leave last?",
    model: "qwen3-0.6b",
    precision: "4-bit",
    mode: "replay",
    answer_text: "Up to three consecutive terms.",
    cited_sources: [],
    abstained: false,
    format_ok: true,
    tokens_in: 480,
    tokens_out: 24,
    latency_s: 0.62,
    peak_vram_mb: 1180,
    raw_output: "Up to three consecutive terms. Sources: [1]",
    error: null,
  },
  {
    question: "How long may academic leave last?",
    model: "openai/gpt-oss-20b",
    precision: "MXFP4",
    mode: "replay",
    answer_text: "Up to three consecutive terms.",
    cited_sources: [],
    abstained: false,
    format_ok: true,
    tokens_in: 480,
    tokens_out: 31,
    latency_s: 0.48,
    peak_vram_mb: null,
    raw_output: "Up to three consecutive terms. Sources: [1]",
    error: null,
  },
];

function getModeBadge(mode: string, isReplayFallback: boolean): {
  label: string;
  className: string;
} {
  if (isReplayFallback) {
    return {
      label: "Replay (canned fixture)",
      className: "bg-amber-50 text-amber-800 border-amber-200",
    };
  }
  if (mode === "live-local") {
    return {
      label: "Live (local GPU)",
      className: "bg-emerald-50 text-emerald-800 border-emerald-200",
    };
  }
  if (mode === "live-hosted") {
    return {
      label: "Live (hosted API)",
      className: "bg-blue-50 text-blue-800 border-blue-200",
    };
  }
  return {
    label: "Replay data",
    className: "bg-slate-100 text-slate-700 border-slate-200",
  };
}

export function TelemetryPanel({
  results,
  isReplayFallback = false,
  isLoading = false,
  title = "Telemetry Panel",
}: TelemetryPanelProps) {
  const isStandaloneSample = results === undefined;
  const rows = isStandaloneSample ? SAMPLE_REPLAY_TELEMETRY : results;
  const effectiveFallback = isStandaloneSample ? true : isReplayFallback;

  if (isLoading) {
    return (
      <section
        aria-label={title}
        className="rounded-lg border border-slate-200 bg-white p-4"
      >
        <h2 className="text-base font-semibold text-slate-900">{title}</h2>
        <div className="mt-3 space-y-2" role="status" aria-label="Loading telemetry">
          <div className="h-8 animate-pulse rounded bg-slate-100" />
          <div className="h-8 animate-pulse rounded bg-slate-100" />
        </div>
      </section>
    );
  }

  if (rows.length === 0) {
    return (
      <section
        aria-label={title}
        className="rounded-lg border border-slate-200 bg-white p-4"
      >
        <h2 className="text-base font-semibold text-slate-900">{title}</h2>
        <p className="mt-2 text-sm text-slate-500">
          Run a comparison query to inspect latency, token usage, and local peak VRAM.
        </p>
      </section>
    );
  }

  const maxLatency = Math.max(...rows.map((r) => r.latency_s), 0.01);
  const maxVram = Math.max(...rows.map((r) => r.peak_vram_mb ?? 0), 1);
  const badge = getModeBadge(rows[0].mode, effectiveFallback);

  return (
    <section
      aria-label={title}
      className="rounded-lg border border-slate-200 bg-white p-4"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-base font-semibold text-slate-900">{title}</h2>
          <p className="text-xs text-slate-500">
            Latency, input/output tokens, throughput, and local GPU peak VRAM.
          </p>
        </div>
        <span
          className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${badge.className}`}
        >
          {badge.label}
        </span>
      </div>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full border-collapse text-left text-xs">
          <thead>
            <tr className="border-b border-slate-200 text-slate-500">
              <th className="py-2 pr-3 font-medium">Model</th>
              <th className="py-2 pr-3 font-medium">Precision</th>
              <th className="py-2 pr-3 font-medium">Latency (s)</th>
              <th className="py-2 pr-3 font-medium">Tokens (in / out)</th>
              <th className="py-2 pr-3 font-medium">Throughput</th>
              <th className="py-2 pr-3 font-medium">Peak VRAM</th>
              <th className="py-2 font-medium">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {rows.map((row) => {
              const tokPerSec =
                row.latency_s > 0 && row.tokens_out > 0
                  ? `${(row.tokens_out / row.latency_s).toFixed(1)} tok/s`
                  : "—";
              const latencyPct = Math.min(
                100,
                Math.round((row.latency_s / maxLatency) * 100),
              );
              const vramPct =
                row.peak_vram_mb !== null
                  ? Math.min(100, Math.round((row.peak_vram_mb / maxVram) * 100))
                  : 0;

              return (
                <tr key={`${row.model}-${row.precision}`} className="text-slate-700">
                  <td className="py-2.5 pr-3 font-medium text-slate-900">
                    {row.model}
                  </td>
                  <td className="py-2.5 pr-3">{row.precision}</td>
                  <td className="py-2.5 pr-3">
                    <div className="flex items-center gap-2">
                      <span className="w-12 tabular-nums">
                        {row.latency_s.toFixed(2)}s
                      </span>
                      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-100">
                        <div
                          className="h-full rounded-full bg-blue-500"
                          style={{ width: `${latencyPct}%` }}
                        />
                      </div>
                    </div>
                  </td>
                  <td className="py-2.5 pr-3 tabular-nums">
                    {row.tokens_in} / {row.tokens_out}
                  </td>
                  <td className="py-2.5 pr-3 tabular-nums">{tokPerSec}</td>
                  <td className="py-2.5 pr-3">
                    {row.peak_vram_mb !== null ? (
                      <div className="flex items-center gap-2">
                        <span className="w-16 tabular-nums">
                          {Math.round(row.peak_vram_mb)} MB
                        </span>
                        <div className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-100">
                          <div
                            className="h-full rounded-full bg-violet-500"
                            style={{ width: `${vramPct}%` }}
                          />
                        </div>
                      </div>
                    ) : (
                      <span className="text-slate-400">N/A (hosted)</span>
                    )}
                  </td>
                  <td className="py-2.5">
                    {row.error ? (
                      <span className="rounded bg-red-50 px-1.5 py-0.5 font-medium text-red-700">
                        {row.error.code}
                      </span>
                    ) : !row.format_ok ? (
                      <span className="rounded bg-amber-50 px-1.5 py-0.5 font-medium text-amber-800">
                        Format fail
                      </span>
                    ) : row.abstained ? (
                      <span className="rounded bg-slate-100 px-1.5 py-0.5 font-medium text-slate-700">
                        Abstained
                      </span>
                    ) : (
                      <span className="rounded bg-emerald-50 px-1.5 py-0.5 font-medium text-emerald-700">
                        OK
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
