// Owner: Member A (see docs/project_spec.md §0.2 & §12.2)
// Telemetry panel: latency, tokens in/out, peak VRAM (local), and replay-data label.

export function TelemetryPlaceholder() {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="text-base font-medium">Telemetry Panel (Member A)</h2>
      <p className="mt-1 text-sm text-slate-600">
        Displays latency, input/output tokens, peak VRAM (local), and replay vs. live mode indicator.
      </p>
    </section>
  );
}
