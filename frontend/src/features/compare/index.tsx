// Owner: Member A (see docs/project_spec.md §0.2 & §12.2)
// Compare page: send the same question + identical context to 2-3 models side by side.

export function ComparePlaceholder() {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="text-base font-medium">Model Compare (Member A)</h2>
      <p className="mt-1 text-sm text-slate-600">
        Side-by-side comparison of 2–3 models on the same question and cached context.
      </p>
    </section>
  );
}
