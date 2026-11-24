# Handoff: Member A -> Member D (Compare & Telemetry UI Integration)

- **From:** Member A (Compare & Telemetry UI Owner)
- **To:** Member D (Frontend Lead)
- **Date:** 2026-11-24
- **Scope Reference:** `docs/project_spec.md` §0.2, §12.2

---

## 1. Exported Components & Backward-Compatible Aliases

Member A implemented the interactive Compare and Telemetry components strictly inside `frontend/src/features/compare/` and `frontend/src/features/telemetry/`, reusing your `frontend/src/api/` types and `getModels()` helper:

- `frontend/src/features/compare/index.tsx`:
  - Exports `ComparePage` **and** `ComparePlaceholder` (alias of `ComparePage`, preserving `"Model Compare"` in the heading so `App.test.tsx` passes untouched).
  - Exports `compareModels` from `compareClient.ts`.
- `frontend/src/features/telemetry/index.tsx`:
  - Exports `TelemetryPanel`, `TelemetryPanelProps`, **and** `TelemetryPlaceholder` (alias of `TelemetryPanel`).
  - `TelemetryPanel` accepts optional `results?: AskResponse[]`, `isReplayFallback?: boolean`, `isLoading?: boolean`, and `title?: string`.

## 2. Small Optional Follow-Ups in Member D's Scope

1. **Scrolling on Compare tab (`frontend/src/App.tsx`):** `<main className="flex-1 overflow-hidden">` currently clips tall content when switching to the Compare tab. Adding `h-full overflow-y-auto` to the Compare tab wrapper `<div>` will keep side-by-side cards + Telemetry scrolling smoothly on smaller viewports.
2. **Optional `compareModels` promotion (`frontend/src/api/client.ts`):** `frontend/src/features/compare/compareClient.ts` implements `POST /api/compare` with automatic canned-replay fallback matching your `client.ts` pattern. Feel free to promote it into `frontend/src/api/client.ts` (and re-export from `frontend/src/api/index.ts`) whenever convenient.
