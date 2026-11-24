/**
 * SourcesPanel — slide-in side panel showing the original regulation text.
 *
 * When a user clicks a citation [n] in the chat, this panel opens with the
 * relevant section. The cited passage fragment is highlighted.
 *
 * Owner: Member D (spec §12.2).
 */

import { useEffect, useState } from "react";
import type { SectionDetailResponse } from "../../api/types";
import { getSection } from "../../api/client";

interface SourcesPanelProps {
  sectionId: string;
  highlightText?: string;
  onClose: () => void;
}

export function SourcesPanel({
  sectionId,
  highlightText,
  onClose,
}: SourcesPanelProps) {
  const [section, setSection] = useState<SectionDetailResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getSection(sectionId)
      .then((data) => {
        if (data) {
          setSection(data);
        } else {
          setError(`Section "${sectionId}" not found.`);
        }
      })
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Failed to load section"),
      )
      .finally(() => setLoading(false));
  }, [sectionId]);

  /* ---------------------------------------------------------------- */
  /*  Highlight helper: wraps matching substring in <mark>             */
  /* ---------------------------------------------------------------- */

  function renderHighlightedText(fullText: string, highlight?: string) {
    if (!highlight) return <p className="whitespace-pre-wrap">{fullText}</p>;

    const idx = fullText.toLowerCase().indexOf(highlight.toLowerCase());
    if (idx === -1) return <p className="whitespace-pre-wrap">{fullText}</p>;

    const before = fullText.slice(0, idx);
    const match = fullText.slice(idx, idx + highlight.length);
    const after = fullText.slice(idx + highlight.length);

    return (
      <p className="whitespace-pre-wrap">
        {before}
        <mark className="rounded bg-yellow-200 px-0.5">{match}</mark>
        {after}
      </p>
    );
  }

  /* ---------------------------------------------------------------- */
  /*  Render                                                           */
  /* ---------------------------------------------------------------- */

  return (
    <aside className="w-80 flex-shrink-0 border-l border-slate-200 bg-white flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
        <h3 className="text-sm font-semibold text-slate-800">
          📋 Source Document
        </h3>
        <button
          onClick={onClose}
          className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600 transition-colors"
          aria-label="Close sources panel"
        >
          <svg className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
            <path
              fillRule="evenodd"
              d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
              clipRule="evenodd"
            />
          </svg>
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {loading && (
          <div className="space-y-3 animate-pulse">
            <div className="h-4 w-32 rounded bg-slate-200" />
            <div className="h-3 w-full rounded bg-slate-200" />
            <div className="h-3 w-5/6 rounded bg-slate-200" />
            <div className="h-3 w-4/6 rounded bg-slate-200" />
          </div>
        )}

        {error && (
          <div className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            {error}
          </div>
        )}

        {section && !loading && (
          <div className="space-y-3">
            {/* Breadcrumb labels */}
            <div className="flex flex-wrap gap-1.5">
              <span className="rounded bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600">
                {section.chapter}
              </span>
              <span className="rounded bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600">
                {section.article}
              </span>
            </div>

            {/* Section title */}
            {section.title && (
              <h4 className="text-sm font-semibold text-slate-900">
                {section.title}
              </h4>
            )}

            {/* Section ID */}
            <p className="text-[11px] text-slate-400">
              Section ID: <code className="font-mono">{section.section_id}</code>
            </p>

            {/* Full text with highlight */}
            <div className="rounded border border-slate-150 bg-slate-50 px-3 py-2.5 text-sm leading-relaxed text-slate-700">
              {renderHighlightedText(section.text, highlightText)}
            </div>

            {/* Footer note */}
            <p className="text-[10px] text-slate-400 italic">
              Source retrieved from the translated regulation corpus. Always
              verify against the official document.
            </p>
          </div>
        )}
      </div>
    </aside>
  );
}
