/**
 * ChatPage — primary Q&A interface.
 *
 * Layout: question input bar at the bottom, messages scroll upward.
 * Features: model selector, precision dropdown, mode badge, citation links,
 * NOT_FOUND friendly text, loading skeleton, error retry.
 *
 * Owner: Member D (spec §12.2).
 */

import { useState, useCallback, useRef, useEffect } from "react";
import type { AskResponse, ModelInfo } from "../../api/types";
import { askQuestion, getModels } from "../../api/client";
import { SourcesPanel } from "../sources/SourcesPanel";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  response?: AskResponse;
  loading?: boolean;
  error?: string;
}

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const PRECISION_OPTIONS = ["4-bit", "8-bit", "fp16"] as const;

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function ChatPage() {
  /* — state — */
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState("qwen3-4b");
  const [selectedPrecision, setSelectedPrecision] = useState<string>("4-bit");
  const [mode] = useState<"replay" | "live-hosted" | "live-local">("replay");
  const [submitting, setSubmitting] = useState(false);

  // Sources panel
  const [openSectionId, setOpenSectionId] = useState<string | null>(null);
  const [highlightText, setHighlightText] = useState<string | undefined>();

  const scrollRef = useRef<HTMLDivElement>(null);

  /* — load models on mount — */
  useEffect(() => {
    getModels()
      .then(setModels)
      .catch(() => {});
  }, []);

  /* — auto-scroll — */
  useEffect(() => {
    if (scrollRef.current && typeof scrollRef.current.scrollTo === "function") {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages]);

  /* — submit — */
  const handleSubmit = useCallback(
    async (e?: React.FormEvent) => {
      e?.preventDefault();
      const q = input.trim();
      if (!q || submitting) return;

      const userMsg: ChatMessage = {
        id: `u-${Date.now()}`,
        role: "user",
        content: q,
      };
      const assistantMsg: ChatMessage = {
        id: `a-${Date.now()}`,
        role: "assistant",
        content: "",
        loading: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setInput("");
      setSubmitting(true);

      try {
        const res = await askQuestion({
          question: q,
          model_id: selectedModel,
          precision: selectedPrecision,
          mode,
        });

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsg.id
              ? {
                  ...m,
                  content: res.answer_text,
                  response: res,
                  loading: false,
                }
              : m,
          ),
        );
      } catch (err) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsg.id
              ? {
                  ...m,
                  loading: false,
                  error:
                    err instanceof Error ? err.message : "Something went wrong",
                }
              : m,
          ),
        );
      } finally {
        setSubmitting(false);
      }
    },
    [input, selectedModel, selectedPrecision, mode, submitting],
  );

  /* — retry — */
  const handleRetry = useCallback(
    (msgId: string) => {
      const userMsg = messages.find(
        (m, i) =>
          m.role === "user" && messages[i + 1]?.id === msgId,
      );
      if (userMsg) {
        setInput(userMsg.content);
        setMessages((prev) =>
          prev.filter((m) => m.id !== msgId && m.id !== userMsg.id),
        );
      }
    },
    [messages],
  );

  /* — citation click — */
  const handleCitationClick = useCallback(
    (sectionId: string, text: string) => {
      setOpenSectionId(sectionId);
      setHighlightText(text);
    },
    [],
  );

  /* ---------------------------------------------------------------- */
  /*  Render                                                           */
  /* ---------------------------------------------------------------- */

  return (
    <div className="flex h-full gap-0">
      {/* — Main chat area — */}
      <div className="flex flex-1 flex-col">
        {/* -- Top bar: model selector + mode badge -- */}
        <div className="flex flex-wrap items-center gap-3 border-b border-slate-200 bg-white px-4 py-2">
          {/* Model selector */}
          <label className="flex items-center gap-1.5 text-sm text-slate-600">
            <span className="font-medium">Model:</span>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="rounded border border-slate-300 bg-white px-2 py-1 text-sm focus:border-blue-400 focus:outline-none"
            >
              {models.length > 0
                ? models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))
                : (
                    <option value={selectedModel}>{selectedModel}</option>
                  )}
            </select>
          </label>

          {/* Precision selector */}
          <label className="flex items-center gap-1.5 text-sm text-slate-600">
            <span className="font-medium">Precision:</span>
            <select
              value={selectedPrecision}
              onChange={(e) => setSelectedPrecision(e.target.value)}
              className="rounded border border-slate-300 bg-white px-2 py-1 text-sm focus:border-blue-400 focus:outline-none"
            >
              {PRECISION_OPTIONS.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>

          {/* Mode badge */}
          <span
            className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
              mode === "replay"
                ? "bg-amber-100 text-amber-700"
                : mode === "live-hosted"
                  ? "bg-green-100 text-green-700"
                  : "bg-purple-100 text-purple-700"
            }`}
          >
            {mode}
          </span>
        </div>

        {/* -- Messages area -- */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto px-4 py-4 space-y-4"
        >
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-slate-400">
              <svg className="w-12 h-12 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
              <p className="text-sm">Ask a question about academic regulations</p>
              <p className="text-xs mt-1">e.g. "How long may academic leave last?"</p>
            </div>
          )}

          {messages.map((msg) => (
            <MessageBubble
              key={msg.id}
              message={msg}
              onRetry={handleRetry}
              onCitationClick={handleCitationClick}
            />
          ))}
        </div>

        {/* -- Input bar -- */}
        <form
          onSubmit={handleSubmit}
          className="border-t border-slate-200 bg-white px-4 py-3"
        >
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your question here…"
              disabled={submitting}
              className="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm placeholder:text-slate-400 focus:border-blue-400 focus:outline-none disabled:bg-slate-50"
            />
            <button
              type="submit"
              disabled={submitting || !input.trim()}
              className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors"
            >
              {submitting ? "…" : "Ask"}
            </button>
          </div>
        </form>
      </div>

      {/* — Sources side panel — */}
      {openSectionId && (
        <SourcesPanel
          sectionId={openSectionId}
          highlightText={highlightText}
          onClose={() => setOpenSectionId(null)}
        />
      )}
    </div>
  );
}

/* ================================================================== */
/*  Sub-components                                                     */
/* ================================================================== */

function MessageBubble({
  message,
  onRetry,
  onCitationClick,
}: {
  message: ChatMessage;
  onRetry: (id: string) => void;
  onCitationClick: (sectionId: string, text: string) => void;
}) {
  const isUser = message.role === "user";

  /* -- Loading skeleton -- */
  if (message.loading) {
    return (
      <div className="flex justify-start">
        <div className="max-w-[75%] rounded-lg bg-slate-100 px-4 py-3 space-y-2 animate-pulse">
          <div className="h-3 w-48 rounded bg-slate-200" />
          <div className="h-3 w-64 rounded bg-slate-200" />
          <div className="h-3 w-36 rounded bg-slate-200" />
        </div>
      </div>
    );
  }

  /* -- Error state -- */
  if (message.error) {
    return (
      <div className="flex justify-start">
        <div className="max-w-[75%] rounded-lg border border-red-200 bg-red-50 px-4 py-3">
          <p className="text-sm text-red-700">⚠ {message.error}</p>
          <button
            onClick={() => onRetry(message.id)}
            className="mt-2 text-xs font-medium text-red-600 underline hover:text-red-800"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  /* -- User bubble -- */
  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-lg bg-blue-600 px-4 py-2.5 text-sm text-white">
          {message.content}
        </div>
      </div>
    );
  }

  /* -- Assistant bubble -- */
  const res = message.response;
  const isAbstained = res?.abstained;

  return (
    <div className="flex justify-start">
      <div className="max-w-[75%] space-y-2">
        {/* Answer text */}
        <div
          className={`rounded-lg px-4 py-3 text-sm ${
            isAbstained
              ? "border border-amber-200 bg-amber-50 text-amber-800"
              : "bg-slate-100 text-slate-800"
          }`}
        >
          {isAbstained ? (
            <div className="flex items-start gap-2">
              <span className="text-lg">📭</span>
              <div>
                <p className="font-medium">
                  Jawabannya belum ditemukan di buku panduan.
                </p>
                <p className="mt-1 text-xs text-amber-600">
                  The regulation document does not contain information to answer
                  this question.
                </p>
              </div>
            </div>
          ) : (
            <p className="whitespace-pre-wrap">{message.content}</p>
          )}
        </div>

        {/* Citations */}
        {res && res.cited_sources.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 px-1">
            <span className="text-xs text-slate-500">Sources:</span>
            {res.cited_sources.map((src) => (
              <button
                key={src.passage_idx}
                onClick={() => onCitationClick(src.section_id, src.text)}
                className="inline-flex items-center rounded bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-700 hover:bg-blue-200 transition-colors"
                title={`${src.chapter} › ${src.article}`}
              >
                [{src.passage_idx}] {src.article}
              </button>
            ))}
          </div>
        )}

        {/* Format / error warnings */}
        {res && !res.format_ok && (
          <p className="px-1 text-xs text-red-500">
            ⚠ Output format did not match the expected contract.
          </p>
        )}
        {res?.error && (
          <p className="px-1 text-xs text-red-500">
            ⚠ {res.error.code}: {res.error.message}
          </p>
        )}

        {/* Telemetry summary (small text) */}
        {res && (
          <p className="px-1 text-[11px] text-slate-400">
            {res.model} · {res.precision} · {res.latency_s.toFixed(2)}s ·{" "}
            {res.tokens_in} in / {res.tokens_out} out
            {res.peak_vram_mb != null && ` · ${res.peak_vram_mb} MB VRAM`}
          </p>
        )}
      </div>
    </div>
  );
}
