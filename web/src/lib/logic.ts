/** Pure logic shared by components (unit-tested in logic.test.ts). */
import { ApiError, type ModelOut, type StatusOut } from "../api/client";

// ---------- citations ----------

export type Segment = { kind: "text"; text: string } | { kind: "cite"; n: number };

/** Split an answer into plain text and [n] citation markers (only numbers that have a citation). */
export function splitCitations(text: string, valid: Set<number>): Segment[] {
  const out: Segment[] = [];
  let last = 0;
  for (const m of text.matchAll(/\[(\d+)\]/g)) {
    const n = Number(m[1]);
    if (!valid.has(n)) continue;
    if (m.index! > last) out.push({ kind: "text", text: text.slice(last, m.index) });
    out.push({ kind: "cite", n });
    last = m.index! + m[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}

// ---------- model tiers ----------

export type Tier = "economy" | "standard" | "premium" | "local";
export const TIERS: { id: Tier; label: string; hint: string }[] = [
  { id: "economy", label: "Cheap", hint: "Fast and low cost; great for factual lookups" },
  { id: "standard", label: "Balanced", hint: "Stronger reasoning at moderate cost" },
  { id: "premium", label: "Best", hint: "Most capable models, highest cost" },
  { id: "local", label: "Local", hint: "Runs on your machine: free and private" },
];

/** Model to use for a tier: the server default if it fits, else the cheapest available model. */
export function modelForTier(models: ModelOut[], tier: Tier, serverDefault?: string): ModelOut | undefined {
  const usable = models.filter((m) => m.available && m.tier === tier);
  const preferred = usable.find((m) => m.name === serverDefault);
  if (preferred) return preferred;
  return [...usable].sort((a, b) => price(a) - price(b) || a.name.localeCompare(b.name))[0];
}

/** Blended price per 1M tokens, weighted like a RAG answer (long prompt, short reply). */
export function price(m: ModelOut): number {
  return m.input_per_m * 0.95 + m.output_per_m * 0.05;
}

/** Rough cost of one answer (~1,500 input + 100 output tokens), for the picker hint. */
export function costPerAnswer(m: ModelOut): string {
  const usd = (1500 * m.input_per_m + 100 * m.output_per_m) / 1_000_000;
  if (usd === 0) return "free";
  return usd < 0.001 ? `~$${usd.toFixed(5)}` : `~$${usd.toFixed(3)}`;
}

// ---------- errors ----------

/** Human message for an API error code (never show raw errors to users). */
export function friendlyError(err: unknown): string {
  if (!(err instanceof ApiError)) return "Something went wrong. Please try again.";
  const retry = err.details?.retry_after_seconds;
  switch (err.code) {
    case "network_error":
      return "Can't reach the server. Check that the API is running.";
    case "missing_api_key":
    case "invalid_api_key":
      return "This server needs a valid API key. Add it in Settings.";
    case "rate_limited":
      return `You're sending requests too quickly. Try again in ${retry ?? "a few"} seconds.`;
    case "provider_error":
      return "The AI provider isn't responding right now. Try another model.";
    case "embedding_error":
      // Not an LLM problem: the server explains which embedding provider failed and how to fix it.
      return err.message;
    case "refused":
      return "The model declined to answer this question.";
    case "store_unavailable":
      return "The document database is unavailable. Try again shortly.";
    case "file_too_large":
      return "That file is too large to upload.";
    case "unsupported_file":
      return "Only PDF files can be uploaded.";
    case "chunker_not_allowed":
      return "That chunking strategy isn't allowed on this server.";
    case "index_not_found":
      return "That index doesn't exist anymore. Pick another one.";
    case "file_not_available":
      return "The original file isn't stored for this document (it was added outside the web app).";
    default:
      return err.message || "Something went wrong. Please try again.";
  }
}

// ---------- misc ----------

export function formatMs(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

/** Answer text + citations as Markdown (for copy/export). */
export function answerToMarkdown(question: string, answer: string, citations: { n: number; source?: string | null; page?: number | null }[]): string {
  const refs = citations.map((c) => `[${c.n}] ${c.source ?? "unknown"}, page ${c.page ?? "?"}`).join("\n");
  return `**Q:** ${question}\n\n${answer}${refs ? `\n\n${refs}` : ""}\n`;
}

// ---------- passage highlighting ----------

/** Words for matching: lowercase, numbers kept whole ("27,360", "2.5"), punctuation dropped. */
export function matchTokens(text: string): string[] {
  return text.toLowerCase().match(/[\p{L}\p{N}]+(?:[.,][\p{N}]+)*/gu) ?? [];
}

/**
 * Which text items of a PDF page belong to the cited passage.
 *
 * 1. Every run of `n` consecutive words of the passage that also appears, in order, in the page text
 *    marks those page words (robust to table separators, line breaks and hyphenation).
 * 2. Only long continuous stretches of marked words are kept (at least `minRun` words, or the whole
 *    passage if it is shorter). A real passage is one long stretch; a common phrase that happens to
 *    recur elsewhere on the page ("is expected to") is only a few words and is dropped.
 */
export function matchItems(items: string[], passage: string, n = 3, minRun = 8): Set<number> {
  const words: { w: string; item: number }[] = [];
  items.forEach((s, item) => matchTokens(s).forEach((w) => words.push({ w, item })));
  const target = matchTokens(passage);
  const size = Math.min(n, target.length);
  const hits = new Set<number>();
  if (size === 0) return hits;
  const grams = new Set<string>();
  for (let i = 0; i + size <= target.length; i++) grams.add(target.slice(i, i + size).join(" "));
  const marked = new Array<boolean>(words.length).fill(false);
  for (let i = 0; i + size <= words.length; i++) {
    if (grams.has(words.slice(i, i + size).map((x) => x.w).join(" "))) marked.fill(true, i, i + size);
  }
  const needed = Math.min(minRun, target.length);
  for (let start = 0; start < words.length; ) {
    if (!marked[start]) { start++; continue; }
    let end = start;
    while (end < words.length && marked[end]) end++;
    if (end - start >= needed) for (let j = start; j < end; j++) hits.add(words[j].item);
    start = end;
  }
  return hits;
}

// ---------- status pill ----------

export type Health = { tone: "ok" | "warn" | "error"; label: string; problem?: string };

/** One-glance health of the selected index: can a question be answered right now? */
export function statusHealth(s: StatusOut): Health {
  if (s.store_error) return { tone: "error", label: "Database unavailable", problem: s.store_error };
  if (!s.embedding_ok) return { tone: "error", label: "Embeddings failing", problem: s.embedding_error ?? undefined };
  if (s.documents === 0) return { tone: "warn", label: "No documents", problem: "Upload PDFs to this index first." };
  if (s.llm_available === false)
    return { tone: "warn", label: "Default model unavailable", problem: s.llm_reason ?? "Pick another model in the question box." };
  return { tone: "ok", label: `${s.documents ?? 0} docs` };
}
