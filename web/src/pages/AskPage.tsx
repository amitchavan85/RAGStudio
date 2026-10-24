import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowUp, BookOpen, Check, Copy, FileSearch, FileText, PanelLeft, RotateCcw, Search, SearchX, UploadCloud, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, type AnswerResponse, type Citation, type ChunkOut } from "../api/client";
import { ModelPicker, useChosenModel } from "../components/ModelPicker";
import { PdfViewer } from "../components/PdfViewer";
import { Badge, Button, cx, IconButton, Notice, Segmented, Tip } from "../components/ui";
import { answerToMarkdown, formatMs, friendlyError, splitCitations } from "../lib/logic";
import { useSettings } from "../settings";

type Turn = {
  id: string;
  question: string;
  index: string;
  kind: "answer" | "search"; // search = passages only, no LLM
  status: "pending" | "done" | "error";
  answer?: AnswerResponse;
  results?: ChunkOut[];
  error?: unknown;
  ms?: number;
};

type Opened = { source: string; page: number; snippet?: string; passage?: string; n?: number; index: string };

const HISTORY = "rag.history";

function loadHistory(): Turn[] {
  try {
    return (JSON.parse(sessionStorage.getItem(HISTORY) ?? "[]") as Turn[])
      .filter((t) => t.status === "done")
      .map((t) => ({ ...t, kind: t.kind ?? "answer" }));
  } catch {
    return [];
  }
}

export function AskPage() {
  const { settings } = useSettings();
  const [turns, setTurns] = useState<Turn[]>(loadHistory);
  const [opened, setOpened] = useState<Opened | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    try {
      sessionStorage.setItem(HISTORY, JSON.stringify(turns.filter((t) => t.status === "done").slice(-30)));
    } catch {
      /* storage unavailable */
    }
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  const { model } = useChosenModel();
  const retrieval = () => ({
    mode: settings.mode ?? undefined,
    k: settings.k ?? undefined,
    reranker: settings.reranker ?? undefined,
    filters: settings.sources.length ? { source: settings.sources } : undefined,
  });
  const search = useMutation({
    mutationFn: (question: string) => api.search({ question, index: settings.index, retrieval: retrieval() }),
  });
  const ask = useMutation({
    mutationFn: (question: string) =>
      api.answer({
        question,
        index: settings.index,
        retrieval: retrieval(),
        llm: { model: model?.name, max_tokens: 2048 },
        include_sources: true,
      }),
  });

  const submit = async (question: string, kind: Turn["kind"] = settings.askMode) => {
    const id = crypto.randomUUID();
    const started = performance.now();
    setTurns((t) => [...t, { id, question, index: settings.index, kind, status: "pending" }]);
    try {
      const done =
        kind === "search"
          ? { results: (await search.mutateAsync(question)).results }
          : { answer: await ask.mutateAsync(question) };
      const ms = performance.now() - started;
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, ...done, status: "done", ms } : x)));
    } catch (error) {
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, status: "error", error } : x)));
    }
  };

  const [drawer, setDrawer] = useState(false);
  const docs = useQuery({ queryKey: ["documents", settings.index], queryFn: () => api.documents(settings.index) });
  const docCount = docs.data?.documents.length;

  return (
    <div className="flex h-full">
      <Sidebar mobileOpen={drawer} onClose={() => setDrawer(false)} />
      <section className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-2 border-b border-border bg-surface px-4 py-2 md:hidden">
          <Button size="sm" variant="ghost" onClick={() => setDrawer(true)} aria-label="Index and documents">
            <PanelLeft size={16} /> {settings.index}
            {settings.sources.length > 0 ? ` · ${settings.sources.length} selected` : docCount != null ? ` · ${docCount} docs` : ""}
          </Button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl space-y-8 px-4 py-8 sm:px-6">
            {turns.length === 0 && <EmptyState onPick={submit} docCount={docCount} index={settings.index} />}
            {turns.map((t) => (
              <TurnView
                key={t.id}
                turn={t}
                onRetry={() => {
                  setTurns((all) => all.filter((x) => x.id !== t.id));
                  submit(t.question, t.kind);
                }}
                onOpen={(o) => setOpened({ ...o, index: t.index })}
              />
            ))}
            <div ref={endRef} />
          </div>
        </div>
        <QuestionBox onSubmit={submit} busy={ask.isPending || search.isPending} onClear={turns.length ? () => setTurns([]) : undefined} />
      </section>
      {opened && <SourcePanel opened={opened} onClose={() => setOpened(null)} />}
    </div>
  );
}

function Sidebar({ mobileOpen, onClose }: { mobileOpen: boolean; onClose: () => void }) {
  useEffect(() => {
    if (!mobileOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mobileOpen, onClose]);
  return (
    <>
      <aside className="hidden w-64 shrink-0 flex-col gap-6 overflow-y-auto border-r border-border bg-surface p-4 md:flex">
        <SidebarContent />
      </aside>
      {mobileOpen && (
        <div className="fixed inset-0 z-40 md:hidden" role="dialog" aria-modal="true" aria-label="Index and documents">
          <div className="absolute inset-0 bg-black/40 animate-fade-in" onClick={onClose} />
          <aside className="absolute inset-y-0 left-0 flex w-[min(85vw,320px)] flex-col gap-6 overflow-y-auto border-r border-border bg-surface p-4 shadow-2xl animate-slide-in">
            <div className="flex items-center justify-between">
              <span className="font-medium">Index & documents</span>
              <IconButton label="Close" onClick={onClose}><X size={18} /></IconButton>
            </div>
            <SidebarContent />
          </aside>
        </div>
      )}
    </>
  );
}

function SidebarContent() {
  const { settings, update } = useSettings();
  const indexes = useQuery({ queryKey: ["indexes"], queryFn: api.indexes });
  const docs = useQuery({ queryKey: ["documents", settings.index], queryFn: () => api.documents(settings.index) });
  const list = docs.data?.documents ?? [];
  const toggle = (source: string) =>
    update({ sources: settings.sources.includes(source) ? settings.sources.filter((s) => s !== source) : [...settings.sources, source] });

  return (
    <>
      <div>
        <label htmlFor="index-select" className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-text-muted">Index</label>
        <select
          id="index-select"
          value={settings.index}
          onChange={(e) => update({ index: e.target.value, sources: [] })}
          className="h-9 w-full rounded-lg border border-border bg-bg px-2 text-sm"
        >
          {(indexes.data ?? [{ name: settings.index, documents: null }]).map((i) => (
            <option key={i.name} value={i.name}>
              {i.name}{i.documents != null ? ` (${i.documents} docs)` : ""}
            </option>
          ))}
        </select>
      </div>
      <div className="min-h-0">
        <div className="mb-1.5 flex items-center justify-between">
          <span className="text-xs font-medium uppercase tracking-wide text-text-muted">Search in</span>
          {settings.sources.length > 0 && (
            <button className="text-xs text-accent hover:underline" onClick={() => update({ sources: [] })}>All documents</button>
          )}
        </div>
        {docs.isLoading && <div className="space-y-2">{[0, 1, 2].map((i) => <div key={i} className="skeleton h-7" />)}</div>}
        {docs.isSuccess && list.length === 0 && (
          <p className="text-sm text-text-muted">
            No documents yet. <Link to="/documents" className="text-accent hover:underline">Upload PDFs</Link>
          </p>
        )}
        <ul className="space-y-0.5">
          {list.map((d) => {
            const on = settings.sources.includes(d.source);
            return (
              <li key={d.source}>
                <button
                  onClick={() => toggle(d.source)}
                  aria-pressed={on}
                  className={cx("flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-sm", on ? "bg-tint text-tint-text" : "hover:bg-surface-2")}
                >
                  {on ? <Check size={14} className="shrink-0" /> : <FileText size={14} className="shrink-0 text-text-muted" />}
                  <span className="truncate" title={d.source}>{d.source}</span>
                </button>
              </li>
            );
          })}
        </ul>
        {list.length > 0 && settings.sources.length === 0 && <p className="mt-2 text-xs text-text-muted">Searching all documents. Click to narrow down.</p>}
      </div>
    </>
  );
}

function EmptyState({ onPick, docCount, index }: { onPick: (q: string) => void; docCount?: number; index: string }) {
  const examples = ["What is the main conclusion of the report?", "Summarize the outlook in three points.", "Which figures changed the most?"];
  if (docCount === 0) {
    return (
      <div className="flex flex-col items-center pt-16 text-center animate-fade-in">
        <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-tint text-tint-text">
          <UploadCloud size={22} />
        </div>
        <h1 className="text-2xl font-semibold tracking-tight">No documents in “{index}” yet</h1>
        <p className="mt-2 max-w-md text-text-muted">
          Upload PDFs first; answers come only from them. (If you just changed the embedding model, re-upload: each
          embedding model has its own index.)
        </p>
        <Link to="/documents" className="mt-6 inline-flex h-10 items-center gap-2 rounded-lg bg-accent px-4 text-sm font-medium text-on-accent hover:opacity-90">
          <UploadCloud size={16} /> Upload documents
        </Link>
      </div>
    );
  }
  return (
    <div className="flex flex-col items-center pt-16 text-center animate-fade-in">
      <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-tint text-tint-text">
        <BookOpen size={22} />
      </div>
      <h1 className="text-2xl font-semibold tracking-tight">Ask your documents</h1>
      <p className="mt-2 max-w-md text-text-muted">Answers come only from your PDFs, and every claim links to the page it came from.</p>
      <div className="mt-6 flex flex-wrap justify-center gap-2">
        {examples.map((q) => (
          <button key={q} onClick={() => onPick(q)} className="rounded-full border border-border bg-surface px-3 py-1.5 text-sm text-text-muted hover:border-accent hover:text-text">
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

function TurnView({ turn, onRetry, onOpen }: { turn: Turn; onRetry: () => void; onOpen: (o: Omit<Opened, "index">) => void }) {
  return (
    <article className="animate-fade-in" aria-busy={turn.status === "pending"}>
      <h2 className="mb-3 text-lg font-semibold leading-snug">{turn.question}</h2>
      {turn.status === "pending" && <Thinking searchOnly={turn.kind === "search"} />}
      {turn.status === "error" && (
        <div className="flex items-center gap-3">
          <Notice>{friendlyError(turn.error)}</Notice>
          <Button size="sm" onClick={onRetry}><RotateCcw size={14} /> Retry</Button>
        </div>
      )}
      {turn.status === "done" && turn.kind === "answer" && turn.answer && <AnswerView turn={turn} answer={turn.answer} onOpen={onOpen} />}
      {turn.status === "done" && turn.kind === "search" && <SearchResults results={turn.results ?? []} ms={turn.ms} onOpen={onOpen} />}
    </article>
  );
}

function Thinking({ searchOnly = false }: { searchOnly?: boolean }) {
  const [phase, setPhase] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setPhase(1), 900);
    return () => clearTimeout(t);
  }, []);
  return (
    <div className="space-y-2.5" role="status">
      <p className="text-sm text-text-muted">{phase === 0 || searchOnly ? "Searching your documents…" : "Writing the answer…"}</p>
      <div className="skeleton h-4 w-full" />
      <div className="skeleton h-4 w-11/12" />
      <div className="skeleton h-4 w-2/3" />
    </div>
  );
}

function AnswerView({ turn, answer, onOpen }: { turn: Turn; answer: AnswerResponse; onOpen: (o: Omit<Opened, "index">) => void }) {
  const [copied, setCopied] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const byN = useMemo(() => new Map(answer.citations.map((c) => [c.n, c])), [answer.citations]);
  // Full chunk text (citations carry a 300-character snippet), used to highlight the passage in the PDF.
  const fullText = useMemo(() => new Map((answer.sources ?? []).map((s) => [s.chunk_id, s.text])), [answer.sources]);
  const openCitation = (c: Citation) =>
    onOpen({ source: c.source ?? "", page: c.page ?? 1, snippet: c.snippet, passage: fullText.get(c.chunk_id) ?? c.snippet, n: c.n });
  const segments = useMemo(() => splitCitations(answer.answer, new Set(byN.keys())), [answer.answer, byN]);

  if (!answer.found) {
    return (
      <div className="rounded-xl border border-border bg-surface p-5">
        <div className="flex items-start gap-3">
          <SearchX size={20} className="mt-0.5 shrink-0 text-text-muted" />
          <div>
            <p className="font-medium">Not found in your documents</p>
            <p className="mt-1 text-sm text-text-muted">The answer isn't in the selected documents, so nothing was made up. Try rephrasing, searching all documents, or <Link to="/documents" className="text-accent hover:underline">uploading</Link> the right PDF.</p>
          </div>
        </div>
      </div>
    );
  }

  const copy = async () => {
    await navigator.clipboard.writeText(answerToMarkdown(turn.question, answer.answer, answer.citations));
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="group">
      <p className="whitespace-pre-wrap text-[16px] leading-relaxed">
        {segments.map((s, i) =>
          s.kind === "text" ? <span key={i}>{s.text}</span> : <CitationChip key={i} c={byN.get(s.n)!} onOpen={openCitation} />,
        )}
      </p>

      {answer.citations.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {answer.citations.map((c) => (
            <button
              key={c.n}
              onClick={() => openCitation(c)}
              className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-2.5 py-1 text-xs text-text-muted hover:border-accent hover:text-text"
            >
              <span className="font-medium text-tint-text">[{c.n}]</span> {c.source} · p{c.page}
            </button>
          ))}
        </div>
      )}

      <footer className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-text-muted">
        <span>{answer.usage.model}</span>
        <span>{answer.usage.input_tokens.toLocaleString()} in · {answer.usage.output_tokens.toLocaleString()} out tokens</span>
        {turn.ms != null && <span>{formatMs(turn.ms)}</span>}
        <span className="ml-auto flex gap-1 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
          <IconButton label={copied ? "Copied" : "Copy answer with citations (Markdown)"} onClick={copy}>
            {copied ? <Check size={15} /> : <Copy size={15} />}
          </IconButton>
          {answer.sources && (
            <IconButton label={showAll ? "Hide retrieved sources" : "Show all retrieved sources"} onClick={() => setShowAll((s) => !s)}>
              <FileSearch size={15} />
            </IconButton>
          )}
        </span>
      </footer>

      {showAll && answer.sources && <RetrievedList sources={answer.sources} onOpen={onOpen} />}
    </div>
  );
}

function CitationChip({ c, onOpen }: { c: Citation; onOpen: (c: Citation) => void }) {
  return (
    <Tip
      text={
        <span className="block">
          <span className="font-medium">{c.source}, page {c.page}</span>
          <span className="mt-1 block text-text-muted">{snippetText(c.snippet).slice(0, 180)}…</span>
        </span>
      }
    >
      <button
        onClick={() => onOpen(c)}
        aria-label={`Source ${c.n}: ${c.source}, page ${c.page}`}
        className="mx-0.5 inline-flex h-5 min-w-5 -translate-y-px items-center justify-center rounded-md bg-tint px-1 align-middle text-[11px] font-semibold text-tint-text hover:ring-2 hover:ring-accent/40"
      >
        {c.n}
      </button>
    </Tip>
  );
}

function RetrievedList({ sources, onOpen }: { sources: ChunkOut[]; onOpen: (o: Omit<Opened, "index">) => void }) {
  return (
    <div className="mt-4 space-y-2 rounded-xl border border-border bg-surface p-3 animate-fade-in">
      <p className="text-xs font-medium uppercase tracking-wide text-text-muted">All retrieved sources ({sources.length})</p>
      {sources.map((s, i) => (
        <button
          key={s.chunk_id ?? i}
          onClick={() => onOpen({ source: s.source ?? "", page: s.page ?? 1, snippet: s.text, passage: s.text })}
          className="block w-full rounded-lg p-2 text-left hover:bg-surface-2"
        >
          <div className="flex items-center gap-2 text-xs text-text-muted">
            <Badge>{i + 1}</Badge> {s.source} · p{s.page}
            <span className="ml-auto">{Object.entries(s.scores).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(" · ")}</span>
          </div>
          <p className="mt-1 line-clamp-2 text-sm">{s.text}</p>
        </button>
      ))}
    </div>
  );
}

function SourcePanel({ opened, onClose }: { opened: Opened; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <aside className="flex w-[min(46vw,640px)] shrink-0 flex-col gap-3 border-l border-border bg-surface p-4 animate-slide-in" aria-label="Source viewer">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate font-medium" title={opened.source}>{opened.n != null && <span className="text-tint-text">[{opened.n}] </span>}{opened.source}</p>
          <p className="text-xs text-text-muted">Page {opened.page}</p>
        </div>
        <IconButton label="Close source viewer (Esc)" onClick={onClose}><X size={18} /></IconButton>
      </div>
      {opened.snippet && (
        <blockquote className="max-h-32 overflow-y-auto rounded-lg border-l-4 border-accent bg-tint px-3 py-2 text-sm text-tint-text">
          {snippetText(opened.snippet)}
        </blockquote>
      )}
      <PdfViewer index={opened.index} source={opened.source} page={opened.page} highlight={opened.passage ?? opened.snippet} />
    </aside>
  );
}

/** Search-only mode: ranked passages straight from retrieval (no LLM); each opens its PDF page. */
function SearchResults({ results, ms, onOpen }: { results: ChunkOut[]; ms?: number; onOpen: (o: Omit<Opened, "index">) => void }) {
  if (results.length === 0) {
    return (
      <div className="flex items-start gap-3 rounded-xl border border-border bg-surface p-5">
        <SearchX size={20} className="mt-0.5 shrink-0 text-text-muted" />
        <div>
          <p className="font-medium">No matching passages</p>
          <p className="mt-1 text-sm text-text-muted">Try other words, or search all documents.</p>
        </div>
      </div>
    );
  }
  return (
    <div className="space-y-2">
      <p className="text-xs text-text-muted">
        {results.length} passages · no AI used (free){ms != null ? ` · ${formatMs(ms)}` : ""}
      </p>
      {results.map((r, i) => (
        <button
          key={r.chunk_id ?? i}
          onClick={() => onOpen({ source: r.source ?? "", page: r.page ?? 1, snippet: r.text, passage: r.text })}
          className="block w-full rounded-xl border border-border bg-surface p-4 text-left transition-colors hover:border-accent"
        >
          <div className="flex flex-wrap items-center gap-2 text-xs text-text-muted">
            <Badge tone="accent">{i + 1}</Badge>
            <span className="font-medium text-text">{r.source}</span> · page {r.page}
            <span className="ml-auto">{Object.entries(r.scores).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(" · ")}</span>
          </div>
          <p className="mt-2 line-clamp-4 text-sm leading-relaxed">{snippetText(r.text)}</p>
        </button>
      ))}
    </div>
  );
}

/** Chunks overlap, so a snippet can start mid-word: mark that with an ellipsis. */
function snippetText(s: string): string {
  const t = s.trim();
  return /^[a-z]/.test(t) ? `…${t}` : t;
}

function QuestionBox({ onSubmit, busy, onClear }: { onSubmit: (q: string) => void; busy: boolean; onClear?: () => void }) {
  const [text, setText] = useState("");
  const { settings, update } = useSettings();
  const searchOnly = settings.askMode === "search";
  const modelReason = useChosenModel().reason;
  const reason = searchOnly ? undefined : modelReason; // search needs no model
  const ref = useRef<HTMLTextAreaElement>(null);
  const send = () => {
    const q = text.trim();
    if (!q || busy || reason) return;
    onSubmit(q);
    setText("");
  };
  useEffect(() => {
    const el = ref.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
    }
  }, [text]);

  return (
    <div className="border-t border-border bg-bg px-6 py-4">
      <div className="mx-auto max-w-3xl">
        <div className="rounded-2xl border border-border bg-surface p-3 shadow-sm focus-within:border-accent">
          <textarea
            id="question-box"
            ref={ref}
            rows={1}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey || !e.shiftKey)) {
                e.preventDefault();
                send();
              }
            }}
            placeholder={searchOnly ? "Search your documents…" : "Ask a question about your documents…"}
            aria-label="Question"
            className="max-h-[200px] w-full resize-none bg-transparent px-1 text-[15px] leading-relaxed outline-none placeholder:text-text-muted"
          />
          <div className="mt-2 flex items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <Segmented
                ariaLabel="Answer or search only"
                value={settings.askMode}
                onChange={(askMode) => update({ askMode })}
                options={[
                  { id: "answer", label: "Ask", title: "An AI model answers from your documents, with citations" },
                  { id: "search", label: "Search", title: "Find passages only: no AI, free and instant" },
                ]}
              />
              {searchOnly ? (
                <span className="inline-flex items-center gap-1 text-xs text-text-muted"><Search size={13} /> Passages only · free</span>
              ) : (
                <ModelPicker />
              )}
            </div>
            <div className="flex items-center gap-1">
              {onClear && <Button variant="ghost" size="sm" onClick={onClear}>Clear</Button>}
              <button
                onClick={send}
                disabled={!text.trim() || busy || !!reason}
                aria-label={searchOnly ? "Search" : "Ask"}
                className="inline-flex h-9 w-9 items-center justify-center rounded-xl bg-accent text-on-accent transition-opacity hover:opacity-90 disabled:opacity-40"
              >
                <ArrowUp size={18} />
              </button>
            </div>
          </div>
        </div>
        {reason ? (
          <p className="mt-2 text-xs text-warning">{reason}</p>
        ) : (
          <p className="mt-2 hidden text-center text-xs text-text-muted sm:block">Enter to {searchOnly ? "search" : "ask"} · Shift+Enter for a new line · / to focus</p>
        )}
      </div>
    </div>
  );
}
