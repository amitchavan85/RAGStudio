import * as Popover from "@radix-ui/react-popover";
import { useQuery } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { friendlyError, statusHealth } from "../lib/logic";
import { useSettings } from "../settings";
import { cx } from "./ui";

const DOT = { ok: "bg-success", warn: "bg-warning", error: "bg-danger" } as const;

/** Header indicator: is the selected index ready to answer (documents, embeddings, model)? */
export function StatusPill() {
  const { settings } = useSettings();
  const status = useQuery({
    queryKey: ["status", settings.index],
    queryFn: () => api.status(settings.index),
    refetchInterval: 60_000,
  });

  if (status.isLoading) return <div className="skeleton h-7 w-28" aria-label="Checking status" />;
  if (status.isError) {
    return (
      <span className="inline-flex items-center gap-2 rounded-full border border-border px-3 py-1 text-xs text-danger" role="status">
        <span className="h-2 w-2 rounded-full bg-danger" /> {friendlyError(status.error)}
      </span>
    );
  }
  const s = status.data!;
  const h = statusHealth(s);

  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <button
          className="inline-flex items-center gap-2 rounded-full border border-border px-3 py-1 text-xs text-text-muted hover:bg-surface-2 hover:text-text"
          aria-label={`Status of index ${s.index}: ${h.label}`}
        >
          <span className={cx("h-2 w-2 rounded-full", DOT[h.tone])} aria-hidden />
          <span className="font-medium text-text">{s.index}</span>
          <span className="hidden sm:inline">· {h.label}</span>
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content align="end" sideOffset={8} className="z-50 w-80 rounded-xl border border-border bg-surface p-4 text-sm shadow-xl animate-fade-in">
          <div className="mb-3 flex items-center justify-between">
            <span className="font-medium">Index “{s.index}”</span>
            <button className="inline-flex items-center gap-1 text-xs text-text-muted hover:text-text" onClick={() => status.refetch()}>
              <RefreshCw size={12} className={status.isFetching ? "animate-spin" : ""} /> Check again
            </button>
          </div>
          {h.problem && (
            <p className={cx("mb-3 rounded-lg border px-3 py-2 text-xs", h.tone === "error" ? "border-danger/40 text-danger" : "border-warning/40 text-warning")}>
              {h.problem}
            </p>
          )}
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-xs">
            <dt className="text-text-muted">Documents</dt>
            <dd>{s.documents ?? "?"} ({s.chunks ?? "?"} chunks)</dd>
            <dt className="text-text-muted">Embeddings</dt>
            <dd className={s.embedding_ok ? "" : "text-danger"}>
              {s.embedding_provider} · {s.embedding_model} {s.embedding_ok ? "✓" : "✗"}
            </dd>
            <dt className="text-text-muted">Default model</dt>
            <dd className={s.llm_available === false ? "text-warning" : ""}>
              {s.llm_model} {s.llm_available === true ? "✓" : s.llm_available === false ? "✗" : ""}
            </dd>
            <dt className="text-text-muted">Store</dt>
            <dd className="break-all">{s.collection}</dd>
          </dl>
          {s.documents === 0 && (
            <Link to="/documents" className="mt-3 inline-block text-xs text-accent hover:underline">Upload documents →</Link>
          )}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
