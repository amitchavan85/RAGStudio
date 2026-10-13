import { useQuery } from "@tanstack/react-query";
import { KeyRound } from "lucide-react";
import { api, getApiKey } from "../api/client";
import { useSettings } from "../settings";
import { ACCENTS, type Accent } from "../theme";
import { Button, cx, Label, Modal, Segmented } from "./ui";

const CHUNKING_TIPS: Record<string, string> = {
  recursive: "Splits by paragraph, line, then sentence. Best default (won our evaluation).",
  sentence: "Whole sentences only. Good for flowing prose.",
  document: "Splits by headings and keeps each section's title. Good for structured reports.",
  semantic: "Cuts where the topic changes. Slower to ingest.",
  token: "Sizes chunks by LLM tokens.",
  fixed: "Fixed character windows. Simplest, can cut words.",
  sliding_window: "Overlapping word windows. More repetition.",
  agentic: "An LLM decides the cuts. Slowest and costs money per page.",
};

export function SettingsDialog({ open, onOpenChange, onEditKey }: { open: boolean; onOpenChange: (o: boolean) => void; onEditKey: () => void }) {
  const { settings, update, reset } = useSettings();
  const options = useQuery({ queryKey: ["options"], queryFn: api.options, enabled: open });
  const d = options.data?.defaults;
  const maxK = options.data?.limits.max_k ?? 50;

  return (
    <Modal open={open} onOpenChange={onOpenChange} title="Settings" description="Saved in this browser." wide>
      <div className="space-y-6">
        <section className="space-y-4">
          <h3 className="text-xs font-medium uppercase tracking-wide text-text-muted">Appearance</h3>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <Label>Theme</Label>
            <Segmented
              ariaLabel="Theme"
              value={settings.theme}
              onChange={(theme) => update({ theme })}
              options={[{ id: "system", label: "System" }, { id: "light", label: "Light" }, { id: "dark", label: "Dark" }]}
            />
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <Label>Accent color</Label>
            <div role="radiogroup" aria-label="Accent color" className="flex gap-2">
              {(Object.keys(ACCENTS) as Accent[]).map((a) => (
                <button
                  key={a}
                  role="radio"
                  aria-checked={settings.accent === a}
                  aria-label={a}
                  title={a}
                  onClick={() => update({ accent: a })}
                  className={cx("h-7 w-7 rounded-full border-2 transition-transform hover:scale-110", settings.accent === a ? "border-text" : "border-transparent")}
                  style={{ background: ACCENTS[a].light.accent }}
                />
              ))}
            </div>
          </div>
        </section>

        <section className="space-y-4">
          <h3 className="text-xs font-medium uppercase tracking-wide text-text-muted">Answers</h3>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <Label hint="Hybrid combines meaning and exact words; it found the right page most often in our tests.">Search mode</Label>
            <Segmented
              ariaLabel="Search mode"
              value={settings.mode ?? ((d?.retrieval_mode as "hybrid") || "hybrid")}
              onChange={(mode) => update({ mode })}
              options={[{ id: "hybrid", label: "Hybrid" }, { id: "vector", label: "Meaning" }, { id: "keyword", label: "Keywords" }]}
            />
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <Label hint="More sources help comparisons across documents, but cost more tokens.">Sources per answer</Label>
            <input
              type="number"
              min={1}
              max={maxK}
              value={settings.k ?? d?.answer_k ?? 8}
              onChange={(e) => update({ k: Math.max(1, Math.min(maxK, Number(e.target.value) || 1)) })}
              className="h-9 w-20 rounded-lg border border-border bg-bg px-2 text-right text-sm"
            />
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <Label hint="Re-scores sources with a local model. More precise on wording, slower; it hurt bare-number lookups in our tests.">Reranker</Label>
            <Segmented
              ariaLabel="Reranker"
              value={settings.reranker ?? ((d?.reranker as "none") || "none")}
              onChange={(reranker) => update({ reranker })}
              options={[{ id: "none", label: "Off" }, { id: "cross_encoder", label: "On" }]}
            />
          </div>
        </section>

        <section className="space-y-2">
          <h3 className="text-xs font-medium uppercase tracking-wide text-text-muted">Uploads</h3>
          <Label hint={CHUNKING_TIPS[settings.chunking ?? d?.chunking_strategy ?? "recursive"]}>How documents are split</Label>
          <select
            value={settings.chunking ?? d?.chunking_strategy ?? "recursive"}
            onChange={(e) => update({ chunking: e.target.value })}
            className="h-9 w-full rounded-lg border border-border bg-bg px-2 text-sm"
          >
            {(options.data?.chunking_strategies ?? ["recursive"]).map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </section>

        <section className="flex items-center justify-between gap-3 border-t border-border pt-4">
          <Button variant="outline" size="sm" onClick={onEditKey}>
            <KeyRound size={14} /> {getApiKey() ? "Change API key" : "Add API key"}
          </Button>
          <Button variant="ghost" size="sm" onClick={reset}>Reset to server defaults</Button>
        </section>
      </div>
    </Modal>
  );
}
