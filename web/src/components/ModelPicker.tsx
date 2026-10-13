import * as Popover from "@radix-ui/react-popover";
import { useQuery } from "@tanstack/react-query";
import { Check, ChevronDown } from "lucide-react";
import { api, type ModelOut } from "../api/client";
import { costPerAnswer, modelForTier, TIERS, type Tier } from "../lib/logic";
import { useSettings } from "../settings";
import { cx, Segmented } from "./ui";

/** The model actually used for the next question (exact pick, or the tier's choice). */
export function useChosenModel(): { model?: ModelOut; reason?: string } {
  const { settings } = useSettings();
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const options = useQuery({ queryKey: ["options"], queryFn: api.options });
  const list = models.data ?? [];
  if (settings.model) {
    const exact = list.find((m) => m.name === settings.model);
    if (exact?.available) return { model: exact };
    if (exact) return { reason: exact.reason ?? "This model is not available." };
  }
  const model = modelForTier(list, settings.tier, options.data?.defaults.llm_model);
  return model ? { model } : { reason: models.isLoading ? undefined : `No ${TIERS.find((t) => t.id === settings.tier)?.label} model is available on this server.` };
}

export function ModelPicker() {
  const { settings, update } = useSettings();
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const { model } = useChosenModel();
  const list = models.data ?? [];
  const tierAvailable = (t: Tier) => list.some((m) => m.tier === t && m.available);
  const providers = [...new Set(list.map((m) => m.provider))];

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Segmented
        ariaLabel="Model tier"
        value={settings.model ? ("" as Tier) : settings.tier}
        onChange={(tier) => update({ tier, model: null })}
        options={TIERS.map((t) => ({ id: t.id, label: t.label, title: t.hint, disabled: models.isSuccess && !tierAvailable(t.id) }))}
      />
      <Popover.Root>
        <Popover.Trigger asChild>
          <button className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs text-text-muted hover:bg-surface-2 hover:text-text">
            {model ? <>{model.name} · {costPerAnswer(model)}</> : "Choose model"}
            <ChevronDown size={14} />
          </button>
        </Popover.Trigger>
        <Popover.Portal>
          <Popover.Content align="start" sideOffset={6} className="z-50 max-h-96 w-80 overflow-y-auto rounded-xl border border-border bg-surface p-2 shadow-xl animate-fade-in">
            <div className="px-2 pb-2 pt-1 text-xs text-text-muted">Pick an exact model. Price is per answer (about 1,500 tokens in, 100 out).</div>
            {providers.map((p) => (
              <div key={p} className="mb-1">
                <div className="px-2 py-1 text-[11px] font-medium uppercase tracking-wide text-text-muted">{p}</div>
                {list.filter((m) => m.provider === p).map((m) => (
                  <Popover.Close asChild key={m.name}>
                    <button
                      disabled={!m.available}
                      title={m.reason ?? undefined}
                      onClick={() => update({ model: m.name })}
                      className={cx(
                        "flex w-full items-center justify-between gap-2 rounded-lg px-2 py-1.5 text-left text-sm disabled:cursor-not-allowed disabled:opacity-40",
                        settings.model === m.name ? "bg-tint text-tint-text" : "hover:bg-surface-2",
                      )}
                    >
                      <span className="flex items-center gap-2">
                        {settings.model === m.name ? <Check size={14} /> : <span className="w-3.5" />}
                        {m.name}
                      </span>
                      <span className="text-xs text-text-muted">{m.available ? costPerAnswer(m) : "unavailable"}</span>
                    </button>
                  </Popover.Close>
                ))}
              </div>
            ))}
          </Popover.Content>
        </Popover.Portal>
      </Popover.Root>
    </div>
  );
}
