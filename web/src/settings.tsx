/** User preferences, persisted in the browser (localStorage). */
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import type { Tier } from "./lib/logic";
import { applyTheme, type Accent, type ThemePref } from "./theme";

export type Settings = {
  theme: ThemePref;
  accent: Accent;
  index: string;
  tier: Tier;
  model: string | null; // exact model from "Advanced"; null = pick by tier
  mode: "hybrid" | "vector" | "keyword" | null; // null = server default
  k: number | null;
  reranker: "none" | "cross_encoder" | null;
  chunking: string | null;
  sources: string[]; // document filter; [] = all documents
};

const DEFAULTS: Settings = {
  theme: "system",
  accent: "indigo",
  index: "default",
  tier: "economy",
  model: null,
  mode: null,
  k: null,
  reranker: null,
  chunking: null,
  sources: [],
};

const STORAGE = "rag.settings";

function load(): Settings {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(STORAGE) ?? "{}") };
  } catch {
    return DEFAULTS;
  }
}

type Ctx = { settings: Settings; update: (patch: Partial<Settings>) => void; reset: () => void };
const SettingsContext = createContext<Ctx | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(load);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE, JSON.stringify(settings));
    } catch {
      /* storage unavailable: settings last for this page only */
    }
  }, [settings]);

  // Apply the theme now and whenever the OS switches between light and dark.
  useEffect(() => {
    applyTheme(settings.theme, settings.accent);
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    const onChange = () => applyTheme(settings.theme, settings.accent);
    mq?.addEventListener("change", onChange);
    return () => mq?.removeEventListener("change", onChange);
  }, [settings.theme, settings.accent]);

  const update = (patch: Partial<Settings>) => setSettings((s) => ({ ...s, ...patch }));
  const reset = () => setSettings((s) => ({ ...DEFAULTS, theme: s.theme, accent: s.accent }));
  return <SettingsContext.Provider value={{ settings, update, reset }}>{children}</SettingsContext.Provider>;
}

export function useSettings(): Ctx {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("useSettings must be used inside SettingsProvider");
  return ctx;
}
