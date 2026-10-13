/**
 * Design tokens (see UI.md, "Design guide"). Colors live here, not in components, so the theme
 * changes in one place. They are applied as CSS variables on <html>; Tailwind classes such as
 * `bg-surface` or `text-accent` read those variables (src/styles.css).
 */

export type Mode = "light" | "dark";
export type ThemePref = Mode | "system";
export type Accent = "indigo" | "teal" | "violet" | "blue";

type Tokens = Record<string, string>;

export const BASE: Record<Mode, Tokens> = {
  light: {
    bg: "#FAFAF9",
    surface: "#FFFFFF",
    "surface-2": "#F5F5F4",
    border: "#E7E5E4",
    text: "#1C1917",
    "text-muted": "#78716C",
    success: "#15803D",
    warning: "#B45309",
    danger: "#DC2626",
  },
  dark: {
    bg: "#0B0B0F",
    surface: "#15151B",
    "surface-2": "#1C1C24",
    border: "#26262E",
    text: "#EDEDF0",
    "text-muted": "#9A9AA5",
    success: "#4ADE80",
    warning: "#FBBF24",
    danger: "#F87171",
  },
};

/** accent: buttons/links/focus; on-accent: text on accent buttons; tint/tint-text: citation chips. */
export const ACCENTS: Record<Accent, Record<Mode, Tokens>> = {
  indigo: {
    light: { accent: "#4F46E5", "on-accent": "#FFFFFF", tint: "#EEF2FF", "tint-text": "#4338CA" },
    dark: { accent: "#818CF8", "on-accent": "#0B0B0F", tint: "#1E1B4B", "tint-text": "#C7D2FE" },
  },
  teal: {
    light: { accent: "#0F766E", "on-accent": "#FFFFFF", tint: "#F0FDFA", "tint-text": "#115E59" },
    dark: { accent: "#2DD4BF", "on-accent": "#0B0B0F", tint: "#042F2E", "tint-text": "#99F6E4" },
  },
  violet: {
    light: { accent: "#6D28D9", "on-accent": "#FFFFFF", tint: "#F5F3FF", "tint-text": "#5B21B6" },
    dark: { accent: "#A78BFA", "on-accent": "#0B0B0F", tint: "#2E1065", "tint-text": "#DDD6FE" },
  },
  blue: {
    light: { accent: "#1D4ED8", "on-accent": "#FFFFFF", tint: "#EFF6FF", "tint-text": "#1E40AF" },
    dark: { accent: "#60A5FA", "on-accent": "#0B0B0F", tint: "#172554", "tint-text": "#BFDBFE" },
  },
};

export function resolveMode(pref: ThemePref): Mode {
  if (pref !== "system") return pref;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function applyTheme(pref: ThemePref, accent: Accent): Mode {
  const mode = resolveMode(pref);
  const root = document.documentElement;
  const tokens = { ...BASE[mode], ...ACCENTS[accent][mode] };
  for (const [name, value] of Object.entries(tokens)) root.style.setProperty(`--${name}`, value);
  root.dataset.mode = mode;
  root.style.colorScheme = mode;
  return mode;
}

// ---- WCAG contrast (used by tests to keep every theme readable) ----

function luminance(hex: string): number {
  const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const [r, g, b] = c.map((x) => (x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}
