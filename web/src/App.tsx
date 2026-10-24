import * as Tooltip from "@radix-ui/react-tooltip";
import { useQueryClient } from "@tanstack/react-query";
import { Monitor, Moon, Settings as SettingsIcon, Sun } from "lucide-react";
import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { setUnauthorizedHandler } from "./api/client";
import { ApiKeyDialog } from "./components/ApiKeyDialog";
import { SettingsDialog } from "./components/SettingsDialog";
import { StatusPill } from "./components/StatusPill";
import { cx, IconButton } from "./components/ui";
import { AskPage } from "./pages/AskPage";
import { DocumentsPage } from "./pages/DocumentsPage";
import { useSettings } from "./settings";

export function App() {
  const { settings, update } = useSettings();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [keyOpen, setKeyOpen] = useState(false);
  const queryClient = useQueryClient();

  useEffect(() => setUnauthorizedHandler(() => setKeyOpen(true)), []);

  // Keyboard: "/" focuses the question box (unless already typing somewhere).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = e.target instanceof HTMLElement && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
      if (e.key === "/" && !typing) {
        const box = document.getElementById("question-box");
        if (box) {
          e.preventDefault();
          box.focus();
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const nextTheme = { system: "light", light: "dark", dark: "system" } as const;
  const ThemeIcon = settings.theme === "light" ? Sun : settings.theme === "dark" ? Moon : Monitor;

  return (
    <Tooltip.Provider>
      <div className="flex h-dvh flex-col">
        <header className="flex h-14 shrink-0 items-center gap-3 border-b border-border bg-surface px-3 sm:gap-6 sm:px-4">
          <div className="flex items-center gap-2 font-semibold tracking-tight">
            <span aria-hidden className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent text-sm text-on-accent">R</span>
            <span className="hidden sm:inline">RAG Studio</span>
          </div>
          <nav className="flex gap-1" aria-label="Main">
            {[["/", "Ask"], ["/documents", "Documents"]].map(([to, label]) => (
              <NavLink
                key={to}
                to={to}
                end
                className={({ isActive }) =>
                  cx("rounded-lg px-3 py-1.5 text-sm transition-colors", isActive ? "bg-surface-2 font-medium text-text" : "text-text-muted hover:text-text")
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-1">
            <StatusPill />
            <IconButton label={`Theme: ${settings.theme} (click to change)`} onClick={() => update({ theme: nextTheme[settings.theme] })}>
              <ThemeIcon size={18} />
            </IconButton>
            <IconButton label="Settings" onClick={() => setSettingsOpen(true)}>
              <SettingsIcon size={18} />
            </IconButton>
          </div>
        </header>

        <main className="min-h-0 flex-1">
          <Routes>
            <Route path="/" element={<AskPage />} />
            <Route path="/documents" element={<DocumentsPage />} />
            <Route path="*" element={<AskPage />} />
          </Routes>
        </main>
      </div>

      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} onEditKey={() => setKeyOpen(true)} />
      <ApiKeyDialog
        open={keyOpen}
        onOpenChange={setKeyOpen}
        onSaved={() => queryClient.invalidateQueries()}
      />
    </Tooltip.Provider>
  );
}
