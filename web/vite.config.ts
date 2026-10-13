import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// In development the UI runs on :5173 and forwards API calls to the FastAPI server on :8000.
// In production FastAPI serves the built files (web/dist) itself, so no proxy is needed.
const api = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy: { "/v1": api, "/health": api, "/ready": api } },
  build: { outDir: "dist", sourcemap: true },
  test: { environment: "node" },
});
