# Web UI plan

A web app on top of the RAG API (`/v1/*`). Goal: users ask questions about their documents, **trust** the answer (every claim is cited and one click away from its page), and **choose the model by cost**.

Stack: React + TypeScript (Vite), TanStack Query, `pdf.js`, typed API client generated from `/openapi.json`. Built files are served by the FastAPI app (one container).

## Features

### Release 1: MVP

**Ask (main screen)**
1. Question box and answer view with `[1]` `[2]` citation markers.
2. Clickable citations: open the PDF viewer at the cited page, with the cited passage shown next to it.
3. Sources panel: file, page and snippet of each cited chunk; toggle to show all retrieved chunks.
4. Clear "not found" state, with suggestions (rephrase, choose other documents, upload).
5. Model picker: Cheap / Balanced / Best / Local (free), plus an "Advanced" list of every model with provider and price; unavailable models greyed out with the reason.
6. Answer footer: model used, tokens in/out, response time.
7. Document filter: ask across all documents or only selected ones.

**Documents**
8. Drag-and-drop PDF upload with per-file progress (queued, processing, done, failed).
9. Result per file: pages and chunks, "replaced old version", or why it was skipped.
10. Document list (name, chunks) with delete and confirmation.

**General**
11. Index selector with document counts.
12. API key dialog when the server answers 401; key remembered in the browser.
13. Friendly error messages mapped from API error codes (rate limited, provider down, file too large, ...).

### Release 2
14. PDF viewer with page navigation and zoom; cited passage highlighted where possible.
15. Streaming answers (text appears as it is generated).
16. Settings panel (collapsed by default): retrieval mode, number of sources (k), reranker, chunking strategy for uploads, each with a short "when to use" tip.
17. Search-only mode: find passages without an LLM call (free, instant).
18. Copy / export an answer with citations (Markdown).
19. Question history for the browser session.

### Release 3
20. Login with users and teams (beyond API keys).
21. Saved conversations with follow-up questions.
22. Compare mode: 2-3 models answer side by side, with cost and latency.
23. Thumbs up/down feedback, collected for evaluation.
24. Cost per answer and per day (P3 cost features).
25. Admin view: index health, evaluation runs, usage.

### Cross-cutting
- Responsive (laptop first, usable on tablet), light and dark theme.
- Keyboard: `/` focuses the question box, `Ctrl+Enter` asks.
- Accessible: keyboard navigation, screen-reader-friendly citations, WCAG AA contrast.

**Implemented (2026-10-13):** Release 1 (features 1-13) + PDF viewer (14) + settings panel (16) + copy as Markdown (18) + session history (19). Also done: the cited passage is highlighted on the PDF page (14). Not yet: streaming (15), search-only mode (17), Release 3.

## Run it

```bash
# Production-like: build once, FastAPI serves the UI at http://127.0.0.1:8000
cd web && npm install && npm run build && cd ..
python -m src.api

# Development with hot reload: API on :8000, UI on http://localhost:5173 (calls are proxied to the API)
python -m src.api
cd web && npm run dev

# Docker: the image builds the UI in a Node stage (docker compose --profile api up -d --build api)
```

Web scripts (in `web/`): `npm run typecheck`, `npm test` (Vitest), `npm run build`, `npm run api-types` (regenerate `src/api/schema.d.ts` from the API after backend changes).

## Code map (`web/src`)

| File | What |
|---|---|
| `main.tsx`, `App.tsx` | providers, header, routes (`/` Ask, `/documents`), `/` keyboard shortcut, API-key prompt on 401 |
| `pages/AskPage.tsx` | sidebar (index, document filter), answers with citation chips, sources panel, question box |
| `pages/DocumentsPage.tsx` | drag-and-drop upload, job polling, document list with view/delete |
| `components/PdfViewer.tsx` | pdf.js viewer (loaded on first use), page navigation, zoom, cited-passage highlight |
| `components/ModelPicker.tsx` | Cheap / Balanced / Best / Local tiers + exact model list with cost per answer |
| `components/SettingsDialog.tsx` | theme, accent, search mode, sources per answer, reranker, chunking, API key |
| `api/client.ts`, `api/schema.d.ts` | typed API client; schema generated from FastAPI's OpenAPI |
| `lib/logic.ts` | citation parsing, tier-to-model choice, cost hints, friendly error messages, passage matching for highlights |
| `theme.ts`, `styles.css` | design tokens (4 accents x light/dark) mapped to Tailwind colors |

## Verified
- Typecheck, 21 unit tests (including a WCAG AA contrast check of every accent in light and dark), production build (main bundle 135 KB gzip; pdf.js loaded only when a PDF is opened).
- In a real browser (Edge, headless, light and dark): ask -> answer with citation chips in ~2-3 s -> click a citation -> the PDF opens at the cited page; Documents page lists and views files; no browser errors.
- Bugs found that way and fixed: the PDF worker was served as `text/plain` on Windows (browsers refuse module scripts with that type); unknown `/v1/...` URLs returned the web page instead of a JSON 404.
- **Passage highlighting:** the full cited chunk (not the 300-character snippet) is matched against the page's text layer: every 3 consecutive words that also appear in order on the page are marked, and only continuous stretches of at least 8 marked words are kept. In the browser this highlighted exactly the cited paragraph on a two-column World Bank page and the cited table rows on a GDP page. The first version (without the stretch rule) also lit up common phrases elsewhere on the page ("is expected to", "Over 2027-28"); the stretch rule removed them. If nothing matches (e.g. OCR'd pages), the viewer says "Passage not located on this page" and the passage stays visible above the page.
- Known small issue: a cited passage can start mid-word because chunks overlap ("S Dollar ..."); lowercase starts get an ellipsis.

## Backend work needed

| Feature | API today |
|---|---|
| 1-13 (except opening a PDF) | Ready |
| 2, 14: open the PDF at a page | Done: `GET /v1/documents/{source}/file` (PDFs uploaded through the API are kept; CLI-ingested ones are not) |
| 15: streaming | Streaming in the LLM providers + SSE endpoint |
| 20-25 | Users, a database for history/feedback, cost features |

## API mapping

| UI action | Endpoint |
|---|---|
| Ask a question | `POST /v1/answer` |
| Search passages only | `POST /v1/search` |
| Upload PDFs | `POST /v1/documents`, then poll `GET /v1/jobs/{id}` |
| List / delete documents | `GET /v1/documents`, `DELETE /v1/documents/{source}` |
| Open a cited PDF | `GET /v1/documents/{source}/file` |
| Fill pickers | `GET /v1/models`, `GET /v1/indexes`, `GET /v1/options` |

## Design guide

**Character: calm, precise, trustworthy.** A research tool, not a chat toy: the content (answers, sources) is the hero, the interface steps back. References for the feel: Linear, Vercel dashboard, Perplexity, Notion.

### Principles
1. **Content first.** Generous whitespace, one primary action per screen, no decorative clutter.
2. **Evidence is always visible.** Citations look like interactive chips, the source panel sits next to the answer, and opening a page takes one click.
3. **Progressive disclosure.** Simple by default (question box + model tier); advanced settings live in a collapsed panel.
4. **Honest states.** Loading, empty, "not found", and error states are designed, never raw text or a blank screen.
5. **Quiet motion.** Short transitions (150-200 ms) for panels and new content; no bouncing; respect "reduce motion".

### Color (design tokens)
Neutral base with one accent; color carries meaning, not decoration.

| Token | Light | Dark | Use |
|---|---|---|---|
| `bg` | `#FAFAF9` | `#0B0B0F` | page background |
| `surface` | `#FFFFFF` | `#15151B` | cards, panels |
| `border` | `#E7E5E4` | `#26262E` | 1px dividers |
| `text` | `#1C1917` | `#EDEDF0` | body |
| `text-muted` | `#78716C` | `#9A9AA5` | metadata, captions |
| `accent` | `#4F46E5` (indigo) | `#818CF8` | primary buttons, focus, links |
| `citation` | `#EEF2FF` bg / `#4338CA` text | `#1E1B4B` / `#C7D2FE` | citation chips, highlighted passage |
| `success` / `warning` / `danger` | `#15803D` / `#B45309` / `#DC2626` | `#4ADE80` / `#FBBF24` / `#F87171` | job states, errors |

All text/background pairs above were checked and meet WCAG AA (at least 4.5:1; lowest is muted text on the light background at 4.59). The brighter, more common green/amber shades (`#16A34A`, `#D97706`) only reach about 3.2:1 on white, so they are not used for text. Dark mode is a first-class theme (system preference by default, manual toggle).

### Typography
- **Inter** for UI and answers (variable font; tabular numbers for figures so tables of numbers align), **JetBrains Mono** for code/IDs.
- Scale: 12 / 14 / 16 / 20 / 24 / 32 px; body 16 px, line height 1.6 for answers (they are read, not scanned); max line length ~70 characters.
- Weights: 400 body, 500 labels, 600 headings. No all-caps except tiny section labels.

### Layout and spacing
- 4 px spacing grid (4, 8, 12, 16, 24, 32, 48).
- Ask screen: left sidebar (index, documents filter, history), center answer column (max ~720 px), right sources/PDF panel that slides in when a citation is clicked.
- Radius 8 px (inputs, buttons), 12 px (cards); soft shadow only on floating elements (dialogs, popovers); otherwise 1 px borders.

### Components
- **Question box:** large, auto-growing, at the bottom of the answer column (chat convention) with the model tier selector beside the send button.
- **Answer card:** plain text with inline citation chips `[1]`; footer in muted text (model, tokens, time); actions (copy, export) appear on hover.
- **Citation chip:** small rounded pill in `citation` colors; hover shows file + page + snippet preview; click opens the PDF panel at that page.
- **Model picker:** segmented control (Cheap / Balanced / Best / Local) with price hint ("~$0.0002 per answer"); "Advanced" opens a searchable list grouped by provider.
- **Upload zone:** dashed border, large drop target; each file becomes a row with status badge and progress.
- **Toasts** for background events (upload finished), **inline messages** for errors in context.
- **Skeletons** (not spinners) while answers load; a subtle "searching documents... / writing answer..." status line.

### Accessibility and quality bar
- Every interactive element reachable by keyboard with a visible focus ring (accent, 2 px).
- Citations are links with descriptive labels ("Source 1: usa_gdp_report.pdf, page 1").
- Respect `prefers-reduced-motion` and `prefers-color-scheme`.
- Performance: first load under ~200 KB JS (gzip) excluding the PDF viewer, which loads only when first opened.

### Implementation choices
- Theme: follows the system by default; users can switch light/dark and pick one of 4 accents (indigo default, teal, violet, blue), all checked for WCAG AA in both modes.
- **Tailwind CSS** with the tokens above as CSS variables (one place to change the theme).
- **Radix UI** primitives (dialog, popover, select, tooltip) for accessible behavior, styled ourselves (the shadcn/ui approach).
- **Lucide** icons (thin, consistent line icons).
