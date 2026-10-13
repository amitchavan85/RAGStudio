# Production RAG Pipeline

A RAG pipeline where the user chooses the **chunking strategy**, **vector database** and **LLM** (for cost saving). Defaults live in `.env`; the CLI, the Python API and the FastAPI service accept per-request overrides for all three.

New to RAG? Read [LEARN.md](LEARN.md): every concept used here (chunking, embeddings, vector stores, hybrid retrieval, grounding, evaluation) explained in plain language, with this project's measured results and the lessons learned from testing on real data.

## Status

| Layer | Status |
|---|---|
| PDF ingestion | Done |
| Chunking (8 strategies) | Done |
| Embeddings (local + OpenAI) | Done (OpenAI key verified live) |
| Vector stores (Chroma, FAISS, Qdrant, pgvector) | Done (all four verified; Qdrant and pgvector against live servers) |
| Pipeline orchestration (ingest/query/list/delete, collection per model) | Done (Roadmap P1) |
| Retrieval (hybrid vector + BM25, filters, optional reranking) | Done (Roadmap P2) |
| LLM answers: OpenAI, Anthropic, Ollama; model chosen per question; citations | Done (P3). Cost-based selection: prepared, not active |
| FastAPI service (named indexes, per-request options, background ingest, Docker) | Done (P4) |
| Evaluation (30-question GDP set, retrieval + answer metrics, LLM judge, config comparison) | Done (P5) |
| Logging with request IDs; scale benchmark to 50k chunks | Done |
| Production hardening: two-column PDFs, headers, OCR; API keys + rate limits; CI (P6) | Done |
| Cost features (budgets, auto-router, cache), FAISS update speed, pgvector bulk load | Remaining |
| Web UI "RAG Studio" (React): ask with clickable citations + PDF viewer, uploads, settings, themes | Done (release 1), see [UI.md](UI.md) |

## Project structure

```
RAG/
├── .env.example              # default settings; copy to .env
├── docker-compose.yml        # Qdrant + Postgres/pgvector (+ the API with --profile api)
├── azure-pipelines.yml       # CI on Azure DevOps (current repo); .github/workflows/ci.yml for GitHub
├── eval/                     # gdp_questions.jsonl, content_questions.jsonl (questions with known answers); results/ (git-ignored)
├── web/                      # web UI (React + TypeScript + Vite); built files served by the API at /
├── Dockerfile                # API image (CPU-only PyTorch)
├── requirements.txt
├── pyproject.toml            # makes `src` installable (pip install -e .)
├── pytest.ini                # test paths and import path
├── src/
│   ├── evaluation.py         # evaluation runner + metrics: python -m src.evaluation
│   ├── api/                  # FastAPI app: schemas, named indexes, jobs (python -m src.api)
│   ├── benchmark.py          # scale benchmark for the vector stores: python -m src.benchmark --n 50000
│   ├── generation.py         # answer prompt, citations, Answer
│   ├── llm/                  # providers (openai, anthropic, ollama) + model catalog with prices
│   ├── retrieval.py          # BM25 keyword index, rank fusion (RRF), cross-encoder reranker
│   ├── pipeline.py           # RAGPipeline: ingest / query / list / delete with .env defaults + overrides
│   ├── cli.py                # command line for the pipeline: python -m src.cli ingest|query|list|delete
│   ├── config.py             # Settings (pydantic-settings, reads .env)
│   ├── logging_config.py     # setup_logging(), request id, log_step() timer
│   ├── data/pdfs/            # 8 sample GDP reports
│   ├── ingestion/
│   │   └── loader.py         # PDF -> one Document per page (tables kept as [TABLE] blocks); unreadable PDFs are skipped with a warning
│   ├── chunking/             # 8 strategies behind BaseChunker + get_chunker() factory
│   ├── embeddings/           # EmbeddingProvider: sentence-transformers, OpenAI
│   └── vectorstores/         # VectorStore: Chroma, FAISS, Qdrant, pgvector
└── tests/                    # tests (235); data/pdfs = committed copy of the 8 GDP samples; helpers.py = fakes
```

Logs are written to `./logs/rag.log` (git-ignored).

All imports use `from src.…`. Run `pip install -e .` once so `src` is importable from anywhere.

## Chunking strategies

Pick one with `get_chunker(strategy, chunk_size, overlap, embedder=None, llm_fn=None)`.

| Strategy | Splits on | Notes |
|---|---|---|
| `fixed` | character count | cheapest; can cut words |
| `recursive` | paragraph, line, sentence, word | good default |
| `sentence` | sentence boundaries | `chunk_size` is a soft limit |
| `sliding_window` | words | `chunk_size` is words; step = size − overlap |
| `document` | headings/sections | prefixes each chunk with its heading |
| `token` | LLM tokens | uses tiktoken if installed, else an approximation |
| `semantic` | topic shifts, using embeddings of lines and sentences | needs an `embedder`; one embedding per line/sentence; keeps table rows on their own lines; tiny chunks are merged |
| `agentic` | LLM decides | needs an `llm_fn` (`get_llm_fn()`); slowest and most expensive. Output is checked: if any number from the page is missing it falls back to `recursive`; tiny chunks are merged |

## Embeddings

`get_embedder(provider, model)`:
- `sentence_transformers` (default): local and free, `all-MiniLM-L6-v2`.
- `openai`: `text-embedding-3-small` by default, with batching and retries.

## Vector stores

`get_vector_store(name, dimension, collection)`. All four use cosine similarity and return higher scores for better matches. Re-ingesting a chunk replaces it instead of duplicating it.

| Store | Setup |
|---|---|
| `chroma` (default) | embedded, persisted under `.rag_data/chroma` |
| `faiss` | local flat index; vectors (`.f32`) and records (`.jsonl`) are appended under `.rag_data/faiss`, so adds don't rewrite the whole collection. The format changed from `.index`/`.json`: re-ingest old FAISS data |
| `qdrant` | server at `QDRANT_URL` (`docker compose up -d`), or `:memory:` for tests |
| `pgvector` | Postgres with the `vector` extension (`docker compose up -d`). Uses `hnsw.ef_search=100` (pgvector's default 40 lost recall at 50k chunks) |

## Configuration

Copy `.env.example` to `.env`. Main settings: `API_KEYS`, `RATE_LIMIT_PER_MINUTE`, `RATE_LIMIT_ANSWERS_PER_MINUTE`, `OCR`, `INDEXES`, `ALLOW_CUSTOM_INDEX`, `ALLOWED_CHUNKERS`, `MAX_UPLOAD_MB`, `LLM_MODEL`, `ANTHROPIC_API_KEY`, `OLLAMA_URL`, `PDF_DIR`, `RETRIEVAL_MODE`, `RERANKER`, `LLM_MODEL`, `LOG_LEVEL`, `LOG_DIR`, `CHUNKING_STRATEGY`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `OPENAI_API_KEY`, `VECTOR_STORE`, `COLLECTION_NAME`, `QDRANT_URL`, `PG_DSN`.

## Quick start

```bash
pip install -r requirements.txt
pip install -e .                       # makes `src` importable from anywhere (debugger, IDE)
cp .env.example .env
python -m pytest                       # 235 tests
```

```python
from src.pipeline import RAGPipeline

rag = RAGPipeline()                                   # defaults from .env
rag.ingest()                                          # every PDF in PDF_DIR (or a file, list of files, folder)
hits = rag.query("China GDP growth 2023", k=5)       # hybrid search by default
hits = rag.query("27,360", filters={"source": "usa_gdp_report.pdf"}, mode="keyword")
ans = rag.answer("Compare 2023 GDP growth of India and the US", model="claude-haiku-5-5")
print(ans.text, ans.citations)                        # "... +8.2% [5] ... +2.5% [6]" + cited source/page
rag.list_documents()                                  # {"chn_gdp_report.pdf": 6, ...}
rag.delete_document("chn_gdp_report.pdf")

# Any setting can be overridden per pipeline (per request, once the API exists):
fast = RAGPipeline(vector_store="faiss", chunking_strategy="sentence", chunk_size=300)
```

## Pipeline (P1)

`src/pipeline.py` is the single entry point that joins loader, chunker, embedder and vector store.

- **Defaults and overrides:** `PipelineConfig.from_settings(**overrides)` starts from `.env` and replaces only the fields you pass (`None` means "keep the default"). Invalid values (unknown store or strategy, overlap >= size, unknown option names) fail immediately with a clear `ValueError`.
- **One collection per embedding model:** the collection name is `<COLLECTION_NAME>_<model>_<dimension>`, e.g. `rag_docs_all_minilm_l6_v2_384` or `rag_docs_text_embedding_3_small_1536`. Vectors from different models are not comparable, so they can never end up in the same collection. Switching `EMBEDDING_PROVIDER` therefore starts with an empty collection: re-ingest your documents.
- **Documents are identified by file name** (`metadata["source"]`). Re-ingesting a file first deletes all its old chunks, so changing the chunker or chunk size leaves nothing stale. Old chunks are deleted only after the new ones were chunked and embedded, so a failed re-ingest keeps the previous version. Two different files with the same name in one ingest are rejected.
- **Every store supports** `delete_source(name)` and `list_sources()`. Qdrant and pgvector index the `source` field so this stays fast on large collections.
- `query()` returns the most relevant chunks; turning them into an answer is roadmap P3.
- Existing data from before P1 (collections named `rag_docs` or `doc_<file>`) is not picked up: re-ingest.

## Retrieval (P2)

`rag.query(question, k, filters=None, mode=None, reranker=None)` supports three modes (default from `RETRIEVAL_MODE`):

| Mode | How | Good at |
|---|---|---|
| `vector` | embedding similarity in the vector store | paraphrased questions ("which economy relies on iron ore?") |
| `keyword` | BM25 over all chunk texts | exact terms: figures ("27,360"), codes ("E27"), names |
| `hybrid` (default) | both, merged with Reciprocal Rank Fusion (ranks only, so no score tuning) | both kinds of question |

- **Filters** restrict every mode to matching metadata: `{"source": "a.pdf"}`, `{"source": ["a.pdf", "b.pdf"], "page": 2}`. Each store filters natively (Chroma `where`, FAISS ID selector, Qdrant payload filter, pgvector `WHERE` with iterative index scan so filtered searches still return k rows).
- **Reranking** (`reranker="cross_encoder"` or `RERANKER=cross_encoder`) re-scores the top candidates with a free local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~90 MB download on first use). Off by default (see below).
- Each result's `scores` shows every stage (`vector`, `keyword`, `rrf`, `rerank`), so you can see why it ranked where it did. The CLI prints them.
- **Keyword index:** built in memory from the texts already in the vector store (the store stays the single source of truth), cached per collection, and rebuilt after this process ingests/deletes or when the store's chunk count changes (another process wrote to it). Building it reads every chunk once (about 2-3 s at 50k chunks). Limitation: if another process replaces a document with exactly the same number of chunks, this process keeps the old keyword index until its next own ingest/delete or a restart.
- **Shared stores:** all pipelines in one process share one store object per collection (thread-safe for FAISS), so concurrent requests never see stale data or overwrite each other, and server stores reuse one connection. FAISS and Chroma are embedded, single-process stores: for several processes (e.g. multiple API workers) use Qdrant or pgvector.

Measured on the 8 GDP samples (44 chunks, local MiniLM embeddings, 6 exact-term + 6 paraphrased questions, answer document in top 1 / top 3):

| Mode | Exact top 1 | Exact top 3 | Paraphrased top 1 | Paraphrased top 3 |
|---|---|---|---|---|
| vector | 4/6 | 5/6 (misses "27,360") | 6/6 | 6/6 |
| keyword | 6/6 | 6/6 | 5/6 | 6/6 |
| **hybrid** | 4/6 | **6/6** | **6/6** | **6/6** |
| hybrid + rerank | 4/6 | 5/6 (misses "3,170") | 6/6 | 6/6 |

Hybrid is the only mode that puts every answer in the top 3 (what the LLM will read), so it is the default. The reranker did not help on this corpus and hurt bare-number lookups, so it is off by default; re-check it on your own documents once evaluation (P5) exists. This is a small sample; P5 will measure on real questions.

## Answers (P3)

`rag.answer(question, k=8, filters=None, mode=None, reranker=None, model=None)` retrieves the k best chunks and asks an LLM to answer **only from them**, citing each claim as `[n]`. It returns an `Answer` with `text`, `citations` (source file, page, chunk id per cited number), all `sources`, `model`, `provider` and token counts. If the documents do not contain the answer, the reply is "I could not find the answer in the provided documents." (`ans.found` is then `False`); if nothing is retrieved at all, the LLM is not called.

**Choosing the model (mixed providers).** Pass `model=` per question, or set the default with `LLM_MODEL` in `.env`. The model name decides the provider:

| Provider | Example models | Needs |
|---|---|---|
| OpenAI | `gpt-5-nano`, `gpt-4o-mini` (default), `gpt-5-mini`, `gpt-5.4`, `gpt-5.5` | `OPENAI_API_KEY` |
| Anthropic | `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`, `claude-fable-5-1` | `ANTHROPIC_API_KEY` |
| Ollama (local, free) | `llama3.2:3b`, `qwen2.5:7b`, or any pulled model as `ollama:<name>` | Ollama running; `ollama pull <model>` first |

`python -m src.cli models` lists the catalog (`src/llm/catalog.py`) with each model's price per 1M tokens and tier; any other model works as `<provider>:<model>`. Unknown names fail before anything is retrieved.

- **Prices are recorded but not used yet.** The catalog stores price and tier (local / economy / standard / premium) per model, checked 2026-10-13, and every answer reports its token counts, so cost-based selection (budget per question, tier choice, auto-router, cost tracking) can be added without changing the providers. Update prices in `catalog.py` when providers change them.
- **Grounding:** the system prompt says to use only the sources, cite them, keep numbers exactly, and say when the answer is missing. Source text is wrapped as data and the model is told to ignore instructions inside it (a PDF could contain prompt-injection text).
- **Claude specifics:** Opus 5.5, Sonnet 5.5 and Fable 5.1 requests enable Anthropic's server-side refusal fallback (`fallbacks: "default"`), so a policy decline is re-run on Anthropic's recommended fallback model inside the same call; `ans.model` shows which model actually answered. A refusal that still happens raises `LLMRefusal`.
- **Why k=8 for answers** (5 for `query`): comparison questions need facts from several documents. Testing "compare India and US 2023 growth" with k=5 left out the US chunk (it ranked 6th) and the model said the US figure was missing; with k=8 it answered both correctly with citations. Costs about 600 more input tokens per question.
- The agentic chunker uses the same layer (`LLM_MODEL`), so it can run on any provider too.

```bash
python -m src.cli ask "Compare the 2023 GDP growth of India and the United States."
python -m src.cli ask "What does the report say about inflation?" --model claude-haiku-5-5 --source content.pdf
python -m src.cli ask "..." --model ollama:llama3.2:3b --show-sources
```

## Web UI

```bash
cd web && npm install && npm run build && cd ..   # once (needs Node.js 20+)
python -m src.api                                  # open http://127.0.0.1:8000
```

Ask questions with clickable citations that open the PDF at the cited page with the passage highlighted, upload and manage documents, pick the model by cost tier, and switch theme/accent. Development mode, code map and design guide: [UI.md](UI.md).

## HTTP API (P4)

```bash
python -m src.api                                    # http://127.0.0.1:8000, interactive docs at /docs
docker compose --profile api up -d --build api       # or in Docker, next to Qdrant and Postgres
```

| Method & path | Purpose |
|---|---|
| `POST /v1/documents` | Upload PDFs (multipart `files`) into an index. Returns a job (202); `?wait=true` waits and returns the finished job. Re-uploading a file name replaces that document. |
| `GET /v1/jobs/{job_id}` | Ingest job status: `queued`, `running`, `succeeded` (with per-file results) or `failed` (with the error) |
| `GET /v1/documents?index=` / `DELETE /v1/documents/{source}?index=` | List / delete documents |
| `POST /v1/search` | Ranked chunks with per-stage scores |
| `POST /v1/answer` | Grounded answer, citations (renumbered 1..n, with snippets), token usage |
| `GET /v1/indexes`, `/v1/models`, `/v1/options` | What a client can choose: indexes with document counts, models with availability and price, chunkers, modes, defaults, limits |
| `GET /v1/documents/{source}/file?index=` | The original PDF (kept for uploads through the API), used by the UI to open citations |
| `GET /health`, `GET /ready` | Liveness; readiness (every index's store reachable) |

Every option is optional and falls back to `.env`. A request picks:

```jsonc
// POST /v1/answer
{ "question": "Compare 2023 GDP growth of India and the US",
  "index": "default",                                          // named index (store + embedding model)
  "retrieval": { "mode": "hybrid", "k": 8, "reranker": "none", "filters": { "source": ["a.pdf"] } },
  "llm": { "model": "claude-haiku-5-5", "max_tokens": 2048 },
  "include_sources": false }
// multipart form field `options` on POST /v1/documents:
{ "index": "default", "chunking": { "strategy": "semantic", "chunk_size": 500, "overlap": 50 } }
```

- **Named indexes.** The vector store and embedding model decide the collection, so ingest and query must use the same pair. Users therefore pick an index by name; the operator defines them in `.env`, e.g. `INDEXES={"production": {"vector_store": "qdrant", "embedding_provider": "openai"}, "local": {"vector_store": "faiss", "embedding_provider": "sentence_transformers"}}` (without `INDEXES` there is one index, `default`, from `VECTOR_STORE` / `EMBEDDING_PROVIDER`). `ALLOW_CUSTOM_INDEX=true` additionally accepts a free-form `{"vector_store": ..., "embedding_provider": ...}` (off by default: a mismatch silently searches an empty collection).
- **Guardrails:** `ALLOWED_CHUNKERS` (e.g. exclude `agentic`, which calls an LLM per page), `MAX_UPLOAD_MB`, `MAX_K`; only real PDFs (`%PDF-` header) are accepted; unknown fields in a request are rejected (typos don't silently fall back to defaults).
- **Errors** always look like `{"error": {"code", "message", "details"}, "request_id"}`: 400 invalid option (unknown model, bad filter, k too large), 403 custom index disabled, 404 index/document/job not found, 413 file too large, 415 not a PDF, 422 invalid request or model refusal, 502 model provider failed, 503 vector store unreachable, 500 otherwise.
- **Request IDs:** send `X-Request-ID` (or get a generated one); it is echoed in the header and body and appears on every log line of that request, including the background ingest job.
- **Concurrency:** endpoints run in a thread pool; ingest jobs run two at a time in the background. Jobs live in memory (lost on restart). Run a single worker (as `python -m src.api` does): FAISS and Chroma are single-process stores; for several workers use Qdrant or pgvector indexes and an external job queue.
- **Docker:** the `api` service reads `.env`, but the compose file pins container paths (`DATA_DIR=/data`, logs in `/data/logs`) and reaches Qdrant/Postgres by service name and Ollama on the host. The image uses CPU-only PyTorch and downloads embedding models on first use into the `/data` volume.
- **Auth and rate limits:** see Production hardening (P6) below. **Reserved for later:** `"stream": true` (token streaming).

## Evaluation (P5)

`python -m src.evaluation` runs a set of questions with known answers through any combination of chunker x store x retrieval mode x LLM, prints a comparison table, and saves every answer and metric to `eval/results/<timestamp>.json`.

```bash
python -m src.evaluation --retrieval-only --chunkers recursive semantic --modes vector hybrid   # no LLM calls
python -m src.evaluation --models gpt-4o-mini claude-haiku-5-5 --judge gpt-4o-mini              # answers + faithfulness
python -m src.evaluation --dataset eval/my_questions.jsonl --docs path/a.pdf path/b.pdf          # your own documents
```

**Dataset** (`eval/gdp_questions.jsonl`, 30 questions over the 8 GDP reports): table figures, simple lookups, outlook text, paraphrased questions, rankings, two-document comparisons, and 4 questions the documents cannot answer. Each line lists the question, where the evidence is (`[[file, page], ...alternatives]`, one group per document a comparison needs) and the facts the answer must contain (`"27,360|27360"` = alternatives). A test checks that every expected fact really appears on its evidence page, so typos in the ground truth are caught.

**Metrics:** `hit@1/3/5/k` = all evidence pages retrieved within the top n; `mrr` = how high the evidence ranks; `fact_acc` = answer contains every expected fact; `cited_ok` = citations point only to evidence documents; `false_nf` = wrongly said "not found"; `abstain_acc` = correctly said "not found" for unanswerable questions; `faithful` = the `--judge` model found no unsupported claim. Ingestion runs once per chunker/store in a temporary folder (your indexes are untouched; evaluation collections on Qdrant/pgvector are removed afterwards).

Results on the GDP set (local MiniLM embeddings, FAISS, k=8), 2026-10-13:

| Retrieval (no LLM) | hit@1 | hit@3 | hit@5 | mrr |
|---|---|---|---|---|
| recursive + vector | 0.77 | 0.81 | 0.96 | 0.84 |
| recursive + keyword | 0.62 | 0.89 | 0.89 | 0.75 |
| **recursive + hybrid** | 0.69 | **0.96** | **1.00** | 0.82 |
| sentence + hybrid | 0.69 | 0.92 | 0.96 | 0.83 |
| document + hybrid | 0.73 | 0.92 | 0.92 | 0.83 |
| semantic + hybrid | 0.69 | 0.89 | 0.96 | 0.80 |

| Answers (recursive + hybrid) | fact_acc | cited_ok | false_nf | abstain_acc | faithful | latency | out tokens |
|---|---|---|---|---|---|---|---|
| **gpt-4o-mini** | **1.00** | **1.00** | 0.00 | 1.00 | 1.00 | 1.4 s | 21 |
| gpt-5-mini | 0.96 | 0.96 | 0.04 | 1.00 | 0.96 | 3.4 s | 288 |

What this shows: hybrid retrieval puts the evidence in the top 5 for every question; semantic chunking does not beat recursive on short, well-structured reports; on this corpus `gpt-4o-mini` answers as accurately as `gpt-5-mini` at a fraction of the latency and output tokens (gpt-5-mini spends tokens on reasoning and once wrongly said "not found"). Caveats: 30 questions on synthetic reports is a small sample, so a 1-question difference is 3-4 points; the judge is itself a small model and made one false "unsupported" call (a table it misread), so check flagged rows by hand or use a stronger judge. Build a question set for your own documents before choosing settings for them.

## Production hardening (P6)

### Ingestion quality
`src/ingestion/layout.py` runs on every PDF:

- **Two-column pages** are read in order: it finds a vertical gutter that prose lines never cross and reads the left column, then the right; titles and full-width text stay in page order. Tables with a gap in the middle are not mistaken for columns (both sides must hold prose).
- **Running headers/footers** ("GLOBAL ECONOMIC PROSPECTS | JUNE 2026 CHAPTER 1 19") that repeat at the top/bottom of at least 30% of pages (ignoring page numbers) are removed. Lines without words (table rows) never are.
- **Drop capitals** ("G rowth") are re-joined.
- **OCR:** pages without a text layer (scans) are rendered and read with RapidOCR (`rapidocr-onnxruntime`, pip only, no system install). OCR pages carry `metadata["ocr"] = True`. `OCR=off` disables it.

Measured on `content.pdf` (45-page two-column World Bank report, 10 questions in `eval/content_questions.jsonl`, local embeddings):

| Retrieval | top 1 | top 3 | top 5 | MRR |
|---|---|---|---|---|
| hybrid, old extraction | 0.4 | 0.7 | 0.8 | 0.56 |
| **hybrid, column-aware** | **0.7** | **0.9** | **1.0** | **0.82** |

The 8 single-column GDP samples extract byte-for-byte as before (a test checks that no page of them is treated as two columns).

### API keys and rate limits
```bash
# .env
API_KEYS={"webapp": "a-long-random-key-1", "ci": "a-long-random-key-2"}   # name -> key, keys >= 16 chars
RATE_LIMIT_PER_MINUTE=120            # all /v1 requests, per client (0 = unlimited)
RATE_LIMIT_ANSWERS_PER_MINUTE=30     # POST /v1/answer (paid LLM calls), per client

curl.exe http://127.0.0.1:8000/v1/indexes -H "X-API-Key: a-long-random-key-1"
```
- Every `/v1` endpoint needs `X-API-Key`; `/health`, `/ready` and `/docs` stay open. In `/docs`, click **Authorize** and paste a key.
- Missing or wrong key: 401 (`missing_api_key` / `invalid_api_key`). Over the limit: 429 `rate_limited` with a `Retry-After` header; each client has its own budget.
- Without `API_KEYS` the API is open (development) and logs a warning at startup; limits then apply per IP address. A malformed `API_KEYS` stops the API at startup.
- Keys are compared in constant time and never logged; logs show the client **name** (`GET /v1/indexes -> 200 client=webapp ms=12`).
- Limits are kept in memory per API process (fine for one worker; several workers would need a shared store such as Redis).

### CI
Two equivalent pipelines: install dependencies (CPU-only PyTorch), start Qdrant and pgvector as service containers, run the whole test suite, typecheck/test/build the web UI, and build the Docker image.

- **Azure DevOps** (this project currently lives in the `D:/Repos` repository whose remote is Azure DevOps): Pipelines > New pipeline > Azure Repos Git > *Existing Azure Pipelines YAML file* > `/AI-Repo/RAG/azure-pipelines.yml`. It runs only when files under `AI-Repo/RAG` change.
- **GitHub Actions** (`.github/workflows/ci.yml`): used automatically if this folder becomes its own GitHub repository.
- Paid live LLM tests run only if you add `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` as secret pipeline variables (GitHub: repository secrets); otherwise they skip. A local run that mimics CI (no keys, services up): 226 passed, 9 skipped.
- Tests use a committed copy of the GDP samples in `tests/data/pdfs`, so they never depend on your data folder.

## Logging

Plain-text logs go to the console and to a rotating file (`LOG_DIR/rag.log`, 5 MB x 3). A relative `LOG_DIR` (default `./logs`) is resolved from the project root, and `.env` is also read from the project root, so the same settings and log folder apply wherever you run from. Logging is only active after `setup_logging()` is called: `python -m src.cli` and `python -m src.benchmark` do this; in your own scripts, call it once at startup. Test runs write their logs to `logs/pytest.log` (set in `pytest.ini`). Call `setup_logging()` once at startup and `new_request_id()` once per ingest or query request; every line then carries `[req=<id>]`.

Each pipeline step writes one timed line through `log_step()`: `load_pdf` (per file), `load` (directory), `chunk`, `embed`, `store.add`, `store.reset`, `retrieve`. Example:

```
2026-10-10 22:16:13 INFO [req=dd3699f3] src.chunking.base: chunk done strategy=RecursiveChunker chunk_size=500 overlap=50 docs=16 chunks=44 len_min_avg_max=136/390/517 ms=0
2026-10-10 22:16:27 INFO [req=dd3699f3] src.vectorstores.base: retrieve done store=faiss collection=logdemo k=3 hits=3 top_score=0.753 ms=0
```

Rules: log counts, ids, scores and durations only; never document text, questions or secrets at INFO. Failures log ERROR with the traceback once and re-raise; fallbacks and retries log WARNING. Set `LOG_LEVEL=DEBUG` to also see step-start lines. Set `LOG_DIR=` (empty) to disable the file.

## Try it on your own PDF

```bash
python -m src.cli ingest path/to/doc.pdf                 # or several files / a folder; no path = PDF_DIR
python -m src.cli query "What is the main conclusion?" -k 3
python -m src.cli query "27,360" --mode keyword --source usa_gdp_report.pdf --page 1   # modes, filters
python -m src.cli query "What is the outlook?" --rerank                                  # cross-encoder reranking
python -m src.cli list                                   # documents and chunk counts
python -m src.cli delete doc.pdf
python -m src.cli --store faiss --chunker semantic ingest path/to/doc.pdf   # overrides go before the command
```

Use the same `--store` / `--embedding-provider` for ingest and query: together they pick the collection. Re-running `ingest` on a file replaces its previous chunks.

Check that the whole pipeline works on any single PDF (load, all 8 chunkers, no numbers lost, size limits, retrieval on Chroma/FAISS/Qdrant, re-ingest):

```powershell
$env:RAG_TEST_PDF = "src/data/pdfs/your.pdf"; python -m pytest tests/test_single_document.py -v
```

Without `RAG_TEST_PDF` it runs on a bundled sample. A 45-page report takes about 4 minutes (the semantic chunker embeds every line).

## Scale benchmark

`python -m src.benchmark --n 50000` grows the corpus to N chunks (real chunks embedded once, then varied with small noise) and measures each store: batched ingest, query latency, recall@10 against exact search, and re-adding 100 existing chunks. Live stores need `docker compose up -d`; benchmark collections are deleted afterwards. With `EMBEDDING_PROVIDER=openai` it embeds the real chunks through the API (well under a cent for ~500 chunks).

Measured on this machine (1536-dim OpenAI embeddings, batches of 1000), **before** the FAISS and pgvector fixes below:

| Store | Ingest 10k | Ingest 50k | Query p50 at 50k | Recall@10 at 50k | Update 100 at 50k |
|---|---|---|---|---|---|
| Chroma | 15.5 s | 132.7 s | 32 ms | 0.994 | 1.0 s |
| FAISS | 2.9 s | 114.8 s | 43 ms | 1.0 | 3.7 s |
| Qdrant | 18.9 s | 221.2 s | 81 ms | 1.0 | 0.7 s |
| pgvector | 41.8 s | 472.1 s | 61 ms | **0.812** | 0.6 s |

After the fixes (50k chunks, re-measured):

| Store | Ingest 50k | Query p50 | Recall@10 | Update 100 |
|---|---|---|---|---|
| FAISS | **16.5 s** (was 114.8) | 39 ms | 1.0 | 4.6 s (was 3.7, not fixed) |
| pgvector | 329.8 s (was 472.1) | 61 ms | **0.99** (was 0.812) | 0.4 s |

Findings and fixes:
- **FAISS ingest grew with the square of the size** (5× data, 40× time): every add rewrote the whole index (~300 MB at 50k) and JSON file. Storage is now append-only: 7× faster, close to linear. **Still open:** updating existing chunks (re-ingesting an edited document) rewrites the whole vector file, about 4.6 s at 50k.
- **pgvector silently lost recall** (0.81 at 50k) with the default HNSW search width. Now `ef_search=100` (recall 0.99) and inserts are sent in pages of 1000 (1.4× faster). **Still open:** pgvector is the slowest to ingest because it updates the HNSW index on every insert.
- Chroma and Qdrant behaved as expected for HNSW indexes (ingest slows somewhat with size, recall stays ≥ 0.99). Qdrant ingest over REST is the next-slowest; gRPC would likely help.

## Testing

Run from the repo root (after `pip install -e .`):

```bash
python -m pytest                          # all tests; last line shows e.g. "235 passed"
python -m pytest -v                       # one line per test (PASSED/FAILED)
python -m pytest tests/test_logging.py    # one file
python -m pytest -k "chroma"              # tests whose name matches
python -m pytest -x                       # stop at first failure
python -m pytest -s                       # show print output
python -m pytest tests/test_logging.py -v -o log_cli=true -o log_cli_level=INFO   # live logs
```

In VS Code: Ctrl+Shift+P, "Python: Configure Tests" (pytest, folder `tests`), then use the Testing (flask) panel to run or debug tests. Select the same interpreter you ran `pip install -e .` with. If you see `No module named 'src'`, the editable install is missing for that interpreter.

Optional coverage: `pip install pytest-cov` then `python -m pytest --cov=src --cov-report=term-missing`.

`pytest.ini` sets `testpaths` and puts the repo root and `tests/` on the import path; `pyproject.toml` makes `src` installable.

What the tests cover (live tests skip automatically when their key, server or model is missing):

| File | Covers |
|---|---|
| `test_components.py`, `test_pipeline_foundation.py` | loader, all 8 chunkers, config, factories, stores (Chroma, FAISS, in-memory Qdrant) |
| `test_pipeline.py` | P1: overrides, collection per model, re-ingest without stale chunks, delete, failure safety |
| `test_retrieval.py`, `test_retrieval_real.py` | P2: BM25, RRF, filters on every store, keyword-index invalidation; real-model check that hybrid finds "27,360" where vector search misses it |
| `test_generation.py`, `test_generation_live.py` | P3: model catalog, prompt, citations, providers (mocked); live answers per provider |
| `test_api.py` | P4: every endpoint and error code, jobs, named/custom indexes, uploads |
| `test_evaluation.py` | P5: the answer key itself (every fact is on its evidence page), metrics, runner |
| `test_logging.py` | log file, request IDs, no document text in logs |
| `test_layout.py` | P6: two-column reading order, header/footer removal, drop capitals, OCR of a scanned page, GDP pages not split |
| `test_api_auth.py` | P6: API keys (401), rate limits (429 + Retry-After, per client, answers bucket), keys never logged |
| `web/src/lib/logic.test.ts` (Vitest, `npm test`) | UI logic: citation parsing, model per tier, cost hints, error messages, passage matching for PDF highlights, WCAG contrast of every theme |
| `test_live_servers.py` | real Qdrant and pgvector (`docker compose up -d`): round trip, upsert, filters, delete, ranking equals exact cosine |
| `test_openai_live.py`, `test_agentic_live.py`, `test_semantic_real.py`, `test_single_document.py` | real OpenAI key, agentic chunker with a real LLM, semantic chunker with the real model, any single PDF (`RAG_TEST_PDF`) |

Most tests use a fake embedder and a fake LLM, so they need no downloads or keys. Remaining untested items are listed under TODO below.

## TODO: still untested

These parts are written but have never been run. Verify each before relying on it.

- [x] **pgvector store** (`src/vectorstores/pgvector_store.py`): verified against the docker-compose Postgres by `tests/test_live_servers.py`.
- [x] **Qdrant against a real server**: verified against the docker-compose container by `tests/test_live_servers.py`.
- [x] **OpenAI embedder** (`src/embeddings/openai_provider.py`): verified with a live call (`tests/test_openai_live.py`, skipped when no key is set). Batching and retry behaviour under load are still unchecked.
- [x] **Agentic chunker with a real LLM**: verified with `gpt-4o-mini` (`tests/test_agentic_live.py`). On the 16 sample pages the first prompt let the model paraphrase and drop table rows (caught by the number check and sent to the recursive fallback); after requiring verbatim copy, 16/16 pages were faithful, 48 chunks averaging 328 characters. Cost was about $0.004 and 3.5 s per page. Not yet checked: long pages, other models, retry behaviour under rate limits.
- [ ] **Token chunker with `tiktoken`**: tests use the word-count fallback. Install `tiktoken` and confirm exact token counts.
- [x] **Semantic chunker with the real embedding model**: verified with `all-MiniLM-L6-v2` (`tests/test_semantic_real.py`). Found and fixed two problems: it split on sentences only, so line-based PDF text produced about 2 cuts per page (the 500-char fallback did the rest) and table newlines were flattened; it now splits on lines and sentences, keeps the original whitespace and merges tiny chunks. Result at the default `breakpoint_percentile=85`: 59 chunks, avg 269 chars, none under 100, no numbers lost. Still to tune: the percentile on a larger, more varied corpus. Note PDF hard line-wraps can still put a cut mid-sentence.
- [ ] **Large-scale behaviour (mostly done)**: benchmarked to 50k chunks (see Scale benchmark). FAISS ingest and pgvector recall are fixed, re-measured, and the full test suite passes with the new FAISS storage. Open: FAISS updates rewrite the whole vector file (4.6 s at 50k); pgvector bulk ingest is slow (151 chunks/s); RAM use not measured.
- [x] **Tests depend on the data folder**: tests now use a committed copy in `tests/data/pdfs`.

- [x] **Multi-column PDFs**: fixed by column-aware extraction (P6); retrieval on `content.pdf` went from hit@1 0.4 to 0.7.
- [ ] **Anthropic and Ollama live**: implemented and unit-tested with mocks; add `ANTHROPIC_API_KEY` / run `ollama pull llama3.2:3b` and the live tests in `test_generation_live.py` run.
- [ ] **Evaluation on your own documents**: started with 10 questions for `content.pdf` (`eval/content_questions.jsonl`, retrieval checked); extend it and run answers before choosing settings for that report.
- [ ] **Judge reliability**: the gpt-4o-mini judge wrongly flagged the same correct answer ("Germany ranked third") in 2 of 103 judged answers; use a stronger judge for decisions.
- [ ] **Tiny chunks from the `document` chunker**: one-line sections become very small chunks (21 under 50 characters on a 45-page report). Consider merging them like the semantic and agentic chunkers do.

## Roadmap: remaining steps (in priority order)

Steps 1–3 give a working question-and-answer pipeline; step 4 makes it the selectable, cost-saving service; step 5 shows whether it is actually good. Add `log_step` calls to each new layer as it is built.

### P1. Pipeline orchestration (done)
- [x] One `ingest(files, chunker, store)` and `query(question, store, llm)` entry point that applies `.env` defaults and per-request overrides.
- [x] Collection naming per embedding model, so 384-dim and 1536-dim vectors can never be mixed in one collection.
- [x] Document update and delete: re-ingesting an edited PDF removes its old chunks first.

### P2. Retrieval (done)
- [x] Hybrid search: vector search plus BM25 keyword search (catches exact terms like "E27" or "2023" that embeddings miss).
- [x] Optional reranking of the top results for precision.
- [x] Metadata filters (source file, page).

### P3. Generation (LLM layer) (done, cost features prepared)
- [x] Mixed providers: OpenAI, Anthropic, Ollama; the user picks the model per question (`model=` / `--model`) or via `LLM_MODEL`.
- [x] Answer prompt: retrieved chunks as context, answer with page citations, say "not in the documents" when it isn't.
- [ ] Cost saving (deferred; catalog prices + token counts are in place): budget per question, tier selection, auto-router (easy questions to a cheap model), response cache, cost tracking per request.
- [ ] Live check of Anthropic (needs `ANTHROPIC_API_KEY`) and Ollama (needs `ollama pull llama3.2:3b`); their tests skip until then.

### P4. API service (done)
- [x] FastAPI endpoints: upload/ingest, query, list and delete documents, health.
- [x] Per-request overrides for chunker, vector store and LLM (the original requirement).
- [x] Dockerfile for the app, alongside Qdrant and Postgres in `docker-compose.yml`.

### P5. Evaluation (done)
- [x] A small set of questions with known answers (30, GDP samples). Next: a question set for your own documents (e.g. content.pdf).
- [x] Retrieval metrics (is the right page in the top k?) and answer metrics (correct, faithful to sources, cited).
- [x] Use it to compare chunkers, stores and models with numbers.

### P6. Production hardening (mostly done)
- [x] Ingestion quality: column-aware extraction, running header/footer removal, drop capitals, OCR for scanned pages (RapidOCR). Other file formats: not needed yet.
- [x] API auth (keys) and rate limits; CI that runs the test suite (Azure Pipelines + GitHub Actions).
- [ ] Open items: FAISS update speed, pgvector bulk load, `tiktoken`, the deferred cost features (P3), Anthropic/Ollama live checks. (Test data moved to `tests/data/`: done.)
