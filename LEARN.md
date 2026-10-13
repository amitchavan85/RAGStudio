# Learning RAG with this project

This guide explains every idea used in this repository: what it is, why it exists, what it costs, and what we learned when we tested it on real documents. Read it top to bottom once; later, jump to a section. For commands and settings, see [README.md](README.md).

**Contents:** [Big picture](#0-the-big-picture) · [Ingestion](#1-ingestion-pdf--text) · [Chunking](#2-chunking) · [Embeddings](#3-embeddings) · [Vector stores](#4-vector-stores) · [Pipeline](#5-the-pipeline-one-entry-point-p1) · [Retrieval](#6-retrieval-finding-the-right-chunks-p2) · [Generation](#7-generation-turning-chunks-into-an-answer-p3) · [API](#8-serving-it-the-http-api-p4) · [Evaluation](#9-evaluation-measuring-quality-p5) · [CI](#9b-continuous-integration-ci) · [Logging](#10-observability-logging-and-request-ids) · [Lessons learned](#11-lessons-learned-from-testing-on-real-data) · [Status](#12-project-status-and-whats-next) · [Glossary](#13-glossary)

---

## 0. The big picture

RAG (Retrieval-Augmented Generation) means: **before an LLM answers, fetch the few passages of your own documents that are relevant to the question, and make the model answer from those passages.** The model then works like an open-book exam instead of answering from memory, so it can use private or recent information, cite where facts come from, and say "I don't know" when the documents don't cover the question.

The work splits into two phases:

```
INGEST (once per document)                         QUERY (every question)
PDF ──► text per page ──► chunks ──► vectors ──►   question ──► vector of question
        (loader)          (chunker)  (embedder)            │
                                        │                  ▼
                                        └──► vector store ◄── search (vector + keyword)
                                                           │
                                                           ▼
                                             top chunks ──► LLM ──► answer with citations [1][2]
```

Where each step lives in the code:

| Step | Code | Section |
|---|---|---|
| Read PDFs | `src/ingestion/loader.py` | 1 |
| Cut text into chunks | `src/chunking/` | 2 |
| Turn text into vectors | `src/embeddings/` | 3 |
| Store and search vectors | `src/vectorstores/` | 4 |
| Tie it all together | `src/pipeline.py` | 5 |
| Keyword search, fusion, reranking | `src/retrieval.py` | 6 |
| Prompt, citations, LLM providers | `src/generation.py`, `src/llm/` | 7 |
| HTTP API | `src/api/` | 8 |
| Quality measurement | `src/evaluation.py`, `eval/` | 9 |

---

## 1. Ingestion (PDF → text)

**What:** turn files into clean text with metadata. `loader.py` reads each PDF **page by page** and stores the file name and page number with the text, so every answer can later point to "usa_gdp_report.pdf, page 1". Tables are extracted separately and appended as `[TABLE]` blocks, because plain text extraction often runs table columns together.

**Why it matters most:** every later step works on this text. If a number is garbled here, no chunker, embedding or LLM can fix it. *Garbage in, garbage out* is literal in RAG.

**What we learned:**
- **Two-column reports break simple extraction.** On a real 45-page World Bank report, the extractor read straight across both columns, mixing sentences from left and right: *"Growth in the euro area is expected to recover to t ariffs and the continued strength..."*. **Fix (P6): column-aware extraction.** For each page we look for a vertical *gutter*, an x-position that the text lines never cross, with prose on both sides. If there is one, the page is read left column first, then right column; titles and full-width text stay in page order. Tables with a gap in the middle are not mistaken for columns, because both sides must contain prose (several words per line).
- **Measure the fix, don't just eyeball it.** With 10 questions about that report, the right page was ranked first for 4 of 10 questions with the old extraction and 7 of 10 with the new one; in the top 5, 8 of 10 became 10 of 10. Better text in, better search out.
- **Running headers and footers** ("GLOBAL ECONOMIC PROSPECTS | JUNE 2026 CHAPTER 1 19") repeat on every page and end up in many chunks, adding noise. Lines at the top or bottom of a page that repeat on at least 30% of pages (ignoring page numbers, which change) are removed. Why 30% and not 50%? Books alternate headers on left and right pages, so each version appears on only half the pages.
- **Drop capitals** (a large first letter, "G rowth") are re-joined, but only at the start of a line and never for "A" or "I", which are real words.
- **Scanned PDFs have no text layer.** They are now read with **OCR** (optical character recognition: turning an image of text into text). We render the page as an image and read it with RapidOCR, a pip-installable engine; in the test, the US report "scanned" as an image came back with "Washington", "27,360" and "Services" intact. OCR is slower and can misread characters, so OCR pages are marked in their metadata.
- **One bad file must not stop a batch.** A 0-byte PDF crashed the whole folder load; now unreadable files are skipped with a warning and listed as `skipped`.

---

## 2. Chunking

**What:** cut each page into small pieces ("chunks") that are searched and shown to the LLM.

**Why not whole documents?** An embedding squeezes a text into one vector. A whole report becomes a blurry average of every topic in it and matches everything weakly. A small chunk about one idea matches precisely. Small chunks also keep the LLM prompt short (cheaper, and less irrelevant text to distract the model).

**Chunk size is a trade-off:** too small and a chunk loses context ("+2.5%" without saying which country or year); too large and it mixes topics and wastes prompt space. We use 500 characters by default. **Overlap** (50 characters) repeats the end of one chunk at the start of the next, so a fact sitting on a boundary is not cut in half.

| Strategy | How it cuts | Good | Watch out |
|---|---|---|---|
| `fixed` | every N characters | simplest, fastest | cuts words and sentences in half |
| `recursive` | paragraph, then line, then sentence, then word | solid default; respects structure | ignores meaning |
| `sentence` | packs whole sentences | never breaks a sentence | uneven sizes |
| `sliding_window` | N words, moving by a step | clean word boundaries | lots of repeated text |
| `document` | by headings; each chunk keeps its heading | chunk knows its section | heading detection is a guess |
| `token` | by LLM tokens instead of characters | exact fit for model limits | needs `tiktoken` for exact counts |
| `semantic` | where the topic changes (compares embeddings of neighbouring lines) | follows meaning | one embedding per line; needs tuning |
| `agentic` | an LLM decides the cuts | best on messy text | slowest, costs money, can rewrite text |

**What we measured** (Section 9): on the GDP reports, plain `recursive` beat `semantic` and `document`. Fancier is not automatically better: short, well-structured documents already have good natural boundaries. Rule of thumb: start with `recursive`, measure, and only switch if evaluation says so.

**What we learned (bugs real documents exposed):**
- **Agentic chunking paraphrased and dropped data.** Asked to "split" a page, the LLM *rewrote* it and silently dropped the 2019–2022 table rows. Fix: the prompt now demands verbatim copying, and every result is checked: if any number from the page is missing from the chunks, the page falls back to `recursive`. Lesson: **never trust an LLM to transform your source text without verifying it.**
- **Semantic chunking found almost no boundaries** because it split on sentences, and PDF text is mostly lines (headings, table rows) without full stops: 9 "sentences" on a 51-line page. It now splits on lines *and* sentences and keeps table rows intact.
- **The document chunker lost text**: two headings in a row ("CHAPTER 1", "GLOBAL OUTLOOK"), or a number on its own line mistaken for a heading, were thrown away. Now no line is ever dropped.
- **Chunks exceeded their size limit** (517 characters at size 500) because overlap was added on top of a full chunk. Now capped.

---

## 3. Embeddings

**What:** an embedding model turns text into a vector (a list of a few hundred numbers) so that texts with similar *meaning* land close together. "Which economy exports iron ore?" lands near a chunk saying "a major global exporter of agricultural commodities and iron ore" even without many shared words. Closeness is measured with **cosine similarity** (the angle between vectors: 1 = same direction, 0 = unrelated).

| Option | Model here | Size | Trade-off |
|---|---|---|---|
| Local (`sentence_transformers`) | all-MiniLM-L6-v2 | 384 numbers | free, private, runs on CPU |
| OpenAI | text-embedding-3-small | 1536 numbers | better quality, paid, text leaves your machine |

**One rule you must never break:** documents and questions must be embedded with the **same model**. Different models produce vectors in different "spaces"; comparing them is meaningless, and nothing raises an error, you just get bad results. That is why the collection name includes the model and its size (`rag_docs_text_embedding_3_small_1536`): switching the model automatically switches to a different, empty collection instead of silently mixing.

**Weakness:** embeddings capture meaning, not exact strings. A bare number like "27,360" or a code like "E27" is fuzzy to them. That is the reason for keyword search (Section 6).

---

## 4. Vector stores

**What:** a database that keeps vectors (plus chunk text and metadata) and finds the nearest ones to a question vector fast. All four here use cosine similarity and expose the same operations (add, search with filters, delete a document, list documents), so the pipeline can swap them.

| Store | Kind | Good | Trade-off |
|---|---|---|---|
| **Chroma** | embedded, files on disk | zero setup | one process only |
| **FAISS** | in-memory index + files | very fast, exact search | one process only; we keep the text ourselves |
| **Qdrant** | server (Docker) | built for production, fast filters | needs a server |
| **pgvector** | Postgres extension | vectors next to your SQL data | slower to ingest |

**Exact vs approximate search:** FAISS (our flat index) compares the question with every vector: always correct, slower at huge scale. Chroma, Qdrant and pgvector use **HNSW**, a graph that finds *almost* the nearest neighbours very fast. "Almost" has a cost you must check, see below.

**What we measured at 50,000 chunks** (`python -m src.benchmark`):
- **pgvector silently returned wrong neighbours:** recall@10 was 0.81, meaning 19% of the true top results were missing, with no error. Its HNSW search width (`ef_search`) defaulted to 40; raising it to 100 brought recall to 0.99. Lesson: **approximate indexes trade accuracy for speed; measure recall, don't assume it.**
- **FAISS ingestion grew with the square of the size** (5× more data took 40× longer) because every batch rewrote the entire index file. Now it appends: 7× faster.
- **Filters run inside each database** (for example "only this file"). pgvector needed "iterative scan" so a filtered search still returns k results instead of fewer.
- **Two store objects on the same files go stale.** Two pipelines in one process each kept their own FAISS copy and could overwrite each other's writes. Now one store object per collection is shared, with a lock.

---

## 5. The pipeline: one entry point (P1)

**What:** `RAGPipeline` joins all the steps, so code never wires loader, chunker, embedder and store by hand:

```python
rag = RAGPipeline()                                    # settings from .env
rag = RAGPipeline(vector_store="qdrant", chunk_size=300)  # override only what you need
rag.ingest("report.pdf"); rag.query("..."); rag.answer("..."); rag.delete_document("report.pdf")
```

**Ideas worth knowing:**
- **Defaults plus overrides.** Every option has a default in `.env`; any call can override single fields. That is what makes per-request choices in the API possible. Invalid values (unknown store, overlap bigger than chunk size) fail immediately with a clear message, before any work is done.
- **Document identity.** A document is identified by its file name. **Re-ingesting a file first deletes all its old chunks**, so changing the chunker leaves nothing stale behind (old 500-character chunks mixed with new 200-character ones would distort results).
- **Fail safe.** Old chunks are deleted only *after* the new ones were successfully chunked and embedded. If the embedding API is down, the previous version stays.

---

## 6. Retrieval: finding the right chunks (P2)

Three ways to search, all available per question:

**Vector search** compares meanings (Section 3). Good for paraphrased questions; weak on exact strings.

**Keyword search (BM25)** is the classic search-engine formula. A chunk scores high when it contains the question's words, especially **rare** words (a word in every chunk, like "GDP", counts little; "27,360" appearing in one chunk counts a lot), with diminishing returns for repetition and a correction for chunk length. Good for figures, codes and names; it misses synonyms. Our tokenizer keeps "3,170", "7.2" and "e27" whole.

**Hybrid search** runs both and merges the two rankings with **Reciprocal Rank Fusion (RRF)**: each chunk gets `1/(60 + rank)` from each list, summed. Example: a chunk ranked 1st by keywords and 3rd by vectors gets 1/61 + 1/63 = 0.0323; a chunk ranked 2nd in only one list gets 1/62 = 0.0161. Chunks that **both** methods like rise to the top. RRF uses positions only, so we never have to decide how a cosine score of 0.6 compares to a BM25 score of 7.4.

**Reranking** (optional) re-scores the top ~20 candidates with a **cross-encoder**, a model that reads the question and a chunk *together* (embeddings read them separately). More precise, slower, so only the top candidates get it.

**Filters** restrict any mode to metadata, e.g. `{"source": "a.pdf", "page": [1, 2]}`.

**What we measured** (12 GDP questions, then the 30-question evaluation):

| Mode | Exact-term questions in top 3 | Paraphrased questions in top 3 |
|---|---|---|
| vector | 5/6 (missed "27,360") | 6/6 |
| keyword | 6/6 | 6/6 (weaker at top 1) |
| **hybrid** | **6/6** | **6/6** |

Hybrid is the default because it is the only mode that never missed. The reranker **hurt** bare-number lookups (it is trained on natural-language questions), so it is off by default. Lesson: a "smarter" component is not better until your own data says so.

---

## 7. Generation: turning chunks into an answer (P3)

**The prompt** gives the LLM numbered sources and strict rules: use only these sources, cite each claim as `[n]`, copy numbers exactly, and reply *"I could not find the answer in the provided documents."* when the answer isn't there. This is **grounding**: the model summarises evidence instead of recalling facts.

```
<source id="1" from="usa_gdp_report.pdf, page 1"> ... 2023 27,360 +2.5% ... </source>
Question: What was US GDP in 2023?
→ "US nominal GDP in 2023 was USD 27,360 billion [1]."
```

**Key ideas:**
- **Citations make answers checkable.** Each `[n]` maps back to file, page and chunk, so a user can verify the claim. The API renumbers them 1..n.
- **Saying "not found" is a feature.** In testing, "What is the GDP of Australia?" was answered with the not-found sentence instead of an invented number. An honest "I don't know" is worth more than a confident guess.
- **Prompt injection.** Documents can contain text like "ignore your instructions". Sources are wrapped as data and the model is told to ignore instructions inside them.
- **How many chunks to give the model (k).** Comparison questions need facts from several documents. With k=5, "compare India and the US" lost the US chunk (it ranked 6th) and the model said the US figure was missing; with k=8 both figures were found. More chunks cost more input tokens (here about 600, a fraction of a cent), so k is a quality/cost dial.

**Choosing the model (mixed providers):** the model name decides the provider. You choose per question.

| Provider | Examples | Cost character |
|---|---|---|
| OpenAI | gpt-5-nano, gpt-4o-mini, gpt-5-mini, gpt-5.4, gpt-5.5 | pay per token |
| Anthropic | claude-haiku-5-5, claude-sonnet-5-5, claude-opus-5-5 | pay per token |
| Ollama | llama3.2:3b, qwen2.5:7b | free, runs on your machine, slower and weaker |

**Tokens and cost:** providers charge per token (roughly ¾ of a word), separately for input (prompt + chunks) and output (the answer), listed as dollars per million tokens. Example: gpt-4o-mini at $0.15 in / $0.60 out, with 1,400 input and 21 output tokens per question, costs about $0.0002 per answer. Every answer reports its token counts, and the catalog (`src/llm/catalog.py`) records prices and tiers, so cost-based selection (budgets, cheap-vs-strong routing) can be added later.

**Reasoning models are not automatically better for RAG.** gpt-5-mini "thinks" before answering: in our evaluation it used 14× more output tokens and was 2.5× slower than gpt-4o-mini, with no accuracy gain on these factual lookups. Reasoning pays off on hard multi-step questions, not on "find and quote".

**Refusals and fallbacks (Claude):** a model may decline a request for safety reasons. For Claude Opus/Sonnet 5.5 the pipeline enables Anthropic's server-side fallback, which re-runs a declined request on a recommended fallback model within the same call; a refusal that still happens raises a clear error (HTTP 422 in the API).

---

## 8. Serving it: the HTTP API (P4)

**What:** a FastAPI service so other programs (a UI, a chatbot) can use the pipeline: upload documents, search, ask, list, delete. Interactive docs at `/docs`.

**Design ideas worth knowing:**
- **Named indexes.** The vector store and embedding model together decide *where* data lives. If a user uploads with one pair and asks with another, the search hits an empty collection and silently answers "not found". So users pick an index by **name** (defined by the operator), which guarantees ingest and query match. Users still choose chunker, retrieval settings and LLM freely per request.
- **Background jobs for slow work.** Ingesting a 45-page report with the semantic chunker takes minutes, longer than an HTTP request should wait. Upload returns a job id immediately (HTTP 202); the client polls `/v1/jobs/{id}`. Small uploads can use `?wait=true`.
- **A predictable error contract.** Every error has the same shape `{"error": {"code", "message"}, "request_id"}` and a meaningful status: 400 bad option, 404 not found, 413 too large, 415 not a PDF, 422 refused, 502 the LLM provider failed, 503 the database is down. Clients can react to the code instead of parsing text.
- **Strict input.** Unknown fields are rejected: a typo like `"chunkin"` returns an error instead of silently using defaults.
- **Guardrails:** upload size limit, real-PDF check (`%PDF-` header), allowed chunkers (the operator can forbid the LLM-priced `agentic`), maximum k.
- **One worker by default**, because FAISS and Chroma are single-process stores; scaling out needs Qdrant/pgvector and an external job queue.

**Protecting it: API keys and rate limits (P6).**
- **Authentication** answers "who is calling?". Each client (the web app, a CI job) gets its own long random key, sent in the `X-API-Key` header. The server maps the key to a client *name* (so logs say `client=webapp`, never the key). Keys are compared in constant time, so an attacker can't learn a key character by character from response times.
- **Rate limiting** answers "how much may they call?". It protects the service from overload and, here, from runaway costs, because every `/v1/answer` calls a paid LLM. We use a **token bucket**: each client has a bucket of N tokens per minute that refills continuously; each request takes one; an empty bucket means HTTP 429 with a `Retry-After` header saying when to try again. Answers get a smaller, separate bucket than cheap calls like listing documents.
- Health checks and the docs page stay open, so monitoring and developers are not locked out. Without configured keys the API runs open (for development) and warns loudly at startup.

**Lesson from Docker:** the container crashed on start because `.env` set `LOG_DIR=./logs`, a folder the container user cannot write. Settings that only make sense *inside* the container (paths, service host names) are pinned in `docker-compose.yml`, which takes precedence over `.env`.

---

## 9. Evaluation: measuring quality (P5)

**Why:** RAG fails quietly. It returns fluent, confident answers even when retrieval fetched the wrong chunk. Unit tests prove the code runs; only evaluation tells you whether the answers are right. It also replaces opinions ("semantic chunking sounds smarter") with numbers.

**The question set** (`eval/gdp_questions.jsonl`, 30 questions) mixes question types on purpose, because each stresses a different part: table figures, simple lookups, outlook text, paraphrased questions, rankings, two-document comparisons, and **4 unanswerable questions** that test whether the system admits it doesn't know. For each question we record *where* the evidence is (file and page, with alternatives, because GDP figures appear in two reports) and *which facts* the answer must contain. A test verifies every expected fact really appears on its evidence page, so mistakes in the answer key are caught.

**Measure retrieval and answers separately**, so you know *which* stage failed:

| Metric | Question it answers |
|---|---|
| **hit@k** | Was the evidence page among the top k chunks? (comparisons need every document) |
| **MRR** | How high did it rank? 1st = 1.0, 2nd = 0.5, 4th = 0.25 |
| **fact_acc** | Does the answer contain every expected fact? |
| **cited_ok** | Do the citations point to the right documents? |
| **false_nf** | Did it wrongly say "not found"? (usually a retrieval miss) |
| **abstain_acc** | For unanswerable questions, did it correctly say "not found"? |
| **faithful** | Does a judge model find every claim supported by the sources? |

**LLM-as-judge:** checking "is every claim supported?" by hand is slow, so a second LLM does it. Useful, but a judge is a model too: ours wrongly flagged "Germany ranked third" as unsupported although the ranking table says exactly that. Treat judge flags as "look at this", not as verdicts, and use a stronger judge for important decisions.

**Results on the GDP reports** (local embeddings, FAISS, 8 chunks):

| Retrieval | top 3 | top 5 |
|---|---|---|
| **recursive + hybrid** | **0.96** | **1.00** |
| recursive + vector | 0.81 | 0.96 |
| semantic + hybrid | 0.89 | 0.96 |

| Answers (recursive + hybrid) | facts | citations | correct "not found" | faithful | time |
|---|---|---|---|---|---|
| **gpt-4o-mini** | **100%** | **100%** | 100% | 100% | 1.4 s |
| gpt-5-mini | 96% | 96% | 100% | 96% | 3.4 s |

**How to read results honestly:**
- **Sample size.** With 30 questions, one question is 3-4 points. Differences smaller than that are noise.
- **Evaluate the evaluation.** Our first answer run exposed a mistake in the answer key (a comparison question listed its evidence too narrowly). Look at the failing rows before trusting a score.
- **Results don't transfer automatically.** These are short, clean, synthetic reports. A 45-page two-column report may favour different settings; build a question set for *your* documents before choosing.
- **Cheap first, then justify upgrades.** Here the cheapest settings won. That is common for factual lookup over clean documents.

---

## 9b. Continuous integration (CI)

**CI** means every change is automatically tested on a clean machine before it is trusted. A pipeline file describes the steps; the server (Azure DevOps or GitHub) runs them on each push or pull request and shows pass/fail.

Ours installs everything from scratch, starts real Qdrant and Postgres containers next to the tests (so the live database tests run instead of skipping), runs all tests, and builds the Docker image. Things worth knowing:
- **"Works on my machine" is the bug CI catches.** Our tests used to read sample PDFs from the data folder you edit; a clean CI checkout would not have your files. The samples now live in `tests/data/`, committed with the tests.
- **Secrets stay out of the code.** Paid LLM tests run only when API keys are configured as secret pipeline variables; otherwise they skip, so CI is free by default.
- **Each platform has quirks.** In Azure Pipelines an undefined variable is passed as the literal text `$(OPENAI_API_KEY)`, which looks like a real key; the pipeline treats that text as "not set".

## 10. Observability: logging and request IDs

Every pipeline step writes one timed log line (`load_pdf`, `chunk`, `embed`, `store.add`, `retrieve`, `query`, `llm`, `answer`) with counts, scores, token usage and milliseconds, but never document text or keys. Each request gets an **ID** (echoed in API responses as `X-Request-ID`) that appears on every line, including background jobs, so one user's problem can be traced end to end:

```
INFO [req=c79b622a] src.pipeline: query done mode=hybrid k=8 hits=6 ms=397
INFO [req=c79b622a] src.llm: llm done model=gpt-4o-mini input_tokens=911 output_tokens=22 ms=983
```

Logs show where time and money go (here: the LLM call dominates latency), which is the starting point for any optimisation.

---

## 11. Lessons learned from testing on real data

Each of these was invisible in unit tests with clean sample data and showed up only with a real document, a real model or real scale:

1. **LLMs rewrite text when asked to split it** → verify LLM output against the source (agentic chunker).
2. **PDF text is lines, not sentences** → chunkers must handle tables and headings (semantic chunker).
3. **Heuristics drop data on unexpected layouts** → never discard text silently (document chunker).
4. **Approximate search can silently lose accuracy at scale** → measure recall (pgvector).
5. **Costs that look small grow with the square of the data** → benchmark at realistic scale (FAISS).
6. **Comparison questions need more context** → k is a quality/cost dial (generation).
7. **Shared mutable state breaks under concurrency** → one store per collection with a lock (API).
8. **Configuration leaks between environments** → pin container settings (Docker).
9. **Your answer key can be wrong too** → inspect failures before trusting a score (evaluation).
10. **"Smarter" components can be worse on your data** → reranker, semantic chunking and reasoning models all lost here.
11. **Reading order matters as much as the words** → column-aware extraction lifted top-1 retrieval on a two-column report from 0.4 to 0.7.
12. **Clean-up rules can delete content** → the first header remover also removed body lines and would have removed numeric table rows; it now only removes lines with words, from page edges, and short pages are treated carefully.
13. **Test data belongs with the tests** → CI exposed that tests depended on a folder users change.

---

## 12. Project status and what's next

| Area | Status |
|---|---|
| Ingestion, 8 chunkers, 2 embedders, 4 vector stores | Done and tested (stores verified against live servers) |
| P1 Pipeline: one entry point, collection per model, replace/delete documents | Done |
| P2 Retrieval: vector, keyword (BM25), hybrid (RRF), filters, optional reranker | Done |
| P3 Generation: OpenAI / Anthropic / Ollama, model per question, citations, not-found | Done; Anthropic and Ollama not yet tested live (need a key / a pulled model) |
| P4 HTTP API: named indexes, background jobs, error contract, Docker | Done |
| P5 Evaluation: 30-question set, retrieval + answer metrics, judge | Done |
| P6 Hardening: two-column PDFs, headers, OCR; API keys + rate limits; CI | Done |
| Cost features (budgets, auto-router, cache, cost tracking) | Prepared (prices + token counts recorded), not active |
| Remaining: cost features, FAISS update speed, pgvector bulk load, `tiktoken` | Next |

**Next steps:**
- Extend the `content.pdf` question set (10 questions so far) and evaluate answers on it, because the GDP results may not transfer.
- The deferred cost features: budget per question, cheap-vs-strong routing, response cache, cost tracking.
- A web UI (see [UI.md](UI.md)).
- Smaller items: faster FAISS updates, pgvector bulk loading, moving test data to `tests/data/`, exact token counts with `tiktoken`.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **Chunk** | A small piece of a document that is searched and shown to the LLM |
| **Embedding / vector** | A list of numbers representing the meaning of a text |
| **Cosine similarity** | How closely two vectors point the same way (1 = identical meaning) |
| **Collection** | One searchable set of vectors, tied to one embedding model |
| **HNSW** | A graph index for fast, approximate nearest-neighbour search |
| **Recall@k** | Share of the true top-k results that a search actually returned |
| **BM25** | Classic keyword-ranking formula that favours rare matching words |
| **RRF** | Reciprocal Rank Fusion: merges ranked lists by position (1 / (60 + rank)) |
| **Cross-encoder / reranker** | A model that reads question and chunk together to re-score them |
| **k** | How many chunks are retrieved / given to the LLM |
| **Grounding** | Making the LLM answer only from supplied sources |
| **Citation** | A `[n]` marker linking a claim to the source chunk it came from |
| **Token** | The unit LLMs read and bill by, roughly ¾ of an English word |
| **Prompt injection** | Instructions hidden in data that try to take over the model |
| **Hallucination** | A fluent but unsupported or invented statement |
| **hit@k / MRR** | Was the evidence in the top k / how high did it rank |
| **LLM-as-judge** | Using a model to grade another model's answers |
| **Request ID** | An identifier that ties all log lines of one request together |
| **Gutter** | The empty vertical strip between two text columns |
| **OCR** | Optical character recognition: reading text from an image (scanned pages) |
| **API key** | A secret string that identifies a client to the API |
| **Rate limit / token bucket** | A cap on requests per minute; each request uses a token that refills over time |
| **CI** | Continuous integration: tests run automatically on every change |
