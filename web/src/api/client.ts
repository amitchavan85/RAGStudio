/** Typed client for the RAG API. Types come from the generated OpenAPI schema (npm run api-types). */
import type { components } from "./schema";

type S = components["schemas"];
/** Every field except the question has a server default, so it is optional when sending. */
export type AnswerRequest = Partial<S["AnswerRequest"]> & { question: string };
export type AnswerResponse = S["AnswerResponse"];
export type SearchResponse = S["SearchResponse"];
export type Citation = S["Citation"];
export type ChunkOut = S["ChunkOut"];
export type Job = S["Job"];
export type DocumentList = S["DocumentList"];
export type IndexOut = S["IndexOut"];
export type ModelOut = S["ModelOut"];
export type RetrievalOptions = S["RetrievalOptions"];

export type Options = {
  indexes: string[];
  allow_custom_index: boolean;
  chunking_strategies: string[];
  retrieval_modes: string[];
  rerankers: string[];
  defaults: {
    chunking_strategy: string;
    chunk_size: number;
    chunk_overlap: number;
    retrieval_mode: string;
    reranker: string;
    llm_model: string;
    search_k: number;
    answer_k: number;
  };
  limits: { max_upload_mb: number; max_k: number };
};

/** Error with the API's machine-readable code (see the README error contract). */
export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public details?: Record<string, unknown> | null,
    public requestId?: string,
  ) {
    super(message);
  }
}

const KEY_STORAGE = "rag.apiKey";

export function getApiKey(): string {
  try {
    return localStorage.getItem(KEY_STORAGE) ?? "";
  } catch {
    return "";
  }
}

export function setApiKey(key: string): void {
  try {
    if (key) localStorage.setItem(KEY_STORAGE, key);
    else localStorage.removeItem(KEY_STORAGE);
  } catch {
    /* storage unavailable (private mode): the key lives only for this page */
  }
}

/** Called when the server answers 401, so the app can ask for a key. */
let onUnauthorized: (() => void) | null = null;
export function setUnauthorizedHandler(fn: () => void): void {
  onUnauthorized = fn;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const key = getApiKey();
  if (key) headers.set("X-API-Key", key);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");

  let res: Response;
  try {
    res = await fetch(path, { ...init, headers });
  } catch {
    throw new ApiError(0, "network_error", "Cannot reach the API server.");
  }
  if (res.ok) {
    const type = res.headers.get("content-type") ?? "";
    return (type.includes("application/json") ? res.json() : res.blob()) as Promise<T>;
  }
  let body: { error?: { code?: string; message?: string; details?: Record<string, unknown> }; request_id?: string } = {};
  try {
    body = await res.json();
  } catch {
    /* non-JSON error body */
  }
  if (res.status === 401) onUnauthorized?.();
  const details = { ...(body.error?.details ?? {}) };
  const retry = res.headers.get("Retry-After");
  if (retry) details.retry_after_seconds = Number(retry);
  throw new ApiError(
    res.status,
    body.error?.code ?? `http_${res.status}`,
    body.error?.message ?? res.statusText,
    details,
    body.request_id ?? res.headers.get("X-Request-ID") ?? undefined,
  );
}

const qs = (index: string) => `index=${encodeURIComponent(index)}`;

export const api = {
  options: () => request<Options>("/v1/options"),
  models: () => request<ModelOut[]>("/v1/models"),
  indexes: () => request<IndexOut[]>("/v1/indexes"),
  documents: (index: string) => request<DocumentList>(`/v1/documents?${qs(index)}`),
  deleteDocument: (index: string, source: string) =>
    request<S["DeleteResult"]>(`/v1/documents/${encodeURIComponent(source)}?${qs(index)}`, { method: "DELETE" }),
  documentFile: (index: string, source: string) =>
    request<Blob>(`/v1/documents/${encodeURIComponent(source)}/file?${qs(index)}`),
  upload: (index: string, files: File[], chunking: { strategy?: string } = {}) => {
    const form = new FormData();
    for (const f of files) form.append("files", f, f.name);
    form.append("options", JSON.stringify({ index, chunking }));
    return request<Job>("/v1/documents", { method: "POST", body: form });
  },
  job: (id: string) => request<Job>(`/v1/jobs/${encodeURIComponent(id)}`),
  search: (body: { question: string; index: string; retrieval?: RetrievalOptions }) =>
    request<SearchResponse>("/v1/search", { method: "POST", body: JSON.stringify(body) }),
  answer: (body: AnswerRequest) => request<AnswerResponse>("/v1/answer", { method: "POST", body: JSON.stringify(body) }),
};
