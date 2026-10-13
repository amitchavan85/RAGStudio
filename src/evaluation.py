"""
Evaluation (roadmap P5): measure retrieval and answer quality on questions with known answers,
and compare configurations (chunker x store x retrieval mode x LLM) with numbers.

    python -m src.evaluation                                   # eval/gdp_questions.jsonl, .env defaults
    python -m src.evaluation --chunkers recursive semantic --modes hybrid vector --models gpt-4o-mini claude-haiku-5-5
    python -m src.evaluation --retrieval-only                  # no LLM calls (free with local embeddings)
    python -m src.evaluation --judge gpt-4o-mini               # + LLM-as-judge faithfulness check

Dataset: JSONL, one question per line:
    {"id", "type", "question",
     "evidence": [[["file.pdf", page], ...alternatives], ...one group per required fact source],
     "facts": ["27,360|27360", ...]   # every entry must appear in the answer; "|" separates alternatives
     "answerable": false}             # for questions the documents cannot answer (expects "not found")

Metrics (answerable questions unless noted):
  hit@n        all evidence groups appear in the top n retrieved chunks (page-level)
  mrr          mean reciprocal rank of the evidence groups
  fact_acc     answer contains every expected fact
  cited_ok     every [n] citation points to an evidence document
  false_nf     wrongly answered "not found"
  abstain_acc  unanswerable questions answered with "not found" (unanswerable only)
  faithful     judge found no unsupported claim (only with --judge)
Ingestion runs once per (chunker, store) in a temporary data folder; your real indexes are untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

from src.config import PROJECT_ROOT, get_settings

DEFAULT_DATASET = PROJECT_ROOT / "eval" / "gdp_questions.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
HIT_AT = (1, 3, 5)


# ---------- dataset ----------

@dataclass
class EvalQuestion:
    id: str
    question: str
    type: str = "general"
    answerable: bool = True
    evidence: List[List[Tuple[str, int]]] = field(default_factory=list)
    facts: List[List[str]] = field(default_factory=list)

    @property
    def evidence_sources(self) -> set:
        return {src for group in self.evidence for src, _ in group}


def load_dataset(path) -> List[EvalQuestion]:
    questions, ids = [], set()
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            q = EvalQuestion(
                id=raw["id"], question=raw["question"], type=raw.get("type", "general"),
                answerable=raw.get("answerable", True),
                evidence=[[(s, int(p)) for s, p in group] for group in raw.get("evidence", [])],
                facts=[[alt.strip().lower() for alt in fact.split("|") if alt.strip()] for fact in raw.get("facts", [])],
            )
            if q.id in ids:
                raise ValueError(f"{path}:{n}: duplicate id {q.id!r}")
            if q.answerable and not (q.evidence and q.facts):
                raise ValueError(f"{path}:{n}: answerable question {q.id!r} needs evidence and facts")
            ids.add(q.id)
            questions.append(q)
    return questions


# ---------- metrics ----------

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("−", "-"))


def fact_present(text: str, alternatives: Sequence[str]) -> bool:
    """True if any alternative occurs as a whole token sequence ("2.5" does not match "12.5")."""
    t = _norm(text)
    return any(re.search(rf"(?<![a-z0-9]){re.escape(_norm(alt))}(?![a-z0-9])", t) for alt in alternatives)


def retrieval_metrics(q: EvalQuestion, hits: list) -> dict:
    found = [(h.metadata.get("source"), h.metadata.get("page")) for h in hits]
    ranks = []
    for group in q.evidence:
        allowed = set(group)
        ranks.append(next((i for i, sp in enumerate(found, 1) if sp in allowed), None))
    out = {f"hit@{n}": all(r is not None and r <= n for r in ranks) for n in HIT_AT}
    out["hit@k"] = all(r is not None for r in ranks)
    out["mrr"] = statistics.mean((1 / r) if r else 0.0 for r in ranks)
    out["ranks"] = ranks
    return out


def answer_metrics(q: EvalQuestion, ans) -> dict:
    if not q.answerable:
        return {"abstain_ok": not ans.found}
    cited = {c["source"] for c in ans.citations}
    return {
        "fact_ok": ans.found and all(fact_present(ans.text, alts) for alts in q.facts),
        "fact_recall": sum(fact_present(ans.text, alts) for alts in q.facts) / len(q.facts),
        "cited_ok": bool(cited) and cited <= q.evidence_sources,
        "false_nf": not ans.found,
    }


JUDGE_PROMPT = """You check whether an answer is fully supported by the given sources.

Sources:
{sources}

Answer to check:
{answer}

List every factual claim in the answer that is NOT stated in the sources (citation markers like [1] are not claims).
Reply with JSON only: {{"unsupported": ["claim", ...]}} (an empty list if everything is supported)."""


def judge_faithfulness(judge, ans) -> dict:
    sources = "\n\n".join(f"[{i}] {s.text.strip()}" for i, s in enumerate(ans.sources, 1))
    raw = judge.generate(JUDGE_PROMPT.format(sources=sources, answer=ans.text), max_tokens=800).text
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        unsupported = json.loads(m.group(0)).get("unsupported", []) if m else None
    except json.JSONDecodeError:
        unsupported = None
    if unsupported is None:
        return {"faithful": None, "judge_raw": raw[:300]}
    return {"faithful": not unsupported, "unsupported": unsupported}


# ---------- running ----------

@dataclass(frozen=True)
class EvalConfig:
    chunker: str
    store: str
    mode: str
    model: Optional[str]  # None = retrieval only
    reranker: str = "none"
    chunk_size: int = 500
    overlap: int = 50

    @property
    def label(self) -> str:
        rr = "+rerank" if self.reranker != "none" else ""
        return f"{self.chunker}/{self.store}/{self.mode}{rr}/{self.model or '-'}"


def summarize(rows: List[dict]) -> dict:
    ans_rows = [r for r in rows if r["answerable"]]
    na_rows = [r for r in rows if not r["answerable"]]

    def rate(rs, key):
        vals = [r[key] for r in rs if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    out = {f"hit@{n}": rate(ans_rows, f"hit@{n}") for n in HIT_AT}
    out.update({"hit@k": rate(ans_rows, "hit@k"), "mrr": round(statistics.mean(r["mrr"] for r in ans_rows), 3) if ans_rows else None})
    if any("fact_ok" in r for r in ans_rows):
        out.update({
            "fact_acc": rate(ans_rows, "fact_ok"), "cited_ok": rate(ans_rows, "cited_ok"),
            "false_nf": rate(ans_rows, "false_nf"), "abstain_acc": rate(na_rows, "abstain_ok"),
            "faithful": rate(ans_rows, "faithful"),
            "avg_in_tokens": round(statistics.mean(r.get("input_tokens", 0) for r in rows)),
            "avg_out_tokens": round(statistics.mean(r.get("output_tokens", 0) for r in rows)),
        })
    out["avg_latency_s"] = round(statistics.mean(r["latency_s"] for r in rows), 2)
    return out


def run_eval(
    questions: List[EvalQuestion],
    configs: List[EvalConfig],
    docs: List[str],
    k: int = 8,
    embedding_provider: Optional[str] = None,
    judge_model: Optional[str] = None,
    embedder=None,
    llm=None,
    progress=print,
) -> dict:
    """Ingest docs once per (chunker, size, store), then ask every question for every config.

    embedder / llm: injected fakes for tests; otherwise built from the config.
    """
    from src.pipeline import RAGPipeline, reset_shared_state

    tmp = tempfile.mkdtemp(prefix="rag_eval_")
    old_data_dir = os.environ.get("DATA_DIR")
    os.environ["DATA_DIR"] = tmp  # local stores (FAISS/Chroma) go to a throwaway folder
    get_settings.cache_clear()
    reset_shared_state()
    judge = None
    if judge_model:
        from src.llm import get_llm

        judge = get_llm(judge_model)

    results = {"configs": [], "started_at": datetime.now().isoformat(timespec="seconds"), "k": k}
    ingested: Dict[tuple, RAGPipeline] = {}
    try:
        for cfg in configs:
            key = (cfg.chunker, cfg.chunk_size, cfg.overlap, cfg.store)
            if key not in ingested:
                rag = RAGPipeline(
                    chunking_strategy=cfg.chunker, chunk_size=cfg.chunk_size, chunk_overlap=cfg.overlap,
                    vector_store=cfg.store, embedding_provider=embedding_provider,
                    collection_name=f"eval_{cfg.chunker}_{cfg.chunk_size}", embedder=embedder, llm=llm,
                )
                rag.store.reset()  # server stores (Qdrant/pgvector) keep data between runs
                t = time.perf_counter()
                report = rag.ingest(docs)
                progress(f"ingested {cfg.chunker}/{cfg.store}: {report.total_chunks} chunks in {time.perf_counter() - t:.1f}s")
                ingested[key] = rag
            base = ingested[key]
            rag = RAGPipeline(base.config, embedder=base.embedder, llm=llm, retrieval_mode=cfg.mode,
                              reranker=cfg.reranker, llm_model=cfg.model)
            rag._store = base.store

            rows = []
            for q in questions:
                t = time.perf_counter()
                row = {"id": q.id, "type": q.type, "answerable": q.answerable}
                if cfg.model:
                    ans = rag.answer(q.question, k=k)
                    hits = ans.sources
                    row.update(answer=ans.text, citations=[c["source"] for c in ans.citations],
                               input_tokens=ans.input_tokens, output_tokens=ans.output_tokens)
                    row.update(answer_metrics(q, ans))
                    if judge and q.answerable and ans.found:
                        row.update(judge_faithfulness(judge, ans))
                else:
                    hits = rag.query(q.question, k=k)
                row["latency_s"] = round(time.perf_counter() - t, 3)
                if q.answerable:
                    row.update(retrieval_metrics(q, hits))
                rows.append(row)
            summary = summarize(rows)
            progress(f"{cfg.label}: {summary}")
            results["configs"].append({"config": asdict(cfg), "label": cfg.label, "summary": summary, "questions": rows})
    finally:
        for rag in ingested.values():
            if rag.config.vector_store in ("qdrant", "pgvector"):
                try:
                    rag.store.reset()  # don't leave evaluation collections on the servers
                except Exception:
                    pass
        reset_shared_state()
        if old_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = old_data_dir
        get_settings.cache_clear()
        shutil.rmtree(tmp, ignore_errors=True)
    return results


def format_table(results: dict) -> str:
    cols = ["hit@1", "hit@3", "hit@5", "hit@k", "mrr", "fact_acc", "cited_ok", "false_nf", "abstain_acc", "faithful", "avg_latency_s"]
    present = [c for c in cols if any(c in r["summary"] and r["summary"][c] is not None for r in results["configs"])]
    width = max(len(r["label"]) for r in results["configs"]) if results["configs"] else 10
    lines = [f"{'config':{width}}  " + "  ".join(f"{c:>9}" for c in present)]
    for r in results["configs"]:
        s = r["summary"]
        lines.append(f"{r['label']:{width}}  " + "  ".join(f"{'' if s.get(c) is None else s[c]:>9}" for c in present))
    return "\n".join(lines)


def main() -> None:
    from src.chunking import AVAILABLE_STRATEGIES
    from src.logging_config import setup_logging
    from src.pipeline import RETRIEVAL_MODES
    from src.vectorstores import AVAILABLE_STORES

    s = get_settings()
    ap = argparse.ArgumentParser(description="Evaluate retrieval and answers on questions with known answers.")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--docs", nargs="*", help="PDFs to index (default: every document the dataset's evidence names, from PDF_DIR)")
    ap.add_argument("--chunkers", nargs="+", default=[s.chunking_strategy], choices=AVAILABLE_STRATEGIES)
    ap.add_argument("--stores", nargs="+", default=[s.vector_store], choices=AVAILABLE_STORES)
    ap.add_argument("--modes", nargs="+", default=[s.retrieval_mode], choices=RETRIEVAL_MODES)
    ap.add_argument("--models", nargs="+", default=[s.llm_model], help="LLMs to compare")
    ap.add_argument("--rerank", action="store_true", help="also evaluate each config with the cross-encoder reranker")
    ap.add_argument("--chunk-size", type=int, default=s.chunk_size)
    ap.add_argument("--overlap", type=int, default=s.chunk_overlap)
    ap.add_argument("--embedding-provider", choices=("sentence_transformers", "openai"))
    ap.add_argument("-k", type=int, default=8, help="chunks retrieved / given to the LLM")
    ap.add_argument("--retrieval-only", action="store_true", help="skip answers (no LLM calls)")
    ap.add_argument("--judge", help="LLM that checks answer faithfulness, e.g. gpt-4o-mini")
    args = ap.parse_args()

    setup_logging(level="WARNING")
    questions = load_dataset(args.dataset)
    docs = args.docs or sorted({str(s.pdf_path / src) for q in questions for src in q.evidence_sources})
    models = [None] if args.retrieval_only else args.models
    rerankers = ["none", "cross_encoder"] if args.rerank else ["none"]
    configs = [EvalConfig(c, st, m, mo, rr, args.chunk_size, args.overlap)
               for c, st, m, mo, rr in product(args.chunkers, args.stores, args.modes, models, rerankers)]
    llm_calls = 0 if args.retrieval_only else len(configs) * len(questions) * (2 if args.judge else 1)
    print(f"{len(questions)} questions x {len(configs)} configs, {len(docs)} documents, ~{llm_calls} LLM calls")

    results = run_eval(questions, configs, docs, k=args.k, embedding_provider=args.embedding_provider,
                       judge_model=args.judge)
    results["dataset"] = os.path.relpath(args.dataset, PROJECT_ROOT)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{os.path.splitext(os.path.basename(args.dataset))[0]}.json"
    out.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    print("\n" + format_table(results))
    print(f"\nPer-question details: {os.path.relpath(out, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
