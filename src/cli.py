"""
Command-line front end for the RAG pipeline.

    python -m src.cli ingest report.pdf other.pdf      # or a folder; no path = PDF_DIR
    python -m src.cli query "What was GDP growth in 2023?" -k 3
    python -m src.cli list
    python -m src.cli delete report.pdf

Every command accepts the same overrides as the pipeline, e.g.
    python -m src.cli --store faiss --chunker semantic ingest report.pdf
Defaults come from .env. Use the same --store / --embedding-provider for ingest and query:
they decide which collection is used.
"""
from __future__ import annotations

import argparse

from src.chunking import AVAILABLE_STRATEGIES
from src.logging_config import new_request_id, setup_logging
from src.pipeline import EMBEDDING_PROVIDERS, RAGPipeline
from src.vectorstores import AVAILABLE_STORES


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG pipeline: ingest, query, list, delete.")
    ap.add_argument("--chunker", choices=AVAILABLE_STRATEGIES)
    ap.add_argument("--chunk-size", type=int)
    ap.add_argument("--overlap", type=int)
    ap.add_argument("--store", choices=AVAILABLE_STORES)
    ap.add_argument("--embedding-provider", choices=EMBEDDING_PROVIDERS)
    ap.add_argument("--embedding-model")
    ap.add_argument("--collection", help="base collection name (the embedding model is appended)")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ingest", help="ingest PDFs (re-ingesting a file replaces its chunks)")
    p.add_argument("paths", nargs="*", help="files or folders; default: PDF_DIR")
    p = sub.add_parser("query", help="retrieve the most relevant chunks")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--mode", choices=("hybrid", "vector", "keyword"), help="default: RETRIEVAL_MODE")
    p.add_argument("--rerank", action="store_true", help="re-score candidates with the local cross-encoder")
    p.add_argument("--source", action="append", help="only search this document (repeatable)")
    p.add_argument("--page", type=int, action="append", help="only search this page (repeatable)")
    p = sub.add_parser("ask", help="answer a question from the documents, with citations")
    p.add_argument("question")
    p.add_argument("--model", help="e.g. gpt-4o-mini, claude-haiku-5-5, ollama:llama3.2:3b (default: LLM_MODEL)")
    p.add_argument("-k", type=int, default=8, help="chunks given to the model")
    p.add_argument("--mode", choices=("hybrid", "vector", "keyword"))
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--source", action="append")
    p.add_argument("--page", type=int, action="append")
    p.add_argument("--show-sources", action="store_true", help="print every retrieved chunk, not only cited ones")
    sub.add_parser("models", help="list the models that can be used with --model")
    sub.add_parser("list", help="list ingested documents")
    p = sub.add_parser("delete", help="delete a document by file name")
    p.add_argument("source")
    args = ap.parse_args()

    setup_logging()
    new_request_id()
    if args.command == "models":
        from src.llm import MODELS
        from src.llm.ollama_provider import list_local_models

        local = set(list_local_models())
        print(f"{'model':20} {'provider':10} {'tier':9} {'$/1M in':>8} {'$/1M out':>9}")
        for name, m in MODELS.items():
            note = "" if m.provider != "ollama" else ("  (pulled)" if name in local else "  (not pulled)")
            print(f"{name:20} {m.provider:10} {m.tier:9} {m.input_per_m:>8.2f} {m.output_per_m:>9.2f}{note}")
        print("Any other model: --model <provider>:<model>, e.g. ollama:mistral:7b")
        return
    rag = RAGPipeline(
        chunking_strategy=args.chunker, chunk_size=args.chunk_size, chunk_overlap=args.overlap,
        vector_store=args.store, embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model, collection_name=args.collection,
    )
    where = f"{rag.config.vector_store}:{rag.collection}"

    if args.command == "ingest":
        report = rag.ingest(args.paths or None)
        for f in report.files:
            note = f" (replaced {f['replaced']} old chunks)" if f["replaced"] else ""
            print(f"  {f['source']}: {f['pages']} pages -> {f['chunks']} chunks{note}")
        for s in report.skipped:
            print(f"  SKIPPED {s['source']}: {s['reason']}")
        print(f"{len(report.files)} file(s), {report.total_chunks} chunks -> {where}")
    elif args.command == "query":
        filters = {key: vals for key, vals in (("source", args.source), ("page", args.page)) if vals}
        hits = rag.query(args.question, k=args.k, filters=filters or None, mode=args.mode,
                         reranker="cross_encoder" if args.rerank else None)
        print(f"Q: {args.question}   [{where}, mode={args.mode or rag.config.retrieval_mode}"
              f"{', reranked' if args.rerank else ''}]")
        for i, h in enumerate(hits, 1):
            snippet = " ".join(h.text.split())[:150]
            stages = " ".join(f"{name}={val:.3f}" for name, val in h.scores.items())
            print(f"  {i}. {h.metadata.get('source')} p{h.metadata.get('page')} [{stages}] | {snippet}")
    elif args.command == "ask":
        filters = {key: vals for key, vals in (("source", args.source), ("page", args.page)) if vals}
        ans = rag.answer(args.question, k=args.k, filters=filters or None, mode=args.mode,
                         reranker="cross_encoder" if args.rerank else None, model=args.model)
        print(f"Q: {ans.question}\n")
        print(ans.text + "\n")
        shown = ans.sources if args.show_sources else [ans.sources[c["n"] - 1] for c in ans.citations]
        cited = {c["n"] for c in ans.citations}
        for i, s in enumerate(ans.sources, 1):
            if s in shown:
                mark = "*" if i in cited else " "
                print(f" {mark}[{i}] {s.metadata.get('source')} p{s.metadata.get('page')}: {' '.join(s.text.split())[:110]}")
        if ans.model:
            print(f"\n[{ans.provider}:{ans.model}, {ans.input_tokens} in / {ans.output_tokens} out tokens]")
    elif args.command == "list":
        docs = rag.list_documents()
        print(f"{len(docs)} document(s) in {where}")
        for source, n in sorted(docs.items()):
            print(f"  {source}: {n} chunks")
    elif args.command == "delete":
        n = rag.delete_document(args.source)
        print(f"deleted {n} chunks of {args.source} from {where}" if n else f"{args.source} not found in {where}")


if __name__ == "__main__":
    main()
