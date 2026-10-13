"""
Step 2: PDF ingestion.

Turns PDF files into a list of `Document` objects: raw text plus
metadata (source filename, page number). This is the unglamorous but
critical first stage of any RAG pipeline -- if extraction is wrong or
lossy, nothing downstream (chunking, embedding, retrieval) can fix it.

Design notes for production:
  - We keep one Document per PAGE (not per file) so we can cite an
    exact page number back to the user later.
  - Tables are extracted separately from prose text, because
    `extract_text()` alone frequently mangles tabular data (columns
    get smashed together). We append a normalized "[TABLE]" block so
    the numbers survive, at the cost of losing perfect table structure
    -- a reasonable trade-off for a text-based RAG system. If your
    documents are table-heavy, consider a dedicated table-extraction
    library (e.g. camelot, or an "unstructured"-style layout model)
    instead.
  - We skip empty pages rather than embedding blank chunks later.
  - Layout clean-up (two-column reading order, running headers/footers, drop capitals)
    and OCR for scanned pages live in layout.py.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List

import pdfplumber

from src.logging_config import log_step

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A single unit of extracted content with metadata."""

    text: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        preview = self.text[:60].replace("\n", " ")
        return (
            f"Document(source={self.metadata.get('source')!r}, "
            f"page={self.metadata.get('page')}, text={preview!r}...)"
        )


def load_pdf(path: str) -> List[Document]:
    """Extract one Document per non-empty page from a single PDF."""
    with log_step(logger, "load_pdf", file=os.path.basename(path)) as out:
        docs = _extract_pages(path)
        out["pages"] = len(docs)
    return docs


def _extract_pages(path: str) -> List[Document]:
    from src.config import get_settings
    from src.ingestion.layout import column_aware_text, fix_drop_caps, ocr_available, ocr_page, remove_running_headers

    use_ocr = get_settings().ocr != "off" and ocr_available()
    filename = os.path.basename(path)
    texts, tables, ocr_pages = [], [], set()
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            if page.chars:
                text = fix_drop_caps(column_aware_text(page))
            elif use_ocr and page.images:  # scanned page: no text layer, but an image to read
                text = ocr_page(page)
                ocr_pages.add(page_number)
            else:
                text = ""  # blank page (or a scan with OCR unavailable)
            texts.append(text)
            table_blocks = []
            for table in page.extract_tables():
                rows = [" | ".join(cell or "" for cell in row) for row in table]
                table_blocks.append("\n".join(rows))
            tables.append(table_blocks)

    texts = remove_running_headers(texts)
    docs: List[Document] = []
    for page_number, (text, table_blocks) in enumerate(zip(texts, tables), start=1):
        full_text = text.strip()
        if table_blocks:
            full_text += "\n\n[TABLE]\n" + "\n\n[TABLE]\n".join(table_blocks)
        if full_text.strip():
            meta = {"source": filename, "page": page_number, "doc_id": f"{filename}#p{page_number}"}
            if page_number in ocr_pages:
                meta["ocr"] = True
            docs.append(Document(text=full_text, metadata=meta))
    if ocr_pages:
        logger.info("OCR used for %s pages %s", filename, sorted(ocr_pages))
    return docs


def load_pdf_directory(directory: str | None = None) -> List[Document]:
    """Load every .pdf file in a directory (sorted for determinism).

    Defaults to the PDF_DIR setting (src/data/pdfs unless overridden in .env).
    """
    if directory is None:
        from src.config import get_settings

        directory = str(get_settings().pdf_path)
    all_docs: List[Document] = []
    with log_step(logger, "load", directory=directory) as out:
        files = [f for f in sorted(os.listdir(directory)) if f.lower().endswith(".pdf")]
        skipped = 0
        for fname in files:
            try:
                all_docs.extend(load_pdf(os.path.join(directory, fname)))
            except Exception as e:  # one bad file must not stop the whole load
                skipped += 1
                logger.warning("skipping unreadable PDF %s (%s: %s)", fname, type(e).__name__, e)
        out.update(files=len(files) - skipped, skipped=skipped, pages=len(all_docs))
    return all_docs
