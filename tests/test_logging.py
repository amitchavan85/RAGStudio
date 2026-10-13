import logging
import os

import pytest
from helpers import HashEmbedder, PDF_DIR

from src.chunking import get_chunker
from src.ingestion.loader import load_pdf_directory
from src.logging_config import log_step, new_request_id, setup_logging
from src.vectorstores.chroma_store import ChromaStore



@pytest.fixture
def restore_root_logger():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for h in list(root.handlers):
        if h not in handlers:
            root.removeHandler(h)
            h.close()
    root.setLevel(level)


def test_setup_logging_writes_text_file_with_request_id(tmp_path, restore_root_logger):
    setup_logging(level="INFO", log_dir=str(tmp_path))
    rid = new_request_id("abc123")
    logging.getLogger("src.test").info("hello pipeline")
    for h in logging.getLogger().handlers:
        h.flush()
    text = (tmp_path / "rag.log").read_text(encoding="utf-8")
    assert "hello pipeline" in text
    assert f"[req={rid}]" in text
    assert "INFO" in text and "src.test" in text


def test_setup_logging_is_idempotent(tmp_path, restore_root_logger):
    before = len(logging.getLogger().handlers)
    setup_logging(level="INFO", log_dir=str(tmp_path))
    setup_logging(level="INFO", log_dir=str(tmp_path))
    assert len(logging.getLogger().handlers) == before + 2  # console + file, not doubled


def test_log_step_success_and_extra_fields(caplog):
    logger = logging.getLogger("t.step")
    with caplog.at_level(logging.INFO, logger="t.step"):
        with log_step(logger, "demo", a=1) as out:
            out["result"] = 7
    msg = caplog.records[-1].getMessage()
    assert msg.startswith("demo done") and "a=1" in msg and "result=7" in msg and "ms=" in msg


def test_log_step_logs_error_and_reraises(caplog):
    logger = logging.getLogger("t.step")
    with caplog.at_level(logging.INFO, logger="t.step"):
        with pytest.raises(RuntimeError):
            with log_step(logger, "boom"):
                raise RuntimeError("bad")
    assert caplog.records[-1].levelno == logging.ERROR
    assert "boom failed" in caplog.records[-1].getMessage()


def test_pipeline_steps_are_logged_without_document_text(tmp_path, caplog):
    emb = HashEmbedder()
    with caplog.at_level(logging.INFO):
        docs = load_pdf_directory(PDF_DIR)
        chunks = get_chunker("recursive", 400, 40).chunk_documents(docs)
        store = ChromaStore("log_test", emb.dimension, data_dir=str(tmp_path))
        store.add(chunks, emb.embed_documents([c.text for c in chunks]))
        store.search(emb.embed_query("India GDP"), k=3)
    msgs = [r.getMessage() for r in caplog.records]
    for step in ("load done", "chunk done", "store.add done", "retrieve done"):
        assert any(m.startswith(step) for m in msgs), step
    assert any("strategy=RecursiveChunker" in m for m in msgs)
    assert any("hits=3" in m and "top_score=" in m for m in msgs)
    # privacy: no chunk text may leak into INFO logs
    sample = chunks[0].text[:40]
    assert not any(sample in m for m in msgs)


def test_agentic_fallback_logs_warning(caplog):
    chunker = get_chunker("agentic", llm_fn=lambda p: "not json")
    with caplog.at_level(logging.WARNING):
        chunker.split("Some text.")
    assert any(r.levelno == logging.WARNING and "falling back" in r.getMessage() for r in caplog.records)


def test_relative_log_dir_resolves_to_project_root(tmp_path, monkeypatch, restore_root_logger):
    from src.config import PROJECT_ROOT

    monkeypatch.chdir(tmp_path)  # run from a different folder
    setup_logging(level="INFO", log_dir="logs_test_tmp")
    logging.getLogger("src.test").info("where am I")
    for h in logging.getLogger().handlers:
        h.flush()
    target = PROJECT_ROOT / "logs_test_tmp" / "rag.log"
    try:
        assert target.exists() and "where am I" in target.read_text(encoding="utf-8")
        assert not (tmp_path / "logs_test_tmp").exists()
    finally:
        for h in list(logging.getLogger().handlers):
            if getattr(h, "_rag_handler", False):
                logging.getLogger().removeHandler(h)
                h.close()
        import shutil

        shutil.rmtree(PROJECT_ROOT / "logs_test_tmp", ignore_errors=True)


def test_single_pdf_load_is_logged(caplog):
    from src.ingestion.loader import load_pdf

    with caplog.at_level(logging.INFO):
        load_pdf(os.path.join(PDF_DIR, "usa_gdp_report.pdf"))
    assert any(r.getMessage().startswith("load_pdf done file=usa_gdp_report.pdf pages=") for r in caplog.records)
