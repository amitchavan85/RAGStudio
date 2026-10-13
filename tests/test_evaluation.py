"""Roadmap P5: evaluation dataset, metrics and runner."""
import os
from types import SimpleNamespace

import pytest
from helpers import PDF_DIR, FakeLLM, HashEmbedder

from src.evaluation import (DEFAULT_DATASET, EvalConfig, EvalQuestion, answer_metrics, fact_present,
                            format_table, judge_faithfulness, load_dataset, retrieval_metrics, run_eval)
from src.generation import NOT_FOUND
from src.ingestion.loader import load_pdf


# ---------- the dataset itself must be correct ----------

@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DEFAULT_DATASET)


def test_dataset_loads_with_both_kinds_of_question(dataset):
    assert len(dataset) >= 25
    assert sum(not q.answerable for q in dataset) >= 3
    assert {"table", "lookup", "outlook", "semantic", "ranking", "comparison", "unanswerable"} <= {q.type for q in dataset}


def test_every_expected_fact_really_is_in_its_evidence_pages(dataset):
    """Guards against typos in the ground truth: each fact must be findable in at least one evidence page."""
    pages = {}
    for q in dataset:
        for src, page in {sp for group in q.evidence for sp in group}:
            if (src, page) not in pages:
                docs = {d.metadata["page"]: d.text for d in load_pdf(os.path.join(PDF_DIR, src))}
                assert page in docs, f"{q.id}: {src} has no page {page}"
                pages.update({(src, p): t for p, t in docs.items()})
        evidence_text = " ".join(pages[sp] for group in q.evidence for sp in group)
        for alts in q.facts:
            if q.type == "comparison" and alts == ["germany"]:
                continue  # the conclusion ("Germany is larger") is derived, not quoted
            if q.type in ("semantic", "ranking") and alts in (["india"], ["brazil"], ["germany"]):
                continue  # the country is implied by which report the evidence is in
            assert fact_present(evidence_text, alts), f"{q.id}: {alts} not found in its evidence pages"


# ---------- metrics ----------

def test_fact_matching_respects_number_boundaries():
    assert fact_present("Growth was 2.5% in 2023.", ["2.5"])
    assert not fact_present("Growth was 12.5% in 2023.", ["2.5"])
    assert fact_present("GDP was USD 27,360 billion", ["27360", "27,360"])
    assert not fact_present("Brazil", ["bra"])
    assert fact_present("population AGEING", ["aging", "ageing"])


def hit(src, page):
    return SimpleNamespace(metadata={"source": src, "page": page})


def test_retrieval_metrics_single_and_multi_evidence():
    q = EvalQuestion("q", "?", evidence=[[("a.pdf", 1), ("g.pdf", 1)]], facts=[["x"]])
    m = retrieval_metrics(q, [hit("z.pdf", 1), hit("g.pdf", 1)])
    assert (m["hit@1"], m["hit@3"], m["mrr"]) == (False, True, 0.5)

    cmp_q = EvalQuestion("c", "?", evidence=[[("a.pdf", 1)], [("b.pdf", 1)]], facts=[["x"]])
    only_a = retrieval_metrics(cmp_q, [hit("a.pdf", 1), hit("a.pdf", 1)])
    assert only_a["hit@k"] is False  # comparison needs both documents
    both = retrieval_metrics(cmp_q, [hit("a.pdf", 1), hit("x.pdf", 1), hit("b.pdf", 1)])
    assert both["hit@3"] and not both["hit@1"] and both["ranks"] == [1, 3]


def test_answer_metrics():
    q = EvalQuestion("q", "?", evidence=[[("usa.pdf", 1)]], facts=[["27,360"], ["2.5"]])
    good = SimpleNamespace(found=True, text="GDP 27,360 and growth 2.5% [1].", citations=[{"source": "usa.pdf"}])
    m = answer_metrics(q, good)
    assert m == {"fact_ok": True, "fact_recall": 1.0, "cited_ok": True, "false_nf": False}
    wrong_cite = SimpleNamespace(found=True, text="GDP 27,360 [1].", citations=[{"source": "chn.pdf"}])
    m = answer_metrics(q, wrong_cite)
    assert m["fact_ok"] is False and m["fact_recall"] == 0.5 and m["cited_ok"] is False
    na = EvalQuestion("n", "?", answerable=False)
    assert answer_metrics(na, SimpleNamespace(found=False, text=NOT_FOUND, citations=[])) == {"abstain_ok": True}


def test_judge_parsing():
    ans = SimpleNamespace(text="X [1]", sources=[SimpleNamespace(text="X")])
    assert judge_faithfulness(FakeLLM(reply='{"unsupported": []}'), ans) == {"faithful": True, "unsupported": []}
    assert judge_faithfulness(FakeLLM(reply='Sure: {"unsupported": ["Y"]}'), ans)["faithful"] is False
    assert judge_faithfulness(FakeLLM(reply="no json"), ans)["faithful"] is None


# ---------- runner (fake embedder + fake LLM, no network) ----------

def test_run_eval_compares_configs_and_cleans_up(dataset, monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "real"))
    docs = [os.path.join(PDF_DIR, f) for f in ("usa_gdp_report.pdf", "ind_gdp_report.pdf", "global_gdp_comparison.pdf")]
    qs = [q for q in dataset if q.id in ("usa-gdp-2023", "cmp-ind-usa-2023", "na-mars")]
    configs = [EvalConfig("recursive", "faiss", "hybrid", "gpt-4o-mini"), EvalConfig("recursive", "faiss", "keyword", None),
               EvalConfig("sentence", "faiss", "hybrid", "gpt-4o-mini")]
    llm = FakeLLM(reply="US GDP was 27,360 [1].")
    res = run_eval(qs, configs, docs, embedder=HashEmbedder(), llm=llm, progress=lambda m: None)

    assert [c["label"] for c in res["configs"]] == ["recursive/faiss/hybrid/gpt-4o-mini", "recursive/faiss/keyword/-",
                                                    "sentence/faiss/hybrid/gpt-4o-mini"]
    answered = res["configs"][0]["summary"]
    assert answered["hit@k"] is not None and answered["fact_acc"] is not None and answered["abstain_acc"] == 0.0
    assert "fact_acc" not in res["configs"][1]["summary"]  # retrieval-only config
    usa = next(r for r in res["configs"][0]["questions"] if r["id"] == "usa-gdp-2023")
    assert usa["fact_ok"] is True and usa["hit@k"] is True
    assert os.environ["DATA_DIR"] == str(tmp_path / "real")  # restored
    assert not (tmp_path / "real").exists()  # evaluation never wrote to the real data folder
    assert "hit@1" in format_table(res)
