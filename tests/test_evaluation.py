from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.metrics import (
    citation_correctness,
    cited_metrics,
    completeness,
    graded_relevance,
    mrr,
    ndcg_at_k,
    numeric_answer_correctness,
    precision_at_k,
    recall_at_k,
)
from src.schemas import DocumentChunk, RetrievedChunk


def _used(cids: list[str]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=DocumentChunk(chunk_id=cid, text="t", page=1, section="", chunk_type="narrative")
        )
        for cid in cids
    ]


def test_cited_metrics() -> None:
    gold = {"c", "z"}
    assert cited_metrics(_used(["a", "b", "c"]), gold) == {
        "cited_recall": 0.5,
        "cited_precision": 0.3333,
    }
    assert cited_metrics([], gold) == {"cited_recall": 0.0, "cited_precision": 0.0}
    assert cited_metrics(_used(["x"]), gold) == {"cited_recall": 0.0, "cited_precision": 0.0}
    assert cited_metrics(_used(["c", "z"]), gold) == {"cited_recall": 1.0, "cited_precision": 1.0}
    assert cited_metrics(_used(["a"]), None) == {"cited_recall": 1.0, "cited_precision": 1.0}


def test_retrieval_metric_formulas() -> None:
    ranked = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c", "z"}
    assert recall_at_k(ranked, relevant, 3) == pytest.approx(2 / 3)
    assert precision_at_k(ranked, relevant, 3) == pytest.approx(2 / 3)
    assert mrr(ranked, relevant) == pytest.approx(1.0)
    assert mrr(["x", "a", "b"], relevant) == pytest.approx(0.5)
    gains = {"a": 1.0, "c": 1.0, "d": 0.5}
    assert ndcg_at_k(ranked, gains, 3) > 0


def test_numeric_answer_correctness() -> None:
    result = numeric_answer_correctness(
        "OpenHands costs $6.38 in 362.41 seconds", ["6.38", "362.41"]
    )
    assert result == {"6.38": True, "362.41": True}
    result = numeric_answer_correctness("costs $6.38", ["6.38", "999"])
    assert result == {"6.38": True, "999": False}


def test_citation_correctness() -> None:
    chunk = DocumentChunk(
        chunk_id="p005-0002", text="...", page=5, section="4.1", chunk_type="result"
    )
    evidence = [RetrievedChunk(chunk=chunk, hybrid_score=1.0)]
    good = citation_correctness("OpenHands costs $6.38 [Page 5, Section 4.1]", evidence)
    assert good["has_citation"] and good["citation_page_matches_evidence"]
    bad = citation_correctness("OpenHands costs $6.38", evidence)
    assert not bad["has_citation"]


def test_completeness() -> None:
    assert completeness("DevAI has 55 tasks and 365 requirements", ["55", "365", "DevAI"]) == 1.0
    assert completeness("DevAI has 55 tasks", ["55", "365", "DevAI"]) == pytest.approx(2 / 3)


def test_graded_relevance() -> None:
    chunk = DocumentChunk(
        chunk_id="x",
        text="DevAI has 55 tasks and 365 requirements",
        page=1,
        section="s",
        chunk_type="n",
    )
    assert graded_relevance(chunk.text, ["55", "365", "xyz"]) == pytest.approx(2 / 3)


@pytest.mark.regression
def test_gold_evidence_map_exists_and_populated() -> None:
    evidra_root = Path(__file__).resolve().parent.parent
    gold_path = evidra_root / "data" / "processed" / "gold_evidence.json"
    if not gold_path.exists():
        pytest.skip("Run `python scripts/build_index.py` first.")
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    questions = json.loads(
        (evidra_root / "src" / "evaluation" / "questions.json").read_text(encoding="utf-8")
    )["questions"]
    assert len(questions) == 18
    for q in questions:
        hits = gold.get(str(q["id"]), [])
        assert hits, f"Q{q['id']} has no gold evidence chunks"


@pytest.mark.regression
def test_all_key_values_present_in_raw_chunks() -> None:
    evidra_root = Path(__file__).resolve().parent.parent
    chunks_path = evidra_root / "data" / "processed" / "chunks.json"
    if not chunks_path.exists():
        pytest.skip("Run `python scripts/build_index.py` first.")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    full_text = " ".join(c["text"].lower() for c in chunks)
    key_values = [
        "6.38",
        "362.41",
        "1.19",
        "44.80",
        "97.72",
        "97.64",
        "90.44",
        "60.38",
        "65.03",
        "75.95",
        "82.24",
        "86.06",
        "87.70",
        "85.52",
        "23.77",
        "30.58",
        "118.43",
        "86.5",
        "1080p",
        "cn9o",
        "svm",
        "lstm",
        "metagpt",
        "gpt-pilot",
        "openhands",
    ]
    missing = [v for v in key_values if v not in full_text]
    assert not missing, f"missing values in corpus: {missing}"
