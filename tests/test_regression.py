from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Known-weak retrieval cases frozen from the tuning session (docs/engineering_audit.md §7).
# Each is sparse-strong / dense-weak: the gold chunk survives weighted fusion (alpha=0.5)
# but was lost or badly ranked under dense-only or RRF. If the parser, chunker, or fusion
# regresses these, the golden summary degrades — these fixtures make that loud.
WEAK_CASES = [
    (5, ["p003-0008"], {3}, "Q5: p3 97.72%/97.64% savings"),
    (6, ["p011-0045"], {11}, "Q6: p11 cost analysis $30.58/118.43 min"),
    (16, ["p005-0015"], {5}, "Q16: p5 R1 1080p requirements/save in results/"),
]


@pytest.mark.regression
def test_weak_case_gold_chunks_are_preserved() -> None:
    gold_path = ROOT / "data" / "processed" / "gold_evidence.json"
    chunks_path = ROOT / "data" / "processed" / "chunks.json"
    if not gold_path.exists() or not chunks_path.exists():
        pytest.skip("Run `python scripts/build_index.py` first.")
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    by_id = {c["chunk_id"]: c for c in chunks}

    for qid, expected_ids, pages, label in WEAK_CASES:
        gold_ids = {h["chunk_id"] for h in gold.get(str(qid), [])}
        missing_gold = [i for i in expected_ids if i not in gold_ids]
        assert not missing_gold, f"{label}: gold map lost {missing_gold}"
        for cid in expected_ids:
            chunk = by_id.get(cid)
            assert chunk is not None, f"{label}: chunk {cid} missing from chunks.json"
            assert chunk["page"] in pages, (
                f"{label}: chunk {cid} moved to page {chunk['page']}, expected {pages}"
            )


@pytest.mark.regression
def test_weak_case_gold_chunks_keep_ranked_in_retrieval() -> None:
    from src.config import get_settings
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.embeddings import build_embedder
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.vector_store import FAISSVectorStore

    settings = get_settings()
    if not (settings.index_dir_path / "chunks.json").exists():
        pytest.skip("Run `python scripts/build_index.py` first.")
    embedder = build_embedder(settings)
    store = FAISSVectorStore.load(settings.index_dir_path)
    if store.size() == 0:
        pytest.skip("Index artifacts present but empty; rebuild the index.")
    hybrid = HybridRetriever(store, BM25Retriever(store.chunks()), embedder)

    questions = {
        q["id"]: q
        for q in json.loads(
            (ROOT / "src" / "evaluation" / "questions.json").read_text(encoding="utf-8")
        )["questions"]
    }
    top_k = max(settings.rerank_top_k, settings.final_top_k)

    for qid, expected_ids, _, label in WEAK_CASES:
        top, sufficient, _ = hybrid.retrieve(
            query=questions[qid]["question"],
            dense_top_k=settings.dense_top_k,
            bm25_top_k=settings.bm25_top_k,
            top_k=top_k,
            alpha=settings.hybrid_alpha,
            method=settings.hybrid_method,
            min_similarity=settings.min_similarity,
        )
        ranked_ids = {c.chunk.chunk_id for c in top}
        missing = [i for i in expected_ids if i not in ranked_ids]
        assert sufficient, f"{label}: sufficiency gate failed"
        assert not missing, (
            f"{label}: gold chunks {missing} dropped from top-{top_k} retrieval ({ranked_ids})"
        )
