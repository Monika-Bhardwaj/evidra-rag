from __future__ import annotations

from src.retrieval.bm25 import BM25Retriever, tokenize
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import NoopReranker, ScoreReranker
from src.retrieval.vector_store import FAISSVectorStore, InMemoryVectorStore
from src.schemas import DocumentChunk, RetrievedChunk


def _build_retriever(corpus, embedder) -> HybridRetriever:
    store = InMemoryVectorStore()
    vectors = embedder.encode([c.text for c in corpus])
    store.add(corpus, vectors)
    bm25 = BM25Retriever(corpus)
    return HybridRetriever(store, bm25, embedder)


def test_bm25_finds_exact_terms(small_corpus) -> None:
    bm25 = BM25Retriever(small_corpus)
    hits = bm25.search("OpenHands average cost", k=3)
    assert hits
    top = hits[0][0]
    assert "OpenHands" in top.text


def test_tokenize() -> None:
    assert tokenize(" OpenHands average cost: $6.38! ") == [
        "openhands",
        "average",
        "cost",
        "6",
        "38",
    ]


def test_dense_retrieval_ranking(small_corpus, fake_embedder) -> None:
    hybrid = _build_retriever(small_corpus, fake_embedder)
    top, sufficient, _ = hybrid.retrieve(
        "What is OpenHands average cost and time?",
        top_k=3,
        alpha=1.0,
        method="weighted",
    )
    assert top
    assert all("OpenHands" in c.chunk.text for c in top[:1])


def test_hybrid_weighted_boost_for_numbers(small_corpus, fake_embedder) -> None:
    hybrid = _build_retriever(small_corpus, fake_embedder)
    top, sufficient, _ = hybrid.retrieve(
        "What is the average cost?",
        top_k=2,
        alpha=0.7,
        method="weighted",
        type_boost={"cost-analysis": 0.25, "result": 0.1},
    )
    assert top
    cost_chunks = [c for c in top if c.chunk.chunk_type == "cost-analysis"]
    assert cost_chunks


def test_rrf_fusion_orders_relevant_first(small_corpus, fake_embedder) -> None:
    hybrid = _build_retriever(small_corpus, fake_embedder)
    top, _, _ = hybrid.retrieve(
        "Agent-as-a-Judge saves time and cost percentage",
        top_k=4,
        alpha=0.5,
        method="rrf",
    )
    assert top
    assert any("97.72%" in c.chunk.text for c in top[:3])


def test_min_similarity_gate_blocks_unrelated(small_corpus, fake_embedder) -> None:
    hybrid = _build_retriever(small_corpus, fake_embedder)
    top, sufficient, _ = hybrid.retrieve(
        "quantum computing zephyr telemetry",
        top_k=2,
        alpha=0.7,
        min_similarity=0.99,
    )
    assert sufficient is False


def test_low_dense_cosine_fails_closed_despite_sparse_overlap() -> None:
    chunk = DocumentChunk(
        chunk_id="p1",
        text="quantum zephyr telemetry computing systems",
        page=1,
        section="s",
        chunk_type="narrative",
    )
    fused = [RetrievedChunk(chunk=chunk, dense_score=0.15, sparse_score=5.0, hybrid_score=0.4)]
    # Sparse token overlap alone must never mark a query sufficient (ABS-6 regression).
    assert HybridRetriever._sufficient(fused, 0.5) is False
    assert HybridRetriever._sufficient(fused, 0.10) is True
    assert HybridRetriever._sufficient([], 0.10) is False


def test_sufficiency_must_use_full_candidate_set_not_topk_slice() -> None:
    # A high-dense chunk can rank outside the top-k slice after hybrid fusion.
    # Evaluating sufficiency on only the top-k would flip the verdict between the
    # cache-hit and cache-miss paths; the gate runs on the full candidate set and
    # the pipeline persists that verdict with the cached entry.
    weak = DocumentChunk(
        chunk_id="p1",
        text="telemetry",
        page=1,
        section="s",
        chunk_type="narrative",
    )
    strong = DocumentChunk(
        chunk_id="p2",
        text="quantum computing telemetry systems",
        page=1,
        section="s",
        chunk_type="narrative",
    )
    full = [
        RetrievedChunk(chunk=weak, dense_score=0.60, sparse_score=9.0, hybrid_score=0.90),
        RetrievedChunk(chunk=strong, dense_score=0.71, sparse_score=2.0, hybrid_score=0.50),
    ]
    topk_slice = [full[0]]
    assert HybridRetriever._sufficient(full, 0.7) is True
    assert HybridRetriever._sufficient(topk_slice, 0.7) is False


def test_sufficiency_gate_uses_raw_query_cosine_not_expanded() -> None:
    # ABS-6 regression: the expansion-averaged cosine can clear min_similarity
    # for an off-topic query while the raw-question cosine stays far below it.
    chunk = DocumentChunk(
        chunk_id="p1",
        text="results/processing time.txt.",
        page=1,
        section="s",
        chunk_type="table",
    )
    fused = [RetrievedChunk(chunk=chunk, dense_score=0.45, sparse_score=3.0, hybrid_score=0.6)]
    raw_dense = {"p1": 0.19}
    assert HybridRetriever._sufficient(fused, 0.35) is True
    assert HybridRetriever._sufficient(fused, 0.35, raw_dense) is False


def test_chunk_type_filter(small_corpus, fake_embedder) -> None:
    hybrid = _build_retriever(small_corpus, fake_embedder)
    top, _, _ = hybrid.retrieve(
        "how much did evaluation cost",
        top_k=3,
        chunk_types=["cost-analysis", "result"],
    )
    assert all(c.chunk.chunk_type in ("cost-analysis", "result") for c in top)


def test_noop_reranker_keeps_hybrid_order(small_corpus, fake_embedder) -> None:
    hybrid = _build_retriever(small_corpus, fake_embedder)
    top, _, _ = hybrid.retrieve("OpenHands average cost", top_k=4)
    reranker = NoopReranker()
    reranked = reranker.rerank("OpenHands average cost", top, 3)
    assert len(reranked) == 3
    assert not reranker.is_used()


def test_score_reranker_assigns_scores(small_corpus, fake_embedder) -> None:
    hybrid = _build_retriever(small_corpus, fake_embedder)
    top, _, _ = hybrid.retrieve("OpenHands average cost", top_k=4)
    reranker = ScoreReranker()
    reranked = reranker.rerank("OpenHands average cost", top, 2)
    assert len(reranked) == 2


def test_build_reranker_strategy() -> None:
    from src.retrieval.reranker import build_reranker

    noop = build_reranker(enabled=False, model_name="x")
    assert isinstance(noop, NoopReranker)
    assert not noop.is_used()

    score = build_reranker(enabled=True, model_name="x", kind="score")
    assert isinstance(score, ScoreReranker)
    assert score.is_used()


def test_vector_store_persist_roundtrip(tmp_path, small_corpus, fake_embedder) -> None:
    store = FAISSVectorStore()
    vectors = fake_embedder.encode([c.text for c in small_corpus])
    store.add(small_corpus, vectors)
    store.save(tmp_path)
    loaded = FAISSVectorStore.load(tmp_path)
    assert loaded.size() == 5
    hits = loaded.search(vectors[0].reshape(1, -1), 3)
    assert hits and hits[0][0] == 0
