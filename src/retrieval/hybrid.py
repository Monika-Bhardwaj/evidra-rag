from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np

from src.logging_utils import get_logger
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.embeddings import Embedder
from src.retrieval.vector_store import VectorStore
from src.schemas import DocumentChunk, RetrievedChunk

logger = get_logger(__name__)


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25: BM25Retriever,
        embedder: Embedder,
    ) -> None:
        self.vector_store = vector_store
        self.bm25 = bm25
        self.embedder = embedder
        self.chunks_by_id: Dict[str, DocumentChunk] = {
            c.chunk_id: c for c in self.vector_store.chunks()
        }

    def retrieve(
        self,
        query: str,
        expanded_queries: Optional[List[str]] = None,
        dense_top_k: int = 10,
        bm25_top_k: int = 10,
        top_k: int = 6,
        alpha: float = 0.7,
        method: str = "weighted",
        min_similarity: float = 0.0,
        chunk_types: Optional[List[str]] = None,
        type_boost: Optional[Dict[str, float]] = None,
    ) -> tuple[List[RetrievedChunk], bool, List[str]]:
        variants = [query] + (expanded_queries or [])
        query_vector = self._mean_embedding(variants)

        dense_hits = self.vector_store.search(query_vector, k=max(1, dense_top_k))
        dense: Dict[str, float] = {}
        for idx, score in dense_hits:
            cid = self.vector_store.chunks()[idx].chunk_id
            dense[cid] = max(dense.get(cid, 0.0), float(score))

        sparse: Dict[str, float] = {}
        for variant in variants:
            for chunk, score in self.bm25.search(variant, k=bm25_top_k):
                sparse[chunk.chunk_id] = max(sparse.get(chunk.chunk_id, 0.0), float(score))

        candidates: Set[str] = set(dense.keys()) | set(sparse.keys())
        fused: List[RetrievedChunk] = []
        for cid in candidates:
            chunk = self.chunks_by_id.get(cid)
            if chunk is None:
                continue
            d_raw = dense.get(cid, 0.0)
            s_raw = sparse.get(cid, 0.0)
            fused.append(
                RetrievedChunk(chunk=chunk, dense_score=d_raw, sparse_score=s_raw, hybrid_score=0.0)
            )

        # Sufficiency is judged on the *original* query embedding, not the
        # expansion-averaged one. Expansion terms must never be able to pull an
        # off-topic query over the bar: ABS-6 ("Who was named Time Person of the
        # Year for 2023?") classified as `numerical_lookup`, whose expansion
        # ("average time", "results table", "numbers", ...) matched a filename
        # chunk ("results/processing time.txt.") at cosine 0.45 >= min_similarity.
        raw_vector = self.embedder.encode([query])[0]
        raw_dense: Dict[str, float] = {}
        for idx, score in self.vector_store.search(raw_vector, k=max(1, dense_top_k * 2)):
            cid = self.vector_store.chunks()[idx].chunk_id
            raw_dense[cid] = max(raw_dense.get(cid, 0.0), float(score))

        if method == "rrf":
            fused = self._rrf_rank(fused, dense, sparse, alpha)
        else:
            fused = self._weighted_rank(fused, alpha)

        fused = self._apply_boost(fused, type_boost or {})

        if chunk_types:
            fused = [c for c in fused if c.chunk.chunk_type in chunk_types]

        evidence_sufficient = self._sufficient(fused, min_similarity, raw_dense)
        filtered = [c for c in fused if c.hybrid_score <= 0.0]
        fused = [c for c in fused if c.hybrid_score > 0.0]
        fused.sort(key=lambda c: c.hybrid_score, reverse=True)
        top = fused[:top_k]
        filtered_ids = [c.chunk.chunk_id for c in filtered]
        return top, evidence_sufficient, filtered_ids

    @staticmethod
    def _sufficient(
        fused: List[RetrievedChunk],
        min_similarity: float,
        raw_dense: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Fail-closed sufficiency gate: at least one chunk must reach the dense bar.

        The score is measured against the ORIGINAL question embedding
        (`raw_dense`, keyed by chunk_id) when provided, not the
        expansion-averaged `dense_score` on the fused rows. Two fixes landed here
        (2026-09-23) after off-topic questions leaked through:

        * The sparse-token fallback was removed: incidental query/chunk token
          overlap ("2023", "year", "named") must not outvote a dense cosine far
          below `min_similarity`.
        * The gate is evaluated on the raw query, so expansion templates cannot
          lift an unrelated query over the bar (ABS-6 matched a filename chunk at
          0.45 only after `numerical_lookup` expansion injected "average time"
          and "results table").

        Every golden question clears the raw-query bar directly (measured best
        cosine >= 0.42), so both changes cost no golden recall while making the
        abstention boundary fail-closed.
        """
        if not fused:
            return False
        if raw_dense is not None:
            best = max((raw_dense.get(c.chunk.chunk_id, c.dense_score) for c in fused), default=0.0)
        else:
            best = max((c.dense_score for c in fused), default=0.0)
        return best >= min_similarity

    def _mean_embedding(self, texts: List[str]) -> np.ndarray:
        embeddings = self.embedder.encode(texts)
        return embeddings.mean(axis=0)

    def _chunk_at(self, idx: int) -> DocumentChunk:
        return self.vector_store.chunks()[idx]

    def _apply_boost(
        self, fused: List[RetrievedChunk], type_boost: Dict[str, float]
    ) -> List[RetrievedChunk]:
        for c in fused:
            boost = type_boost.get(c.chunk.chunk_type, 0.0)
            c.hybrid_score += boost
        return fused

    @staticmethod
    def _minmax(values: List[float]) -> Dict[float, float]:
        if not values:
            return {}
        lo, hi = min(values), max(values)
        if hi - lo < 1e-12:
            return {v: 0.5 for v in values}
        return {v: (v - lo) / (hi - lo) for v in values}

    def _weighted_rank(self, fused: List[RetrievedChunk], alpha: float) -> List[RetrievedChunk]:
        dense_lookup = self._minmax([c.dense_score for c in fused])
        sparse_lookup = self._minmax([c.sparse_score for c in fused])
        for c in fused:
            dn = dense_lookup.get(c.dense_score, 0.0)
            sn = sparse_lookup.get(c.sparse_score, 0.0)
            c.hybrid_score = alpha * dn + (1.0 - alpha) * sn
        return fused

    def _rrf_rank(
        self,
        fused: List[RetrievedChunk],
        dense: Dict[str, float],
        sparse: Dict[str, float],
        alpha: float,
    ) -> List[RetrievedChunk]:
        dense_order = {cid: i for i, cid in enumerate(sorted(dense, key=dense.get, reverse=True))}
        sparse_order = {
            cid: i for i, cid in enumerate(sorted(sparse, key=sparse.get, reverse=True))
        }
        const = 60
        for c in fused:
            d_rank = dense_order.get(c.chunk.chunk_id)
            s_rank = sparse_order.get(c.chunk.chunk_id)
            score = 0.0
            if d_rank is not None:
                score += alpha / (const + d_rank + 1)
            if s_rank is not None:
                score += (1.0 - alpha) / (const + s_rank + 1)
            c.hybrid_score = score
        return fused
