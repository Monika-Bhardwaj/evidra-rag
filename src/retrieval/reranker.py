from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from src.logging_utils import get_logger
from src.schemas import RetrievedChunk

logger = get_logger(__name__)


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        ...

    def is_used(self) -> bool:
        return True


class NoopReranker(Reranker):
    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        ranked = sorted(candidates, key=lambda c: c.hybrid_score, reverse=True)
        return ranked[:top_k]

    def is_used(self) -> bool:
        return False


class ScoreReranker(Reranker):
    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        weighted = [
            (c, 0.8 * c.hybrid_score + 0.2 * c.dense_score) for c in candidates
        ]
        weighted.sort(key=lambda t: t[1], reverse=True)
        for c, _ in weighted:
            c.rerank_score = c.hybrid_score
        return [c for c, _ in weighted[:top_k]]


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model: Optional["CrossEncoder"] = None
        self._device = device

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder %s ...", self.model_name)
            self._model = CrossEncoder(self.model_name, device=self._device)

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        if not candidates:
            return []
        try:
            self._ensure_model()
            pairs = [(query, c.chunk.text) for c in candidates]
            scores = list(self._model.predict(pairs, show_progress_bar=False))
            for c, score in zip(candidates, scores):
                c.rerank_score = float(score)
            candidates = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
        except Exception as exc:
            logger.warning("CrossEncoder reranking failed (%s); falling back to hybrid ranking.", exc)
            candidates = sorted(candidates, key=lambda c: c.hybrid_score, reverse=True)
        return candidates[:top_k]


def build_reranker(enabled: bool, model_name: str, device: str = "cpu") -> Reranker:
    if not enabled:
        return NoopReranker()
    try:
        return CrossEncoderReranker(model_name=model_name, device=device)
    except Exception as exc:
        logger.warning("Could not initialise reranker (%s); using hybrid ranking fallback.", exc)
        return NoopReranker()