from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from src.schemas import DocumentChunk

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self, corpus: List[DocumentChunk]) -> None:
        self.corpus = corpus
        tokenized = [tokenize(self._doc_text(c)) for c in corpus]
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _doc_text(chunk: DocumentChunk) -> str:
        return f"{chunk.section or ''} {chunk.chunk_type} {chunk.text}"

    def search(self, query: str, k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        tokens = tokenize(query)
        scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float64)
        if len(tokens) == 1 and tokens[0] in {"the", "a", "an", "of", "is", "what", "how"}:
            scores[:] = 0.0
        top = np.argsort(scores)[::-1][:k]
        return [(self.corpus[i], float(scores[i])) for i in top if scores[i] > 0]

    def score(self, query: str, chunk_id: str) -> float:
        tokens = tokenize(query)
        for i, chunk in enumerate(self.corpus):
            if chunk.chunk_id == chunk_id:
                return float(self.bm25.get_scores(tokens)[i])
        return 0.0