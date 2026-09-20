from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.embeddings import Embedder
from src.schemas import DocumentChunk


class FakeEmbedder(Embedder):
    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    def encode(self, texts, batch_size=32) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts])

    def _vector(self, text: str) -> np.ndarray:
        import hashlib

        vec = np.zeros(self._dim, dtype=np.float32)
        for token in text.lower().split():
            h = hashlib.md5(token.encode()).digest()
            idx = int.from_bytes(h[:2], "big") % self._dim
            vec[idx] += 1.0
        return vec / max(np.linalg.norm(vec), 1e-6)

    def dim(self) -> int:
        return self._dim

    def name(self) -> str:
        return "fake"


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def small_corpus() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="p003-0000",
            text="The DevAI dataset contains 55 real-world tasks, 365 hierarchical requirements and 125 preferences.",
            page=2,
            section="2.2 DevAI Dataset",
            chunk_type="definition",
        ),
        DocumentChunk(
            chunk_id="p003-0001",
            text="Agent-as-a-Judge saves 97.72% of evaluation time and 97.64% of cost.",
            page=3,
            section="3.1 Savings",
            chunk_type="result",
        ),
        DocumentChunk(
            chunk_id="p005-0002",
            text="OpenHands average cost is $6.38 and average time is 362.41 seconds per task.",
            page=5,
            section="4.1 Benchmark",
            chunk_type="result",
        ),
        DocumentChunk(
            chunk_id="p005-0003",
            text="MetaGPT is the most cost-efficient framework with an average cost of $1.19.",
            page=5,
            section="4.1 Benchmark",
            chunk_type="cost-analysis",
        ),
        DocumentChunk(
            chunk_id="p009-0004",
            text="In the black-box setting Agent-as-a-Judge reaches 90.44% alignment for OpenHands.",
            page=9,
            section="4.2 Judging",
            chunk_type="result",
        ),
    ]