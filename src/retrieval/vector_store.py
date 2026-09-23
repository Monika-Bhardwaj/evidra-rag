from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.logging_utils import get_logger
from src.schemas import DocumentChunk

logger = get_logger(__name__)


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: List[DocumentChunk], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[int, float]]: ...

    @abstractmethod
    def chunks(self) -> List[DocumentChunk]: ...

    @abstractmethod
    def save(self, index_dir: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, index_dir: Path) -> "VectorStore": ...

    def size(self) -> int:
        return len(self.chunks())


class FAISSVectorStore(VectorStore):
    def __init__(self) -> None:
        self._chunks: List[DocumentChunk] = []
        self._index = None

    def add(self, chunks: List[DocumentChunk], vectors: np.ndarray) -> None:
        import faiss

        self._chunks = list(chunks)
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index = faiss.IndexFlatIP(arr.shape[1])
        self._index.add(_l2_normalize(arr))

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[int, float]]:
        if self._index is None:
            return []
        q = _l2_normalize(np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1))
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(q, k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]

    def chunks(self) -> List[DocumentChunk]:
        return self._chunks

    def save(self, index_dir: Path) -> None:
        import faiss

        index_dir.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(index_dir / "vectors.faiss"))
        (index_dir / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self._chunks], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: Path) -> "FAISSVectorStore":
        import faiss

        store = cls()
        chunks_path = index_dir / "chunks.json"
        if chunks_path.exists():
            store._chunks = [
                DocumentChunk.from_dict(d)
                for d in json.loads(chunks_path.read_text(encoding="utf-8"))
            ]
        index_path = index_dir / "vectors.faiss"
        if index_path.exists():
            store._index = faiss.read_index(str(index_path))
        return store


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._chunks: List[DocumentChunk] = []
        self._vectors: Optional[np.ndarray] = None

    def add(self, chunks: List[DocumentChunk], vectors: np.ndarray) -> None:
        self._chunks = list(chunks)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._vectors = vectors / np.maximum(norms, 1e-9)

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[int, float]]:
        if self._vectors is None or len(self._vectors) == 0:
            return []
        q = query_vector.reshape(1, -1)
        qn = np.linalg.norm(q)
        qq = q / max(qn, 1e-9)
        scores = self._vectors @ qq.T
        scores = scores.flatten()
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top]

    def chunks(self) -> List[DocumentChunk]:
        return self._chunks

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        if self._vectors is not None:
            np.save(index_dir / "vectors.npy", self._vectors)
        (index_dir / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self._chunks], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: Path) -> "InMemoryVectorStore":
        store = cls()
        chunks_path = index_dir / "chunks.json"
        if chunks_path.exists():
            store._chunks = [
                DocumentChunk.from_dict(d)
                for d in json.loads(chunks_path.read_text(encoding="utf-8"))
            ]
        vectors_path = index_dir / "vectors.npy"
        if vectors_path.exists():
            store._vectors = np.load(vectors_path)
        return store


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms
