from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.logging_utils import get_logger
from src.schemas import DocumentChunk

logger = get_logger(__name__)


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: List[DocumentChunk], vectors: np.ndarray) -> None:
        ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[int, float]]:
        ...

    @abstractmethod
    def chunks(self) -> List[DocumentChunk]:
        ...

    @abstractmethod
    def save(self, index_dir: Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        ...

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
            store._chunks = [DocumentChunk.from_dict(d) for d in json.loads(chunks_path.read_text(encoding="utf-8"))]
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
            store._chunks = [DocumentChunk.from_dict(d) for d in json.loads(chunks_path.read_text(encoding="utf-8"))]
        vectors_path = index_dir / "vectors.npy"
        if vectors_path.exists():
            store._vectors = np.load(vectors_path)
        return store


class ChromaVectorStore(VectorStore):
    def __init__(self, path: Optional[str] = None) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=str(path) if path else None)
        self._collection = self._client.get_or_create_collection("devai_chunks")
        self._chunks: List[DocumentChunk] = []

    def add(self, chunks: List[DocumentChunk], vectors: np.ndarray) -> None:
        self._chunks = list(chunks)
        ids = [c.chunk_id for c in chunks]
        metadatas = [{"page": c.page, "section": c.section, "chunk_type": c.chunk_type} for c in chunks]
        docs = [c.text for c in chunks]
        self._collection.upsert(ids=ids, embeddings=vectors.tolist(), metadatas=metadatas, documents=docs)

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[int, float]]:
        result = self._collection.query(query_embeddings=[query_vector.tolist()], n_results=max(1, k))
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        index_map = {c.chunk_id: i for i, c in enumerate(self._chunks)}
        hits = []
        for cid, dist in zip(ids, distances):
            if cid in index_map:
                hits.append((index_map[cid], 1.0 - float(dist)))
        return hits

    def chunks(self) -> List[DocumentChunk]:
        if self._chunks:
            return self._chunks
        result = self._collection.get(include=["metadatas"])
        self._chunks = []
        for cid, meta in zip(result["ids"], result["metadatas"]):
            self._chunks.append(
                DocumentChunk(
                    chunk_id=cid,
                    text="",
                    page=int(meta.get("page", 0)),
                    section=meta.get("section", ""),
                    chunk_type=meta.get("chunk_type", "narrative"),
                )
            )
        return self._chunks

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self._chunks], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: Path) -> "ChromaVectorStore":
        path = str(index_dir / "chroma")
        store = cls(path=path)
        if (index_dir / "chunks.json").exists():
            store._chunks = [DocumentChunk.from_dict(d) for d in json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))]
        return store


def build_vector_store(backend: str = "faiss", index_dir: Optional[Path] = None) -> VectorStore:
    if backend == "chroma":
        store = ChromaVectorStore(path=str(index_dir / "chroma") if index_dir else None)
        return store
    if backend == "memory":
        return InMemoryVectorStore()
    return FAISSVectorStore()


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms