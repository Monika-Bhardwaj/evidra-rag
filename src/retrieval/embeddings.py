from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.config import Settings
from src.logging_utils import get_logger

logger = get_logger(__name__)


class Embedder(ABC):
    @abstractmethod
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        ...

    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self._dim: Optional[int] = None

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.model.get_sentence_embedding_dimension())
        return self._dim

    def name(self) -> str:
        return self.model_name


class APIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = "", base_url: str = "") -> None:
        from openai import OpenAI

        self.model_name = model
        if api_key:
            self._client = OpenAI(api_key=api_key, base_url=base_url or None)
        else:
            self._client = OpenAI()
        self._dim: Optional[int] = None

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._client.embeddings.create(model=self.model_name, input=batch)
            vectors.extend(item.embedding for item in resp.data)
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-9)

    def dim(self) -> int:
        if self._dim is None:
            probe = self.encode(["probe"])
            self._dim = probe.shape[1]
        return self._dim

    def name(self) -> str:
        return f"api:{self.model_name}"


def build_embedder(settings: Settings) -> Embedder:
    backend = getattr(settings, "embedder_backend", "local")
    if backend == "api":
        return APIEmbedder()
    model_name = settings.embedding_model
    logger.info("Loading embedding model %s (device=%s)...", model_name, settings.embedding_device)
    return SentenceTransformerEmbedder(model_name=model_name, device=settings.embedding_device)