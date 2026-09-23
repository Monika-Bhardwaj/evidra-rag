from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from src.config import Settings
from src.logging_utils import get_logger

logger = get_logger(__name__)


class Embedder(ABC):
    @abstractmethod
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray: ...

    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def name(self) -> str: ...


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


def build_embedder(settings: Settings) -> Embedder:
    model_name = settings.embedding_model
    logger.info("Loading embedding model %s (device=%s)...", model_name, settings.embedding_device)
    return SentenceTransformerEmbedder(model_name=model_name, device=settings.embedding_device)
