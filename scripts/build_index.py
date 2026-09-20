from __future__ import annotations

import json
import sys
from typing import List

import numpy as np

sys.path.insert(0, ".")

from scripts.ingest import run_ingestion
from src.config import Settings, get_settings
from src.logging_utils import get_logger
from src.retrieval.embeddings import build_embedder
from src.retrieval.vector_store import FAISSVectorStore
from src.schemas import DocumentChunk

logger = get_logger(__name__)


def build_gold_evidence_map(settings: Settings, chunks: List[DocumentChunk]) -> dict:
    questions_path = settings.eval_questions_path_resolved
    data = json.loads(questions_path.read_text(encoding="utf-8"))
    gold: dict = {}
    for q in data["questions"]:
        keys = q.get("key_values") or []
        if keys:
            required = [_strip_num(k) for k in keys]
        else:
            required = [_strip_num(p) for p in (q.get("required_evidence") or [])]
        expected_pages = q.get("page", [])
        hits = []
        for i, chunk in enumerate(chunks):
            if expected_pages and chunk.page not in expected_pages:
                continue
            text = chunk.text.lower()
            matched = [p for p in required if _strip_num(p) in text or p.lower() in text]
            if len(matched) < len(required):
                continue
            hits.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "page": chunk.page,
                    "section": chunk.section,
                    "chunk_type": chunk.chunk_type,
                    "matched_phrases": matched,
                }
            )
        gold[str(q["id"])] = hits
    out = settings.processed_dir_path / "gold_evidence.json"
    out.write_text(json.dumps(gold, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote gold evidence map to %s", out)
    return gold


def _strip_num(value: str) -> str:
    return value.replace(",", "").replace("%", "").replace("$", "").strip().lower()


def main() -> None:
    settings = get_settings()
    settings.ensure_dirs()
    chunks = run_ingestion(settings)

    embedder = build_embedder(settings)
    logger.info("Encoding %d chunks with %s (device=%s)...", len(chunks), embedder.name(), settings.embedding_device)
    vectors = embedder.encode([c.text for c in chunks])

    store = FAISSVectorStore()
    store.add(chunks, np.ascontiguousarray(vectors, dtype=np.float32))
    store.save(settings.index_dir_path)
    logger.info("FAISS index saved to %s (%d vectors, dim=%d)", settings.index_dir_path, len(chunks), vectors.shape[1])

    build_gold_evidence_map(settings, chunks)
    logger.info("Index build complete.")


if __name__ == "__main__":
    main()