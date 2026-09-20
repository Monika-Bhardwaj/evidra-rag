from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.cache import DiskCache
from src.config import get_settings
from src.logging_utils import get_logger
from src.pipeline import IndexNotFoundError, RagPipeline

logger = get_logger(__name__)

settings = get_settings()
app = FastAPI(
    title="Agent-as-a-Judge RAG API",
    version="1.0.0",
    description="Document-grounded question answering over the Agent-as-a-Judge paper.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: Optional[RagPipeline] = None


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class QueryRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []
    include_debug: bool = False


class QueryResponse(BaseModel):
    question: str
    answer: str
    evidence_sufficient: bool
    grounded_confidence: float
    citations: List[dict]
    sources: List[dict]
    retrieval_debug: Optional[dict] = None
    generation: Optional[dict] = None
    warning: Optional[str] = None


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = RagPipeline(settings)
        except IndexNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _pipeline


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "llm_provider": _pipeline.llm.provider_name if _pipeline else "uninitialized"}


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    pipeline = get_pipeline()
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")
    history = [{"role": m.role, "content": m.content} for m in request.history]
    response = pipeline.answer(request.question, history=history, include_debug=request.include_debug)
    return QueryResponse(**{k: v for k, v in response.to_dict().items() if k != "retrieval_debug" or request.include_debug})


@app.get("/api/evidence/{chunk_id}")
def evidence(chunk_id: str) -> dict:
    pipeline = get_pipeline()
    chunk = pipeline._chunks_by_id.get(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail=f"Unknown chunk_id {chunk_id}")
    return chunk.to_dict()


@app.get("/api/stats")
def stats() -> dict:
    pipeline = get_pipeline()
    return {"pipeline": pipeline.stats_summary(), "llm": pipeline.llm.provider_name}


@app.get("/api/evaluation")
def evaluation() -> dict:
    results_path = settings.processed_dir_path / "evaluation_results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Run `python scripts/evaluate.py` first.")
    return json.loads(results_path.read_text(encoding="utf-8"))


@app.post("/api/ingest")
async def ingest_file(pdf: UploadFile = File(...)) -> dict:
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    dest = settings.raw_dir_path / pdf.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = await pdf.read()
    dest.write_bytes(content)
    from scripts.build_index import build_gold_evidence_map
    from scripts.ingest import run_ingestion

    global _pipeline
    _pipeline = None
    chunks = run_ingestion(settings)
    from src.retrieval.embeddings import build_embedder
    from src.retrieval.vector_store import FAISSVectorStore

    import numpy as np

    embedder = build_embedder(settings)
    vectors = embedder.encode([c.text for c in chunks])
    store = FAISSVectorStore()
    store.add(chunks, np.ascontiguousarray(vectors, dtype=np.float32))
    store.save(settings.index_dir_path)
    build_gold_evidence_map(settings, chunks)
    return {"status": "ok", "chunks": len(chunks), "filename": pdf.filename}


def main() -> None:
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()