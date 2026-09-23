from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    page: int
    section: str
    chunk_type: str
    document_id: str = "doc-1"
    source: str = "Agent-as-a-Judge.pdf"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "page": self.page,
            "section": self.section,
            "chunk_type": self.chunk_type,
            "document_id": self.document_id,
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(
            chunk_id=data["chunk_id"],
            text=data["text"],
            page=data["page"],
            section=data.get("section", ""),
            chunk_type=data.get("chunk_type", "narrative"),
            document_id=data.get("document_id", "doc-1"),
            source=data.get("source", "Agent-as-a-Judge.pdf"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "page": self.chunk.page,
            "section": self.chunk.section,
            "chunk_type": self.chunk.chunk_type,
            "text": self.chunk.text,
            "dense_score": round(self.dense_score, 4),
            "sparse_score": round(self.sparse_score, 4),
            "hybrid_score": round(self.hybrid_score, 4),
            "rerank_score": round(self.rerank_score, 4) if self.rerank_score is not None else None,
        }


@dataclass
class RetrievalDebug:
    query: str
    query_type: str
    expanded_queries: List[str] = field(default_factory=list)
    dense_hits: List[Dict[str, Any]] = field(default_factory=list)
    sparse_hits: List[Dict[str, Any]] = field(default_factory=list)
    fused_hits: List[Dict[str, Any]] = field(default_factory=list)
    reranked_hits: List[Dict[str, Any]] = field(default_factory=list)
    reranker_used: bool = False
    filtered_out_ids: List[str] = field(default_factory=list)
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "expanded_queries": self.expanded_queries,
            "dense_hits": self.dense_hits,
            "sparse_hits": self.sparse_hits,
            "fused_hits": self.fused_hits,
            "reranked_hits": self.reranked_hits,
            "reranker_used": self.reranker_used,
            "filtered_out_ids": self.filtered_out_ids,
            "cached": self.cached,
        }


@dataclass
class GenerationInfo:
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "cached": self.cached,
        }


@dataclass
class Citation:
    page: int
    section: str
    chunk_id: str
    source: str

    def label(self) -> str:
        return f"[Page {self.page}, Section {self.section}]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page": self.page,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "source": self.source,
        }


@dataclass
class RAGResponse:
    question: str
    answer: str
    evidence_sufficient: bool
    grounded_confidence: float
    citations: List[Citation] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_debug: Optional[RetrievalDebug] = None
    generation: Optional[GenerationInfo] = None
    warning: Optional[str] = None
    claims: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "evidence_sufficient": self.evidence_sufficient,
            "grounded_confidence": round(self.grounded_confidence, 3),
            "citations": [c.to_dict() for c in self.citations],
            "sources": self.sources,
            "retrieval_debug": self.retrieval_debug.to_dict() if self.retrieval_debug else None,
            "generation": self.generation.to_dict() if self.generation else None,
            "warning": self.warning,
            "claims": self.claims,
        }
