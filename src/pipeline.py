from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.cache import DiskCache
from src.config import Settings, get_settings, retrieval_config_fingerprint
from src.generation.citation import CitationValidator, build_citations
from src.generation.llm import LLMProvider, build_llm
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.generation.security import JailbreakGuard, neutralize_retrieved_text
from src.logging_utils import get_logger
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.embeddings import build_embedder
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.query_expansion import QueryExpander
from src.retrieval.query_understanding import QueryClassifier
from src.retrieval.reranker import build_reranker
from src.retrieval.routing import QueryRouter, RouteSpec
from src.retrieval.vector_store import VectorStore
from src.schemas import (
    DocumentChunk,
    GenerationInfo,
    RAGResponse,
    RetrievalDebug,
    RetrievedChunk,
)
from src.validation.claims import ClaimVerifier

logger = get_logger(__name__)

INSF_MSG = "I could not find sufficient evidence for this answer in the provided document."


class IndexNotFoundError(RuntimeError):
    pass


@dataclass
class PipelineStats:
    queries: int = 0
    retrieval_cache_hits: int = 0
    answer_cache_hits: int = 0
    insufficient_evidence: int = 0
    generation_failures: int = 0
    security_flag_count: int = 0
    total_latency_ms: float = 0.0
    total_est_cost_usd: float = 0.0
    total_tokens: int = 0
    recent: List[Dict] = field(default_factory=list)

    def summary(self) -> Dict:
        return {
            "queries": self.queries,
            "retrieval_cache_hits": self.retrieval_cache_hits,
            "answer_cache_hits": self.answer_cache_hits,
            "cache_hit_rate": round(
                (self.retrieval_cache_hits + self.answer_cache_hits) / max(1, self.queries), 4
            ),
            "insufficient_evidence": self.insufficient_evidence,
            "generation_failures": self.generation_failures,
            "security_flag_count": self.security_flag_count,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "avg_latency_ms": round(self.total_latency_ms / max(1, self.queries), 1),
            "total_est_cost_usd": round(self.total_est_cost_usd, 5),
            "total_tokens": self.total_tokens,
        }


class RagPipeline:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.embedder = build_embedder(self.settings)
        self.vector_store = self._load_vector_store()
        self.bm25 = BM25Retriever(self.vector_store.chunks())
        self.hybrid = HybridRetriever(self.vector_store, self.bm25, self.embedder)
        self.reranker = build_reranker(
            enabled=self.settings.use_reranker,
            model_name=self.settings.reranker_model,
            device=self.settings.embedding_device,
            kind=self.settings.reranker_kind,
        )
        self.llm: LLMProvider = build_llm(self.settings)
        self.classifier = QueryClassifier()
        self.expander = QueryExpander()
        self.router = QueryRouter()
        self.guard = JailbreakGuard()
        self.validator = CitationValidator()
        self.claim_verifier = ClaimVerifier()
        self.cache = DiskCache(self.settings.cache_dir_path)
        self.stats = PipelineStats()
        self._chunks_by_id: Dict[str, DocumentChunk] = {
            c.chunk_id: c for c in self.vector_store.chunks()
        }
        self._warn_on_index_metadata_mismatch()
        self._index_signature = self._compute_index_signature()
        self._retrieval_cfg_hash = self._retrieval_config_fingerprint()
        logger.info(
            "Pipeline ready: %d chunks, embedder=%s, llm=%s, reranker=%s",
            len(self._chunks_by_id),
            self.embedder.name(),
            self.llm.provider_name,
            "on" if self.reranker.is_used() else "off",
        )

    def _load_vector_store(self) -> VectorStore:
        index_dir = self.settings.index_dir_path
        chunks_json = index_dir / "chunks.json"
        if not chunks_json.exists():
            raise IndexNotFoundError("No index found. Run `python scripts/build_index.py` first.")
        return self._load_faiss(index_dir)

    def _load_faiss(self, index_dir) -> VectorStore:
        from src.retrieval.vector_store import FAISSVectorStore, InMemoryVectorStore

        try:
            store = FAISSVectorStore.load(index_dir)
            if store.size() > 0:
                return store
        except Exception as exc:
            logger.warning("FAISS index load failed (%s); trying in-memory fallback.", exc)
        store = InMemoryVectorStore.load(index_dir)
        if store.size() == 0:
            raise IndexNotFoundError(
                "Index artifacts exist but could not be loaded. Rebuild the index."
            )
        logger.info("Using in-memory vector-store fallback (no FAISS index).")
        return store

    def _warn_on_index_metadata_mismatch(self) -> None:
        meta_path = self.settings.index_dir_path / "index_meta.json"
        if not meta_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        stored = meta.get("embedding_model", "")
        if stored and stored != self.settings.embedding_model:
            logger.warning(
                "Index was built with embedding_model=%r but settings use %r; "
                "rebuild the index or scores may be invalid.",
                stored,
                self.settings.embedding_model,
            )

    def _compute_index_signature(self) -> str:
        ids = sorted(self._chunks_by_id.keys())[:50]
        return f"{len(self._chunks_by_id)}:{self.embedder.name()}:{ids}"

    def _retrieval_config_fingerprint(self) -> str:
        """Hash of every retrieval/sufficiency setting that affects cached results.

        The stored retrieval hits are only valid while the retrieval stack and the
        sufficiency gate are configured the same way. Omitting thresholds here caused
        stale entries from an earlier config to be served (ABS-6 regression).
        """
        return retrieval_config_fingerprint(self.settings)

    def answer(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        include_debug: bool = False,
    ) -> RAGResponse:
        start = time.perf_counter()
        self.stats.queries += 1

        if not question or not question.strip():
            return RAGResponse(
                question=question or "",
                answer="Please provide a question about the paper.",
                evidence_sufficient=False,
                grounded_confidence=1.0,
                warning="empty-query",
                generation=GenerationInfo(provider=self.llm.provider_name, model=self.llm.model),
            )

        security = self.guard.assess(question)
        if security.flagged:
            self.stats.security_flag_count += 1

        intent = self.classifier.classify(question)
        expanded = (
            self.expander.expand(question, intent)
            if self.settings.use_query_expansion
            else [question]
        )
        route = (
            self.router.route(intent, question) if self.settings.use_query_routing else RouteSpec()
        )

        retrieval_debug = RetrievalDebug(
            query=question,
            query_type=intent.category,
            expanded_queries=expanded,
        )

        evidence, sufficient, cache_hit = self._retrieve_with_cache(
            question, expanded, intent, route, retrieval_debug
        )

        rerank_hits_raw = []
        if evidence:
            evidence = self.reranker.rerank(question, evidence, self.settings.rerank_top_k)
            evidence = self.router.select_evidence(evidence, intent, self.settings.final_top_k)
            rerank_hits_raw = [c.to_dict() for c in evidence]
        retrieval_debug.reranked_hits = rerank_hits_raw
        retrieval_debug.reranker_used = self.reranker.is_used()

        if not evidence or not sufficient:
            self.stats.insufficient_evidence += 1
            return self._build_response(
                question=question,
                answer=INSF_MSG,
                evidence=[],
                grounded_confidence=0.0,
                sufficient=False,
                retrieval_debug=retrieval_debug,
                generation=None,
                warning=security.safe_message if security.flagged else None,
                start=start,
                history=history,
            )

        answer, generation, failed = self._generate(question, evidence, history, retrieval_debug)
        if failed:
            self.stats.generation_failures += 1

        if answer.strip().lower() == INSF_MSG.lower():
            self.stats.insufficient_evidence += 1
            return self._build_response(
                question=question,
                answer=INSF_MSG,
                evidence=[],
                grounded_confidence=0.0,
                sufficient=False,
                retrieval_debug=retrieval_debug,
                generation=generation,
                warning="provider returned the abstention sentence (fail closed). [provider-abstained]",
                start=start,
                history=history,
            )

        validation = self.validator.validate(
            answer,
            evidence,
            numeric_question=intent.is_numeric,
        )
        claim_verdicts = self.claim_verifier.verify(answer, evidence)
        substantive_answer = INSF_MSG.lower() not in answer.lower()
        has_unsupported = any(c.verdict == "unsupported" for c in claim_verdicts)
        claims_supported = not has_unsupported and (bool(claim_verdicts) or not substantive_answer)
        if not validation.claim_supported or not claims_supported:
            self.stats.insufficient_evidence += 1
            reason = "claim-verification" if not claims_supported else "validation"
            warning = (
                "Low grounding confidence: unsupported claims detected; abstaining with the "
                "exact abstention sentence (fail closed)."
                if not claims_supported
                else "Low grounding confidence; unsupported claims detected. Abstaining (fail closed)."
            )
            return self._build_response(
                question=question,
                answer=INSF_MSG,
                evidence=[],
                grounded_confidence=0.0,
                sufficient=False,
                retrieval_debug=retrieval_debug,
                generation=generation,
                warning=f"{warning} [{reason}]",
                start=start,
                history=history,
            )
        if security.flagged:
            answer = (
                answer
                + "\n\n[Note: a prompt-injection attempt was detected and neutralized; this answer remains PDF-grounded.]"
            )
        return self._build_response(
            question=question,
            answer=answer,
            evidence=evidence,
            grounded_confidence=0.9 if validation.claim_supported else 0.3,
            sufficient=validation.claim_supported,
            retrieval_debug=retrieval_debug,
            generation=generation,
            claims=[c.to_dict() for c in claim_verdicts],
            warning=(
                security.safe_message
                if security.flagged
                else (
                    "Low grounding confidence; unsupported claims detected."
                    if not validation.claim_supported
                    else None
                )
            ),
            start=start,
            history=history,
        )

    def _retrieve_with_cache(
        self,
        question: str,
        expanded: List[str],
        intent,
        route: RouteSpec,
        debug: RetrievalDebug,
    ) -> tuple[List[RetrievedChunk], bool, bool]:

        cache_tag = self._index_signature
        cache_payload = json.dumps(
            {
                "q": question,
                "e": expanded,
                "a": self.settings.hybrid_alpha,
                "m": self.settings.hybrid_method,
                "k": self.settings.dense_top_k,
                "tag": cache_tag,
                "cfg": self._retrieval_cfg_hash,
            },
            sort_keys=True,
        )
        cached = (
            self.cache.get("retrieval", cache_payload)
            if self.settings.enable_retrieval_cache
            else None
        )
        if cached:
            self.stats.retrieval_cache_hits += 1
            chunks = []
            for d in cached["chunks"]:
                c = self._chunks_by_id.get(d["chunk_id"])
                if c is not None:
                    chunks.append(
                        RetrievedChunk(
                            chunk=c,
                            dense_score=d["dense_score"],
                            sparse_score=d["sparse_score"],
                            hybrid_score=d["hybrid_score"],
                            rerank_score=d.get("rerank_score"),
                        )
                    )
            debug.fused_hits = [c.to_dict() for c in chunks]
            debug.cached = True
            # The sufficiency verdict is persisted with the hit list so the
            # cache-hit path is byte-identical to a fresh retrieval for the same
            # query/config/index (fresh `_sufficient` sees the full candidate
            # set, top-k only would diverge — ABS-6 stale-verdict class of bug).
            sufficient = cached.get("sufficient", False)
            return chunks, sufficient, True

        top, sufficient, filtered = self.hybrid.retrieve(
            query=question,
            expanded_queries=expanded,
            dense_top_k=self.settings.dense_top_k,
            bm25_top_k=self.settings.bm25_top_k,
            top_k=max(self.settings.rerank_top_k, self.settings.final_top_k),
            alpha=self.settings.hybrid_alpha,
            method=self.settings.hybrid_method,
            min_similarity=self.settings.min_similarity,
            chunk_types=route.chunk_types or None,
            type_boost=route.type_boost,
        )
        debug.dense_hits = [
            c.to_dict()
            for c in sorted(top, key=lambda c: c.dense_score, reverse=True)[
                : self.settings.dense_top_k
            ]
        ]
        debug.sparse_hits = [
            c.to_dict()
            for c in sorted(top, key=lambda c: c.sparse_score, reverse=True)[
                : self.settings.bm25_top_k
            ]
        ]
        debug.fused_hits = [c.to_dict() for c in top]
        debug.filtered_out_ids = filtered
        if self.settings.enable_retrieval_cache:
            self.cache.set(
                "retrieval",
                cache_payload,
                {
                    "sufficient": sufficient,
                    "chunks": [c.to_dict() for c in top],
                },
            )
        return top, sufficient, False

    def _generate(
        self,
        question: str,
        evidence: List[RetrievedChunk],
        history: Optional[List[Dict[str, str]]],
        debug: RetrievalDebug,
    ) -> tuple[str, GenerationInfo, bool]:
        user_prompt = build_user_prompt(
            question=question,
            evidence=self._sanitize_prompt_evidence(evidence),
            history=history,
            context_max_tokens=self.settings.context_max_tokens,
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history or []:
            role = "assistant" if turn.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": turn.get("content", "")})
        messages.append({"role": "user", "content": user_prompt})

        result = self.llm.chat(messages)
        generation = GenerationInfo(
            provider=result.provider,
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=0.0,
            estimated_cost_usd=result.cost_usd,
            cached=result.cached,
        )
        if result.failed or not result.text.strip():
            evidence_first = self._evidence_first_answer(evidence)
            generation.estimated_cost_usd = 0.0
            return evidence_first, generation, True

        generation.latency_ms = 0.0
        return result.text, generation, False

    def _sanitize_prompt_evidence(self, evidence: List[RetrievedChunk]) -> List[RetrievedChunk]:
        from dataclasses import replace

        sanitized: List[RetrievedChunk] = []
        for c in evidence:
            safe_text = neutralize_retrieved_text(c.chunk.text)
            if safe_text == c.chunk.text:
                sanitized.append(c)
                continue
            clean = replace(c.chunk, text=safe_text)
            sanitized.append(
                RetrievedChunk(
                    chunk=clean,
                    dense_score=c.dense_score,
                    sparse_score=c.sparse_score,
                    hybrid_score=c.hybrid_score,
                    rerank_score=c.rerank_score,
                )
            )
        return sanitized

    def _evidence_first_answer(self, evidence: List[RetrievedChunk]) -> str:
        lines = [
            "The language model could not be reached. Returning the best supporting evidence from the PDF:"
        ]
        for i, rc in enumerate(evidence[:2], start=1):
            preview = rc.chunk.text[:400].strip()
            lines.append(
                f"{i}. [Page {rc.chunk.page}, Section {rc.chunk.section or '(unknown)'}] {preview}"
            )
        return "\n\n".join(lines) + "\n\n(Generation failed; no invented content returned.)"

    def _build_response(
        self,
        question: str,
        answer: str,
        evidence: List[RetrievedChunk],
        grounded_confidence: float,
        sufficient: bool,
        retrieval_debug: RetrievalDebug,
        generation: Optional[GenerationInfo],
        warning: Optional[str],
        start: float,
        history: Optional[List[Dict[str, str]]],
        claims: Optional[List[Dict[str, Any]]] = None,
    ) -> RAGResponse:
        latency_ms = (time.perf_counter() - start) * 1000.0
        if generation is not None:
            generation.latency_ms = latency_ms
            self.stats.total_latency_ms += latency_ms
            self.stats.total_est_cost_usd += generation.estimated_cost_usd
            self.stats.total_tokens += generation.prompt_tokens + generation.completion_tokens
        else:
            generation = GenerationInfo(
                provider=self.llm.provider_name, model=self.llm.model, latency_ms=latency_ms
            )
        citations = build_citations(evidence)
        sources = [
            {
                "source": c.source,
                "page": c.page,
                "section": c.section,
                "chunk_id": c.chunk_id,
            }
            for c in citations
        ]
        response = RAGResponse(
            question=question,
            answer=answer,
            evidence_sufficient=sufficient,
            grounded_confidence=grounded_confidence,
            citations=citations,
            sources=sources,
            retrieval_debug=retrieval_debug,
            generation=generation,
            warning=warning,
            claims=claims or [],
        )
        self._log_event(response)
        return response

    def _log_event(self, response: RAGResponse) -> None:
        data = {
            "event": "rag_answer",
            "question": response.question,
            "query_type": response.retrieval_debug.query_type if response.retrieval_debug else None,
            "chunk_ids": [s["chunk_id"] for s in response.sources],
            "evidence_sufficient": response.evidence_sufficient,
            "confidence": response.grounded_confidence,
            "gen_provider": response.generation.provider if response.generation else None,
            "gen_model": response.generation.model if response.generation else None,
            "tokens": (response.generation.prompt_tokens + response.generation.completion_tokens)
            if response.generation
            else 0,
            "latency_ms": round(response.generation.latency_ms, 1) if response.generation else None,
            "est_cost_usd": response.generation.estimated_cost_usd if response.generation else 0.0,
            "warning": response.warning,
        }
        logger.info(json.dumps(data, ensure_ascii=False))

    def stats_summary(self) -> Dict:
        return self.stats.summary()
