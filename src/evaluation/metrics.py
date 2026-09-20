from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from src.generation.citation import CitationValidator
from src.schemas import RAGResponse, RetrievedChunk
from src.schemas import DocumentChunk


def _norm(value: str) -> str:
    return value.replace(",", "").replace("%", "").strip().lower()


def relevant_labels(chunks: Sequence[DocumentChunk], required_evidence: List[str]) -> Dict[str, Set[str]]:
    labels: Dict[str, Set[str]] = {}
    for chunk in chunks:
        text = chunk.text.lower()
        matched = {phrase for phrase in required_evidence if phrase.lower() in text}
        labels[chunk.chunk_id] = matched
    return labels


def graded_relevance(chunk_text: str, required_evidence: List[str]) -> float:
    if not required_evidence:
        return 0.0
    text = chunk_text.lower()
    hits = sum(1 for phrase in required_evidence if phrase.lower() in text)
    return hits / len(required_evidence)


def recall_at_k(ranked_ids: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & relevant) / len(relevant)


def precision_at_k(ranked_ids: List[str], relevant: Set[str], k: int) -> float:
    if not ranked_ids or k == 0:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & relevant) / k


def mrr(ranked_ids: List[str], relevant: Set[str]) -> float:
    for i, cid in enumerate(ranked_ids):
        if cid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_ids: List[str], gains: Dict[str, float], k: int) -> float:
    if not ranked_ids:
        return 0.0
    k = min(k, len(ranked_ids))
    dcg = 0.0
    for i, cid in enumerate(ranked_ids[:k]):
        gain = gains.get(cid, 0.0)
        dcg += gain / math.log2(i + 2)
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal)) if ideal else 0.0
    if idcg <= 0:
        return 0.0
    return min(1.0, dcg / idcg)


def is_relevant_set(chunks: Sequence[DocumentChunk], required_evidence: List[str], k: int = 6) -> Set[str]:
    relevant: Set[str] = set()
    for chunk in chunks:
        if graded_relevance(chunk.text, required_evidence) > 0:
            relevant.add(chunk.chunk_id)
    return relevant


def retrieval_metrics(
    ranked: List[RetrievedChunk],
    chunks: Sequence[DocumentChunk],
    required_evidence: List[str],
    k: int = 6,
    relevant_ids: Optional[Set[str]] = None,
) -> Dict[str, float]:
    ranked_ids = [c.chunk.chunk_id for c in ranked]
    if relevant_ids is not None:
        relevant = relevant_ids
    else:
        relevant = is_relevant_set(chunks, required_evidence, k)
    gains = {c.chunk_id: graded_relevance(c.text, required_evidence) for c in chunks}
    return {
        "recall_at_k": round(recall_at_k(ranked_ids, relevant, k), 4),
        "precision_at_k": round(precision_at_k(ranked_ids, relevant, k), 4),
        "mrr": round(mrr(ranked_ids, relevant), 4),
        "ndcg_at_k": round(ndcg_at_k(ranked_ids, gains, k), 4),
    }


def page_correctness(source_chunks: List[RetrievedChunk], expected_pages: List[int]) -> bool:
    if not expected_pages:
        return True
    return any(c.chunk.page in expected_pages for c in source_chunks)


def numeric_answer_correctness(answer: str, key_values: List[str]) -> Dict[str, bool]:
    norm_answer = _norm(answer)
    result: Dict[str, bool] = {}
    for value in key_values:
        result[value] = _norm(value) in norm_answer
    return result


def citation_correctness(answer: str, source_chunks: List[RetrievedChunk]) -> Dict[str, bool]:
    pages_cited = {int(p) for p in re.findall(r"Page (\d+)", answer)}
    evidence_pages = {c.chunk.page for c in source_chunks} if source_chunks else set()
    has_citation = bool(pages_cited)
    correct_page = bool(pages_cited & evidence_pages)
    return {
        "has_citation": has_citation,
        "citation_page_matches_evidence": correct_page and any(
            c.chunk.page in pages_cited for c in source_chunks
        ),
    }


def faithfulness(answer: str, source_chunks: List[RetrievedChunk]) -> bool:
    validator = CitationValidator()
    result = validator.validate(answer, source_chunks, numeric_question=True)
    return result.claim_supported


def completeness(answer: str, required_evidence: List[str]) -> float:
    norm_answer = _norm(answer)
    if not required_evidence:
        return 1.0
    short_phrases = [p for p in required_evidence if len(p) <= 24]
    if not short_phrases:
        return 1.0
    hits = sum(1 for p in short_phrases if _norm(p) in norm_answer or p.lower() in answer.lower())
    return hits / len(short_phrases)


def generation_metrics(
    response: RAGResponse,
    key_values: List[str],
    source_chunks: List[RetrievedChunk],
    required_evidence: List[str],
    expected_pages: List[int],
) -> Dict[str, object]:
    return {
        "numeric_correctness": numeric_answer_correctness(response.answer, key_values),
        "numeric_recall": round(
            sum(1 for v in numeric_answer_correctness(response.answer, key_values).values() if v)
            / max(1, len(key_values)),
            4,
        ),
        "citation_correctness": citation_correctness(response.answer, source_chunks),
        "faithful": faithfulness(response.answer, source_chunks),
        "completeness": completeness(response.answer, required_evidence),
        "page_correct": page_correctness(source_chunks, expected_pages),
    }


def mean_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    keys: Set[str] = set()
    for m in metrics_list:
        keys.update(m.keys())
    out: Dict[str, float] = {}
    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            out[key] = round(float(np.mean(values)), 4)
    return out