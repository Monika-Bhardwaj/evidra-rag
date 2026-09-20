from __future__ import annotations

from src.generation.citation import CitationValidator
from src.generation.llm import LocalExtractiveProvider
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.retrieval.hybrid import HybridRetriever
from src.schemas import RetrievedChunk
from src.retrieval.vector_store import InMemoryVectorStore
from src.retrieval.bm25 import BM25Retriever


def _evidence(retriever, query) -> list[RetrievedChunk]:
    top, _, _ = retriever.retrieve(query, top_k=3, alpha=0.7)
    return top


def test_system_prompt_grounding_rules() -> None:
    assert "ONLY using the supplied evidence" in SYSTEM_PROMPT
    assert "I could not find sufficient evidence" in SYSTEM_PROMPT
    assert "do not fabricate" in SYSTEM_PROMPT.lower()


def test_context_builder_has_delimiters_and_metadata(small_corpus, fake_embedder) -> None:
    store = InMemoryVectorStore()
    vectors = fake_embedder.encode([c.text for c in small_corpus])
    store.add(small_corpus, vectors)
    retriever = HybridRetriever(store, BM25Retriever(small_corpus), fake_embedder)
    evidence = _evidence(retriever, "OpenHands average cost")
    prompt = build_user_prompt("What is OpenHands' average cost?", evidence)
    assert ">>RETRIEVED_EVIDENCE_START<<" in prompt
    assert "Page:" in prompt
    assert "Section:" in prompt
    assert "Chunk:" in prompt


def test_numeric_citation_validation_pass(small_corpus, fake_embedder) -> None:
    evidence = _evidence_for(small_corpus, fake_embedder, "OpenHands average cost")
    validator = CitationValidator()
    result = validator.validate(
        "OpenHands average cost is $6.38 and time is 362.41 seconds.",
        evidence,
        numeric_question=True,
    )
    assert result.claim_supported


def test_fabricated_number_rejected(small_corpus, fake_embedder) -> None:
    evidence = _evidence_for(small_corpus, fake_embedder, "OpenHands average cost")
    validator = CitationValidator()
    result = validator.validate(
        "OpenHands average cost is $999.99.",
        evidence,
        numeric_question=True,
        expected_numbers=["999.99"],
    )
    assert result.missing_numbers


def _evidence_for(corpus, embedder, query) -> list[RetrievedChunk]:
    store = InMemoryVectorStore()
    vectors = embedder.encode([c.text for c in corpus])
    store.add(corpus, vectors)
    retriever = HybridRetriever(store, BM25Retriever(corpus), embedder)
    return _evidence(retriever, query)


def test_local_extractive_answer_uses_evidence(small_corpus, fake_embedder) -> None:
    evidence = _evidence_for(small_corpus, fake_embedder, "OpenHands average cost")
    prompt = build_user_prompt("What is OpenHands' average cost?", evidence)
    provider = LocalExtractiveProvider()
    result = provider.chat(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    )
    assert "According to the paper" in result.text
    assert "Page 5" in result.text
    assert any(
        token in result.text.lower() for token in ("openhands", "6.38", "cost")
    )


def test_local_extractive_no_evidence() -> None:
    provider = LocalExtractiveProvider()
    prompt = "Question: what is x?\nThere is no evidence."
    result = provider.chat(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    )
    assert "could not find sufficient evidence" in result.text


def test_offline_provider_never_invents() -> None:
    provider = LocalExtractiveProvider()
    prompt = "Question: How many penguins? There is no evidence."
    result = provider.chat(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    )
    assert "penguin" not in result.text.lower() or "could not find sufficient evidence" in result.text
    assert "72" not in result.text