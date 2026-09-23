from __future__ import annotations

from src.config import Settings, retrieval_config_fingerprint


def test_fingerprint_is_deterministic() -> None:
    a = retrieval_config_fingerprint(Settings())
    b = retrieval_config_fingerprint(Settings())
    assert a == b
    assert len(a) == 12


def test_fingerprint_changes_with_min_similarity() -> None:
    base = retrieval_config_fingerprint(Settings())
    changed = retrieval_config_fingerprint(Settings(min_similarity=0.40))
    assert changed != base


def test_fingerprint_changes_with_dense_top_k() -> None:
    base = retrieval_config_fingerprint(Settings())
    changed = retrieval_config_fingerprint(Settings(dense_top_k=32))
    assert changed != base


def test_fingerprint_changes_with_retrieval_stack() -> None:
    base = retrieval_config_fingerprint(Settings())
    no_rerank = retrieval_config_fingerprint(Settings(use_reranker=not Settings().use_reranker))
    assert no_rerank != base
