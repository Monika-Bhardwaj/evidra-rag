"""Tests for Settings invariants: retrieval-config fingerprint determinism and sensitivity."""

from __future__ import annotations

import pytest

from src.config import Settings, retrieval_config_fingerprint


def test_fingerprint_is_deterministic() -> None:
    a = Settings(hybrid_alpha=0.5, rerank_top_k=5)
    b = Settings(hybrid_alpha=0.5, rerank_top_k=5)
    assert retrieval_config_fingerprint(a) == retrieval_config_fingerprint(b)
    assert len(retrieval_config_fingerprint(a)) == 12


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hybrid_alpha", 0.6),
        ("hybrid_method", "rrf"),
        ("dense_top_k", 24),
        ("bm25_top_k", 24),
        ("rerank_top_k", 8),
        ("final_top_k", 10),
        ("min_similarity", 0.4),
        ("use_query_expansion", False),
        ("use_query_routing", False),
        ("use_reranker", True),
        ("reranker_kind", "score"),
        ("embedding_model", "other-model"),
    ],
)
def test_fingerprint_changes_when_retrieval_setting_changes(field: str, value: object) -> None:
    base = Settings()
    base_fp = retrieval_config_fingerprint(base)
    changed = Settings(**{field: value})
    assert retrieval_config_fingerprint(changed) != base_fp, field


def test_fingerprint_ignores_generation_and_ops_settings() -> None:
    base = Settings()
    altered = Settings(
        llm_temperature=0.7,
        llm_max_tokens=1024,
        llm_timeout_seconds=60.0,
        chunk_size_tokens=300,
        log_level="DEBUG",
        api_rate_limit_per_min=10,
    )
    assert retrieval_config_fingerprint(base) == retrieval_config_fingerprint(altered)


def test_fingerprint_known_string() -> None:
    """Pin the default-config fingerprint hash to a literal.

    If this constant changes, either a retrieval-affecting default changed (intended,
    but then the change must ship with a new eval archive) or the fingerprint function
    itself drifted silently (which would poison cache keys).
    """
    assert retrieval_config_fingerprint(Settings()) == "4ddfdff25521"
