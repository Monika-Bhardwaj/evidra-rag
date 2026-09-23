# Ablation report — retrieval / evidence layer (Phase 3)

## Fixed settings
- embedding_model: sentence-transformers/all-MiniLM-L6-v2
- dense_top_k: 48
- bm25_top_k: 48
- index_chunks: 139
- min_similarity: 0.35
- questions: 15
- reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
- generated_utc: 2026-09-20T18:54:23

## Retrieval sweep (top_k = 6)
Binary gold relevance (gold chunk ids from `gold_evidence.json`).

| variant | method | alpha | routing | expansion | hit | suff | R@6 | P@6 | MRR | nDCG@6 |
|---|---|---|---|---|---|---|---|---|---|---|
| weighted-0.5 | weighted | 0.5 | True | True | 1.000 | 1.000 | 0.8944 | 0.2556 | 0.7989 | 0.8033 |
| weighted-0.5 | weighted | 0.5 | False | True | 1.000 | 1.000 | 0.8944 | 0.2556 | 0.8044 | 0.8026 |
| bm25 | weighted | 0.0 | True | False | 0.933 | 1.000 | 0.8500 | 0.2667 | 0.8500 | 0.8184 |
| bm25 | weighted | 0.0 | True | True | 0.933 | 1.000 | 0.8500 | 0.2667 | 0.8500 | 0.8184 |
| weighted-0.3 | weighted | 0.3 | False | False | 0.933 | 1.000 | 0.8500 | 0.2667 | 0.8500 | 0.8137 |
| weighted-0.3 | weighted | 0.3 | False | True | 0.933 | 1.000 | 0.8389 | 0.2556 | 0.8500 | 0.8076 |
| weighted-0.3 | weighted | 0.3 | True | True | 0.933 | 1.000 | 0.8389 | 0.2556 | 0.8167 | 0.7970 |
| bm25 | weighted | 0.0 | False | False | 0.933 | 1.000 | 0.8278 | 0.2556 | 0.8500 | 0.7929 |
| bm25 | weighted | 0.0 | False | True | 0.933 | 1.000 | 0.8278 | 0.2556 | 0.8500 | 0.7929 |
| weighted-0.3 | weighted | 0.3 | True | False | 0.933 | 1.000 | 0.8333 | 0.2556 | 0.8056 | 0.7835 |
| weighted-0.7 | weighted | 0.7 | False | True | 0.933 | 1.000 | 0.8778 | 0.2444 | 0.7711 | 0.7718 |
| weighted-0.5 | weighted | 0.5 | False | False | 0.933 | 1.000 | 0.8222 | 0.2444 | 0.8133 | 0.7695 |
| weighted-0.5 | weighted | 0.5 | True | False | 0.933 | 1.000 | 0.8222 | 0.2444 | 0.8022 | 0.7687 |
| weighted-0.7 | weighted | 0.7 | False | False | 0.933 | 1.000 | 0.8000 | 0.2333 | 0.7578 | 0.7253 |
| weighted-0.7 | weighted | 0.7 | True | False | 0.867 | 1.000 | 0.7556 | 0.2333 | 0.7467 | 0.7254 |
| weighted-0.7 | weighted | 0.7 | True | True | 0.867 | 1.000 | 0.8111 | 0.2333 | 0.7267 | 0.7227 |
| rrf | rrf | 0.5 | False | False | 0.867 | 1.000 | 0.7167 | 0.2111 | 0.7800 | 0.7099 |
| rrf | rrf | 0.5 | False | True | 0.800 | 1.000 | 0.7444 | 0.2222 | 0.7467 | 0.7152 |
| dense | weighted | 1.0 | False | False | 0.800 | 1.000 | 0.6944 | 0.2000 | 0.6800 | 0.6272 |
| dense | weighted | 1.0 | True | False | 0.800 | 1.000 | 0.7278 | 0.2111 | 0.6500 | 0.6271 |
| dense | weighted | 1.0 | False | True | 0.800 | 1.000 | 0.7444 | 0.2000 | 0.5689 | 0.5702 |
| dense | weighted | 1.0 | True | True | 0.800 | 1.000 | 0.7778 | 0.2111 | 0.5333 | 0.5681 |
| rrf | rrf | 0.5 | True | True | 0.733 | 1.000 | 0.5167 | 0.1667 | 0.6300 | 0.5249 |
| rrf | rrf | 0.5 | True | False | 0.733 | 1.000 | 0.4833 | 0.1556 | 0.6300 | 0.5063 |

**Best retrieval config (k=6):** `weighted-0.5` method=weighted alpha=0.5 routing=True expansion=True (hit=1.000, nDCG@6=0.8033).

## Window-size sweep (best config)
| top_k | hit | suff | R@k | P@k | MRR | nDCG@k |
|---|---|---|---|---|---|---|
| 4 | 0.867 | 1.000 | 0.8167 | 0.3500 | 0.7722 | 0.7697 |
| 6 | 1.000 | 1.000 | 0.8944 | 0.2556 | 0.7989 | 0.8033 |
| 8 | 1.000 | 1.000 | 0.9056 | 0.2000 | 0.7989 | 0.8100 |

## Full-pipeline passes
| variant | pass | citedR | citedP | R@6 | P@6 | MRR | nDCG@6 | num | comp | cit | faith | page | ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default-off (weighted-0.5, rerank off) | 15/15 | 0.8944 | 0.3067 | 0.8944 | 0.2556 | 0.7989 | 0.7081 | 1.0000 | 0.9833 | 1.0000 | 1.0000 | 1.0000 | 5.2 |
| best-off (weighted-0.5, rerank off) | 15/15 | 0.8944 | 0.3067 | 0.8944 | 0.2556 | 0.7989 | 0.7081 | 1.0000 | 0.9833 | 1.0000 | 1.0000 | 1.0000 | 5.6 |
| default-score (rerank=score) | 14/15 | 0.8778 | 0.2933 | 0.8944 | 0.2556 | 0.7989 | 0.7081 | 0.9333 | 0.9500 | 1.0000 | 1.0000 | 1.0000 | 5.1 |
| default-cross (rerank=cross) | 15/15 | 0.8278 | 0.2933 | 0.8944 | 0.2556 | 0.7989 | 0.7081 | 1.0000 | 0.9167 | 1.0000 | 1.0000 | 0.9333 | 848.4 |

## Decision
- Recommended variant: **default-off (weighted-0.5, rerank off)** (pass 15/15, cited-recall 0.8944, cited-precision 0.3067, nDCG@6 0.7081).
- Frozen default remains `weighted` α=0.5 (routing/expansion on) unless the recommended variant differs and is adopted in `src/config.py`.
