# Failure Analysis: ABS-6 Abstention Leak (2026-09-23)

## Summary

`ABS-6` — "Who was named Time Person of the Year for 2023?" — is an
out-of-knowledge abstention case. It must return the exact abstention sentence:
*"I could not find sufficient evidence for this answer in the provided
document."* Instead, it answered with paper content ("Table 1 Preliminary
Statistics of AI Developers") at near-zero citation relevance, while the
"correct" response was only ever reachable through subtraction: the question is
not answerable from the paper at all.

The leak survived **three** separate layers of the retrieval/abstention
pipeline, each of which had to be fixed at the root. This document is the
post-mortem; it is referenced from the evaluation fixture notes in
`src/evaluation/questions.json`.

## Timeline

1. **First symptom (reported)**: ABS-6 returned an answer about the paper
   instead of abstaining; a stale retrieval cache from an older retrieval
   configuration was suspected.
2. **Cache-key hardening (root cause #3)**: `retrieval_config_fingerprint()` was
   added (hash of every retrieval/sufficiency setting) and embedded in the cache
   payload. Old entries could no longer be served. The pipeline then abstained
   deterministically in repeated live runs — but **only on the cache-hit path**.
3. **Full-eval failure reproduced**: a fresh `scripts/evaluate.py` run (no warm
   cache) still leaked `ABS-6`. A direct reproduction showed the *fresh*
   retrieval path returned `sufficient=True` while the *cached* path returned
   `False` for the same question. Two distinct root causes remained.

## Root causes

### #1 — Sparse-token fallback in the sufficiency gate

`HybridRetriever._sufficient` had a fallback: chunks whose dense cosine landed in
`[knowledge_boundary_min_sim, min_similarity)` could still mark a query
"sufficient" when ≥3 non-stopword tokens overlapped between query and chunk.
For ABS-6 the incidental overlap ("2023", "year", "named") outvoted a best dense
cosine of ~0.31 — far below `min_similarity` (0.35).

### #2 — Expansion-averaged embeddings moved the gate

`HybridRetriever.retrieve` builds one query vector as the **mean** of the raw
question plus expanded variants. ABS-6 is classified `numerical_lookup`, whose
expansion injects generic terms ("average time", "duration", "how long",
"results table", "numbers"). That shifted mean embedding reached cosine **0.45**
against a filename chunk (`results/processing time.txt.`, on the paper's page
28) — itself above `min_similarity`. So even with the fallback removed, the gate
passed on an artifact of expansion, not on real topical similarity.

### #3 — Cached vs. fresh paths disagreed on sufficiency

The cache-hit path recomputed sufficiency from the stored **top-k** chunks while
the fresh path evaluated the **full** dense∪sparse candidate set. A high-dense
chunk ranked outside the top-k changed the verdict between the two paths. This:
- produced non-deterministic abstention verdicts for the same query/config, and
- **masked root causes #1/#2** in earlier verification, which exercised the
  cache-hit path.

## Fixes

All changes landed 2026-09-23; test suite 74 green, ruff clean.

| Fix | Mechanism |
|---|---|
| Raw-query gate | `_sufficient` scores the **original** (unexpanded) question embedding against candidates. Expansion remains a *ranking* aid only; it can no longer move the grounding gate. |
| Fallback removed | `knowledge_boundary_min_sim` setting and the sparse-token fallback were deleted; `_sufficient` is now `max(raw dense cosine) >= min_similarity`. |
| Persisted verdict | `_retrieve_with_cache` stores `{sufficient, chunks}` in the cache entry; the cache-hit path returns the stored verdict, making cache-hit and cache-miss answers byte-identical for the same query/config/index. |
| Config fingerprint | (earlier) `retrieval_config_fingerprint()` includes every retrieval setting in the cache key so config changes cannot serve stale hits. |

## Measurements

Raw (unexpanded) query → best dense cosine over the corpus:

| set | min | max | vs. gate (0.35) |
|---|---|---|---|
| 18 golden questions | **0.473** | 0.801 | all clear with margin |
| 6 abstention cases (incl. ABS-6) | 0.153 | **0.300** | all fail closed |

The gate therefore separates the two sets cleanly with no golden regression
(golden pass stays 18/18; recall@6 0.8148, nDCG@6 0.6949 unchanged).

## Result

Archive `docs/eval_results/20260923-153131` (offline-extractive):
18/18 golden, abstention **6/6 (exact string 1.0)**, adversarial 3/3
(resistance 1.0), citation/page accuracy 1.0, faithfulness 1.0. The prior
differently-configured run records the failure for comparison
(`docs/eval_results/20260923-145944`).

## Residual considerations

- **Trade-off**: the gate is deliberately stricter. A user-phrased in-paper
  question that scores below 0.35 raw cosine (but high with expansion) now
  abstains instead of answering. This honors EVIDRA's fail-closed mandate:
  better abstention than unsupported invention. If real traffic shows false
  abstentions, revisit *expansion quality* (below), not the gate.
- **Expansion quality**: `numerical_lookup` adds generic terms ("results
  table", "numbers") that are weak for off-topic questions. Harmless now that
  the gate ignores expansion, but templates remain a tuning candidate.
- **Claim gate still guards**: even with retrieval sufficiency, the post-generation
  claim verifier and citation validator remain the final fail-closed layer.