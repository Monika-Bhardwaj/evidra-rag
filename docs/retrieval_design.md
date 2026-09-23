# Retrieval Design

Why the retrieval stack is shaped the way it is, what was measured, and what changed after
failure analysis.

## Building blocks

| Block | Role |
|---|---|
| Dense (sentence-transformers + FAISS) | Semantic similarity. `all-MiniLM-L6-v2`, local CPU, L2-normalised vectors → inner product == cosine. Top-`DENSE_TOP_K` (48). |
| Sparse (BM25, `rank_bm25`) | Lexical guarantee for exact tokens (`6.38`, `97.72%`, `cn9o`, requirement IDs). Top-`BM25_TOP_K` (48). |
| Fusion | `weighted` (default): min-max-normalise both score sets, `alpha * dense + (1-alpha) * sparse` with `alpha=0.5`. `rrf` available. |
| Rerank (optional) | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) over the fused pool (`RERANK_TOP_K`=5), then `FINAL_TOP_K`=6 evidence rows to the LLM. **Off by default** — the Phase 3 ablation measured no lift on this corpus. |
| Routing | Type boosts (e.g. `table`/`result`/`cost-analysis` for numeric questions) plus a guarantee: numeric questions keep at least one table/result chunk in the evidence. |
| Expansion | Query variants ("average cost", "table", numbers) computed by `QueryClassifier`/`query_expansion.py`. **Ranking-only.** |

## The sufficiency gate (fail-closed)

Everything above decides *which* chunks are candidates and their order. The *go / no-go* decision
is separate and deliberately stricter:

```
evidence_sufficient := ∃ chunk ∈ candidates : raw_dense[chunk] ≥ min_similarity
```

- **`raw_dense` is computed from the ORIGINAL question embedding**, not the
  expansion-averaged vector used for ranking. Query expansion may add recall, but it is
  structurally prevented from adding sufficiency.
- The sparse-token fallback is **removed**: incidental lexical overlap (e.g. "2023", "year",
  "named") cannot outvote a dense cosine far below the bar.
- `_sufficient` is a static method with no hidden state; the gate cannot diverge between code
  paths.

### Measured separation (2026-09-23, frozen settings)

| Set | Best raw-query dense cosine |
|---|---|
| Golden questions (18) | min **0.473** |
| Abstention set (6) | max **0.300** |
| ABS-6 (Time Person of the Year) | **0.214** |

The bar (`MIN_SIMILARITY=0.35`) sits in the gap: no golden regression while every out-of-knowledge
question fails closed. The trade-off — a question phrased inside the paper's vocabulary but
abstractly may abstain — is an intentional, documented fail-closed choice
(`docs/failure_analysis.md`).

## Why the gate uses the raw query (ABS-6 post-mortem)

ABS-6 ("Who was named Time Person of the Year for 2023?") was classified `numerical_lookup`
because of the false numeric-like cues in the year; its expansion ("average time", "results
table", "numbers", ...) matched a filename chunk (`results/processing time.txt.`) at cosine 0.45,
pushing the gate open. The raw-query gate measures 0.214 — clearly below bar. Root causes and
fixes are documented in `docs/failure_analysis.md`; regression tests pin the behaviour.

## Caching

- `src/pipeline.py` keeps a disk cache keyed by `(fingerprint, query)`, where the fingerprint
  (`retrieval_config_fingerprint`) hashes every setting that changes retrieval or sufficiency
  (fusion weights, top-k values, `min_similarity`, expansion/routing/rerank toggles, embedder).
- A cache entry persists **the verdict too** (`{sufficient, chunks}`). Before this change a
  cache hit evaluating only its top-k slice could disagree with the full-candidate gate computed
  on a fresh path — which masked the ABS-6 leak in the cache-hit path. Now both paths are
  identical by construction.

## Paper-informed philosophy

The paper's own ablations show more search is not always better: Agent-as-a-Judge peaked
(90.44% alignment) **without** its search module; BM25 (86.06%), Sentence-BERT (87.70%) and
fuzzy search (85.52%) all hurt on simple workspaces. EVIDRA mirrors that: every retrieval
component can be toggled off and compared, routing is conservative, and the cross-encoder
reranker ships disabled until a measured need exists.