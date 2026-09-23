# Engineering Audit — evidra-rag

Date: 2026-09-20
Audit basis: full source inspection (`src/`, `scripts/`, `app/`, `tests/`, `docs/`, docker files), `pytest -q` (40 passed), CLI/API smoke tests, and the measured baseline in `data/processed/evaluation_results.json`.
Status legend: `IMPLEMENTED`, `TESTED`, `BENCHMARKED`, `PARTIALLY_IMPLEMENTED`, `NOT_IMPLEMENTED`, `BLOCKED`.

---

## 1. Executive summary

`evidra-rag` is a compact, well-structured Python 3.10 RAG system over a single corpus (arXiv 2410.10934, "Agent-as-a-Judge"). It is modular (ingestion / retrieval / generation / evaluation / api), offline-runnable (deterministic extractive provider), passes all 15 golden questions (100%) and all 40 unit tests. The runtime engine and the evaluation harness use the **same** code path — a strong foundation for EVIDRA.

The main gaps are **not in search accuracy** (strong on the current golden set) but in:

- (a) documentation/config drift and committed junk,
- (b) dead or phantom code paths,
- (c) weak security posture once a real "LLM-as-data" pipeline and a network-facing API are in scope,
- (d) an over-narrow evaluation harness (no abstention cases, no reproducible pre-change snapshot, no ablation harness for retrieval decisions),
- (e) reproducibility (unpinned deps, cwd-dependent scripts, no CI/lint/typecheck),
- (f) no multi-document or concurrency story.

Everything in this audit was verified against the code; nothing is assumed or extrapolated.

---

## 2. Current architecture (as-built)

```
PDF ──► src/ingestion/loader.py      pymupdf → pypdf fallback            [TESTED]
        src/ingestion/parser.py      span/font heading detection,        [TESTED]
                                      repeated-line stripping, tables,
                                      definition units, arXiv skips
        src/ingestion/chunker.py     SemanticChunker 500 tok / 15%       [TESTED]
                                      overlap; section+page grouping;
                                      chunk typing (definition / cost /
                                      result / table / narrative ...)
             ▼  data/processed/{chunks,pages,ingestion_stats}.json      (gitignored)
Index ── scripts/build_index.py
        src/retrieval/embeddings.py   sentence-transformers (local)      [TESTED]
        src/retrieval/vector_store.py FAISS (L2-normalized)              [TESTED]
        src/retrieval/bm25.py         section-aware BM25                 [TESTED]
        build_gold_evidence_map → data/processed/gold_evidence.json

Query ── src/pipeline.py (RagPipeline.answer)
        src/generation/security.py      injection guard (question-level) [TESTED]
        src/retrieval/query_understanding.py  intent classifier          [TESTED]
        src/retrieval/query_expansion.py     suffix pools, r<N> variant  [TESTED]
        src/retrieval/routing.py            chunk_types + type_boost     [TESTED]
        src/retrieval/hybrid.py         weighted fusion + sufficiency    [TESTED]
        src/cache.py                    DiskCache (sha256, atomic, TTL)  [TESTED]
        src/retrieval/reranker.py       Noop / Score / CrossEncoder      [TESTED]
        src/generation/prompts.py       grounded prompt + delimiters     [TESTED]
        src/generation/llm.py           OpenAI API | LocalExtractive     [TESTED]
        src/generation/citation.py      cite + validate numeric claims   [TESTED]
        answer + citations + sources + debug + abstention
             ▼
        src/api/main.py (FastAPI: /health /query /evidence /stats /evaluation /ingest)  [TESTED]
        app/streamlit_app.py (Chat / RAG Console / Evaluation / Architecture)           [IMPLEMENTED]
        scripts/ask.py (CLI) · scripts/evaluate.py → docs/evaluation_report.md + plots
```

### Key design facts

- **Determinism by default**: with no API keys, `LLM_PROVIDER=auto` resolves to `LocalExtractiveProvider`, a deterministic extractor. Offline answers are reproducible.
- **Fail-closed abstention exists today** (`src/pipeline.py:181-194`): when retrieval returns no evidence or the sufficiency gate fails, the system returns `INSF_MSG = "I could not find sufficient evidence for this answer in the provided document."` with empty citations, confidence 0.0. `[BENCHMARKED at runtime; NOT evaluated in the harness]`
- **Single pipeline for prod and eval**: `Evaluator` drives `RagPipeline.answer` per question, so evaluation measures the same path users hit. Correct foundation for EVIDRA.
- **Evidence metadata is rich**: every chunk carries `chunk_id`, `page`, `section`, `chunk_type`, `document_id`, `source`; retrieval results carry `dense_score` / `sparse_score` / `hybrid_score` / `rerank_score`; `RetrievalDebug` exposes per-stage hits (`dense_hits`, `sparse_hits`, `fused_hits`, `reranked_hits`, `filtered_out_ids`). Everything needed for an evidence console and an ablation harness already exists.
- **Cache keying is correct**: retrieval cache key includes query, expansions, `hybrid_alpha`, `hybrid_method`, `dense_top_k`, and an index signature — stale-index invalidation is handled.

---

## 3. Capabilities

| Capability | Status | Notes |
|---|---|---|
| PDF ingestion with heading/table/definition structure | TESTED | 139 chunks; `tests/test_ingestion.py`, `tests/test_chunking.py` |
| Section+page-aware chunking (500 tok, 15% overlap) | TESTED | Tables chunked with repeated header |
| Hybrid retrieval (dense + BM25) with sufficiency gate | BENCHMARKED | recall@6 0.8944, mrr 0.7989, ndcg@6 0.7081 |
| Query understanding + expansion + routing | TESTED | route = allowed chunk types + type boost per intent |
| Numeric claim + citation validation | TESTED | citation_accuracy 1.0, faithfulness 1.0, numeric_recall 1.0 |
| Prompt-injection guard (question-level) | TESTED | 8 payloads in `tests/test_security.py` |
| Fail-closed abstention | PARTIALLY_IMPLEMENTED | runtime path exists; not exercised by eval set |
| Offline deterministic provider | TESTED | no-API-key fallback |
| Retrieval disk cache (TTL, atomic replace) | TESTED | keyed on Q, expansions, alpha, method, k, index signature |
| CLI / REST API / Streamlit UI | TESTED | TestClient smoke (health, query 200, evaluation 15/15); app imports clean |
| Report + plots generation | TESTED | `docs/evaluation_report.md` + 3 PNGs |
| Golden-set evaluation (15 Qs) | BENCHMARKED | **15/15 pass** (see §4) |

---

## 4. Measured baseline (Phase 2 reference)

Reproduced from `data/processed/evaluation_results.json` (+ `docs/evaluation_report.md`). Run conditions: offline-extractive provider, `USE_RERANKER=false`, `hybrid_alpha=0.5`, `dense_top_k=48`, `bm25_top_k=48`, `final_top_k=6`, `min_similarity=0.35`, `hybrid_method=weighted`, `rank_bm25` section-aware, evidence set = the golden questions (ids 4-18) in `src/evaluation/questions.json`.

| Metric | Value |
|---|---|
| Passed | **15 / 15 (100%)** |
| Recall@6 | 0.8944 |
| Precision@6 | 0.2556 |
| MRR | 0.7989 |
| nDCG@6 | 0.7081 |
| Numeric recall | 1.0000 |
| Completeness | 0.9833 |
| Citation accuracy | 1.0000 |
| Faithfulness rate | 1.0000 |
| Page accuracy | 1.0000 |
| Avg latency (offline) | 20.1 ms |
| Provider | offline-extractive |

> **Reproducibility caveat (measured 2026-09-20):** the baseline numbers above are reproducing
> byte-for-byte only when the *same environment* is used — specifically `USE_RERANKER=false`
> (the cross-encoder was not part of the original benchmark). Running the same harness with the
> `USE_RERANKER=true` (cached `ms-marco-MiniLM-L-6-v2`) re-orders evidence and drops the
> summary to `completeness 0.9167`, `page_accuracy 0.9333` (pass rate stays 15/15). Never compare
> runs across different env configs; the ablation harness (Phase 3) must log the full settings
> with each run.

### 4.x Phase 3 result (measured 2026-09-20): rerankers and fusion

The ablation harness `scripts/ablate.py` swept fusion strategy × routing × expansion × window size
on the golden set (binary gold relevance), then ran the top configs end-to-end. Full report:
`docs/eval_results/ablation/20260920-185423/REPORT.md`.

Retrieval sweep (k=6), best rows:

| variant | routing | expansion | hit | R@6 | P@6 | MRR | nDCG@6 |
|---|---|---|---|---|---|---|---|
| `weighted-0.5` | on | on | **1.0000** | **0.8944** | 0.2556 | 0.7989 | **0.8033** |
| `weighted-0.5` | off | on | 1.0000 | 0.8944 | 0.2556 | 0.8044 | 0.8026 |
| `weighted-0.3` | on | off | 0.9333 | 0.8500 | 0.2556 | 0.8500 | 0.8137 |
| `rrf` | on | on | 0.7333 | 0.5167 | 0.1667 | 0.6300 | 0.5249 |

Window sweep for `weighted-0.5` (routing+expansion on): k=4 → hit 0.867, k=6 → hit **1.0000**,
k=8 → hit 1.0000 (R@8 0.9056, but P@8 0.2000).

End-to-end passes (recall@6 fixed by retrieval; differences are ranking/generation):

| variant | pass | citedR | citedP | nDCG@6 | page | completeness |
|---|---|---|---|---|---|---|
| default, rerank off | **15/15** | **0.8944** | **0.3067** | 0.7081 | 1.0000 | **0.9833** |
| default, score rerank | 14/15 | 0.8778 | 0.2933 | 0.7081 | 1.0000 | 0.9500 |
| default, cross-encoder | 15/15 | 0.8278 | 0.2933 | 0.7081 | 0.9333 | 0.9167 |

**Decision (data-driven):** keep `weighted` α=0.5 with query expansion+routing on, window k=6,
and **rerankers OFF by default**. Both `ScoreReranker` and `CrossEncoderReranker` degrade
cited-evidence recall/precision and answer completeness; `score` even drops a pass (14/15) and
`cross` drops page accuracy to 0.9333. Consequently `src/config.py` default changed to
`use_reranker=false` (with opt-in `reranker_kind=cross|score`). RRF is rejected as it relies on
rank-only signals that collapse under routing boosts (hit 0.7333).

Interpretation:

- **Recall@6 is 0.89, not 1.0** — for a few questions the gold chunk is retrieved but not ranked inside the final evidence window (e.g., Q5/`p003-0008`, Q16/`p005-0015` are sparse-strong/dense-weak and partly depend on section-aware BM25).
- **Precision@6 is low (0.2556) by construction**: the metric counts every one of the top-6 chunks against the gold set, and citation pull-in co-pages/chunk-types inflate the denominator. On *used*-evidence basis, precision is effectively 1.0 (citation accuracy and page accuracy both 1.0). The ablation harness (§E.3) must define a "precision of cited evidence" metric rather than tuning raw Precision@6 into a local optimum.

### 4.y Phase 4 result (measured 2026-09-20): validation + abstention + security

Phase 4 shipped the EVIDRA core: `ClaimExtractor`/`ClaimVerifier`
(`src/validation/claims.py`), fail-closed abstention exits in the pipeline, per-claim
redaction of retrieved-text injection markers (`src/generation/security.py`,
`pipeline._sanitize_prompt_evidence`), API hardening, and an abstention harness
(`src/evaluation/abstain_questions.json`, ABS-1..4).

**Verifier semantics** (deterministic, tuned to the offline-extractive provider's output):
claims are parsed as bullet units with embedded-newline collapse and PDF-noise
(debrise/mojibake/figure-caption) filtering. A claim is hard-`unsupported` only on a real
fabrication signature: a cited page absent from the final evidence, or numbers with no
trace in the evidence set. Low token overlap alone is recorded as `low-support` and does
not trip the gate — the earlier all-claims-strict gate over-abstained (golden pass dropped
to 2/15) and was corrected.

**Fail-closed exits now measurable:**
1. retrieval gate — fail-closed dense gate: a query is supported only when the
   cosine between the ORIGINAL (unexpanded) question embedding and some retrieved
   chunk clears `min_similarity`. The `knowledge_boundary_min_sim` (0.30) sparse-token
   fallback introduced in Phase 4 was **removed on 2026-09-23** and the gate was moved
   to the raw-query embedding: ABS-6 leaked through both (token overlap on "2023"/"year"
   and `numerical_lookup` expansion terms matching a filename chunk at 0.45). See
   `docs/failure_analysis.md` for the full post-mortem; every golden question clears the
   raw-query bar directly (best cosine ≥ 0.42);
2. provider-abstention — if the provider itself returns the exact abstention sentence
   (`src/pipeline.py`), the response is recorded as `evidence_sufficient=False`
   (previously mis-tagged as a confident answer);
3. post-generation claim gate — `[claim-verification]` abstention on hard-unsupported claims.

Golden-set benchmark (archive `docs/eval_results/20260920-193334`), offline-extractive:

| metric | after Phase 4 | baseline (Phase 2/3) |
|---|---|---|
| pass | **15/15** (1.0) | 15/15 |
| recall@6 | 0.8944 | 0.8944 |
| precision@6 | 0.2556 | 0.2556 |
| MRR / nDCG@6 | 0.7989 / 0.7081 | 0.7989 / 0.7081 |
| numeric recall / completeness | 1.0000 / 0.9833 | 1.0000 / 0.9833 |
| citation / page | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| abstain-correctness (4 cases) | **4/4 (1.0)**, exact-string 1.0 | n/a |

Claims log: `data/processed/claims.jsonl` — 180/180 golden claims verdict = `supported`.
Test suite: 65 pytest green, ruff clean. Phase 5 hardens the abstraction metric
(abstain-precision/recall) and generative-provider claim scoring.

---

## 5. Deficiencies and technical debt

### 5.1 Documentation / config drift (repo is not self-consistent)

- `README.md:73` claims "25 automated tests"; **actual: 40** (`pytest -q`).
- `README.md:134,136`, `docs/architecture.md:35`, `.env.example:49-53` document `hybrid_alpha=0.7`, `final_top_k=4`, `dense/bm25_top_k=10`; **code defaults are** `hybrid_alpha=0.5`, `final_top_k=6`, `dense/bm25_top_k=48` (`src/config.py:38-42`). A user without `.env` gets a materially different (better, measured) configuration than one who copies `.env.example`.
- `README.md:9` promises an "18-step engineering workflow"; no such list exists anywhere. `docs/architecture.md:68-79` documents an 8-step `answer()`.
- `README.md:105` still names the project "rag-devai-chatbot". Project dir/header is "evidra-rag" / "RAG DevAI Chatbot".
- Resolution target: Phase 2 (baseline) — per EVIDRA rule §41, documentation must not fabricate or lag the code.

### 5.2 Committed junk

- `page5.txt` at repo root (2266 B) — committed UTF-16LE PowerShell error traceback; referenced by nothing, not gitignored. Remove and add to `.gitignore`.

### 5.3 Dead / phantom code

- `APIEmbedder` (`src/retrieval/embeddings.py:54-82`) is **unreachable**: selected only via `getattr(settings, "embedder_backend", "local")`, and `Settings` has no such field (config uses `extra="ignore"`). The API-path is dead unless a real config field is added.
- `ChromaVectorStore` / `build_vector_store` (`src/retrieval/vector_store.py:136-201`) is never called, and `chromadb` is **missing from requirements.txt** → latent broken import.
- `ENABLE_ANSWER_CACHE` (`src/config.py:53`, `.env.example:71`) is read by nothing; answer caching is unimplemented (only `enable_retrieval_cache` is used, `src/pipeline.py:246`).
- Unused dependencies in `requirements.txt`: `httpx`, `plotly`, `cachetools`, `python-dotenv` (direct import count 0), `pytest-asyncio` (no async tests).

### 5.4 Tooling not present

- No lint config (`pyproject.toml` / `ruff.toml` / `setup.cfg` / `.flake8`), no type-check config; only `pytest.ini` exists (with a single `regression` marker).
- No CI. No dependency lockfile (`requirements.txt` pins `>=` only). No pinned embedding model snapshot in the artifact path.
- Scripts `ask.py`, `evaluate.py`, `build_index.py`, `generate_diagrams.py` do `sys.path.insert(0, ".")` → **only runnable from the repo root**. (`scripts/ingest.py` is robust, and `scripts/docker_entrypoint.sh` is repo-root-relative.)

### 5.5 Deployment gaps

- `Dockerfile:1` uses `python:3.11-slim` though README targets Python 3.10.
- `docker-compose.yml` `api` service overrides the CMD with `uvicorn` directly (`:4`), **bypassing** `scripts/docker_entrypoint.sh` index-build logic; the app service may carry it via the shared volume, but the ordering is implicit.
- Runs as **root** in the container. No non-root user.
- `/api/health` returns 200 even while `_pipeline is None` (`src/api/main.py:67-69`) — a health check can pass before the index is ready; a following `/api/query` would fail. Health should reflect index readiness.
- Streamlit UI loads the pipeline **in-process** (`@st.cache_resource`), not via the API — fine today; in a deployed stack, UI and API would run separate pipelines.

---

## 6. Security risks (their motto: retrieved text = data)

1. **Retrieved content is untrusted and only lightly defended.** The injection guard (`src/generation/security.py`) inspects the *question*; markers are wrapped around evidence in the prompt (`src/generation/prompts.py`); a neutralization note is appended to the answer (`src/pipeline.py:205-206`). Phase 4 added **per-claim redaction** of retrieved-text markers before prompt composition (`pipeline._sanitize_prompt_evidence` → `neutralize_retrieved_text`) and claim-level verification is a fail-closed second line of defense (risk 5). `[IMPLEMENTED - Phase 4]`
2. **No auth / rate-limit / input-size caps** on the FastAPI endpoints. Once network-exposed: availability and abuse risk. Phase 4 shipped optional Basic auth, an in-process token-bucket rate limit, question length caps, and a readiness-aware `/health`. `[IMPLEMENTED - Phase 4, src/api/]`
3. **Full question text is logged** in the answer event (`src/pipeline.py:386`, `_log_event`) — a privacy exposure if queries contain user data; should be configurable/redactable. `[PARTIALLY_IMPLEMENTED]`
4. **No timeouts/retries/backoff** around the (future/optional) API LLM call (`src/generation/llm.py` passes through to `openai`), and no circuit breaker — Phase 7 production-reliability scope.
5. EVIDRA's claim/confidence layer **must** assume guard-bypass: claims sourced from adversarial text must be scored against citation-level verification, and the abstention path must fire when scores are low. Phase 4 implemented the claim verifier and three fail-closed abstention exits; adversarial injected *questions* are still not adversarially eval'd (Phase 5 fixture). `[IMPLEMENTED - Phase 4; adversarial eval Phase 5]`

---

## 7. Retrieval weaknesses (pre-ablation)

- **Fusion is a fixed heuristic.** `hybrid._weighted_rank` minmax-normalizes dense and sparse across the fused candidates and sums at `α=0.5` (weighted). RRF was tested and rejected: better for Q16, worse for Q5 (gold chunk `p003-0008` ranked ~34 under RRF) → final is `weighted`. **This decision is not recorded in the repo**; it should become a regression fixture. Known weak/edge cases to freeze: Q5 `p003-0008`, Q6 cost tables `p011-0045`, Q16 `p005-0015`, Q12 alignment pair `p010-0037`/`p039-0127`.
- **Sufficiency gate is thin but effective**: `min_similarity` (0.35) on max dense cosine, plus a sparse-token fallback (≥3 shared non-stopword tokens) for dense-weak gold chunks. Phase 4 added the `knowledge_boundary_min_sim` (0.30) floor: the sparse fallback cannot mark a query sufficient when the best dense similarity is below the floor (out-of-knowledge cases measured ≤ 0.284 vs golden ≥ 0.451).
- **BM25 relies on section-led indexing** (`f"{section} {chunk_type} {text}"`). Ordering of sparse hits can be sensitive to section naming — verify in ablation.
- **Single embedding model, unversioned** in the artifact path (`data/processed/index/`); an embedding upgrade silently invalidates old indexes. Should record model name/dim in index metadata.
- **No multi-document filtering** at retrieval (chunks carry `document_id`, but `hybrid.retrieve` does not filter by it). Multi-corpus support is a Phase 3/7 decision, gated on measured need (EVIDRA over-engineering rule).

---

## 8. Evaluation weaknesses (self-inflicted, high priority for EVIDRA)

1. **Golden set is small and self-derived**: 15 questions extracted from the paper itself (ids 4-18; ids 1-3 are "may be appended later"). Numeric-heavy, all answerable — good for precision, bad for robustness.
2. **Abstention/insufficient-evidence cases**: Phase 4 added 4 out-of-knowledge cases (`src/evaluation/abstain_questions.json`) - measured abstain-correctness 4/4, exact-string 1.0. Phase 5 extends to ≥5 cases and adds abstain-precision/recall.
3. **No reproducibility snapshot**: `evaluation_results.json` is overwritten on every run; there is no "pin the current baseline and compare" mechanism. (Phase 2 will add a results/ archive.)
4. **No per-method ablation harness** in the repo: dense-only / bm25-only / hybrid + reranker / routing off / expansion off comparisons are not scripted. The prior tuning was done interactively; the decisions must be reproducible.
5. **Two `@pytest.mark.regression` tests skip** when `gold_evidence.json`/`chunks.json` are absent — artifacts are gitignored, so a fresh clone has a quiet gap.
6. **Faithfulness with a real LLM is unmeasured**: current 1.0 faithfulness is for the deterministic extractive provider. When a generative provider is used, faithfulness must be recomputed (and a claim-level scorer added).

---

## 9. Scalability / operability

- FAISS index is in-memory and tiny (139 chunks); CPU build is sub-second. No sharding/persistence concerns for the single-doc corpus.
- Endpoints are synchronous; model calls block. No queueing or async path (fine for the offline provider).
- Latency is dominated by generation; offline extractive is ~20 ms. Retrieval is far from any budget.
- Multi-doc / multi-corpus would require: per-doc filtering at retrieval, index partitioning, and an ingest API for new documents (an `/api/ingest` route exists but assumes the single configured PDF). **Gated by measured need** — no distributed store, Graph RAG, or microservices without an experiment that demands them (phase gates).

---

## 10. Reproducibility

- PDF is versioned in the repo workflow via arXiv 2410.10934 (downloadable with `scripts/ingest.py --download`, URL at `scripts/ingest.py:23`); `data/raw/Agent-as-a-Judge.pdf` is present.
- Ingestion is deterministic given the same PDF (span-based parse) → same 139 chunks.
- Offline provider answers are deterministic → same 15/15 result, run twice.
- **Gaps**: `requirements.txt` `>=` pins, no lockfile, no model pinning, no CI seed coverage, runtime config defaults differ from `.env.example`, scripts cwd-dependent. Phase 2 fixes these.

---

## A. Adequacy vs the EVIDRA brief

| EVIDRA requirement | Current status | Gap (phase) |
|---|---|---|
| Evidence-centered claim→evidence→source→validation | PARTIALLY_IMPLEMENTED (citations + numeric validation) | claim-level scoring & per-claim grounding (4) |
| Confidence grounded in retrieval, with abstention | PARTIALLY_IMPLEMENTED (sufficiency gate + INSF_MSG) | exercise in harness; finer-grained gates (4) |
| Measurable via harness, no fabricated numbers | BENCHMARKED (15/15 baseline, §4) | baseline pinning + result archive (2) |
| Reproducible & modular | PARTIALLY | lockfile, lint/type, CI, index metadata (2) |
| LLM-as-data security (guard + content defense) | PARTIALLY | per-claim redaction, auth/rate caps (4) |
| Explainability / evidence console | PARTIALLY (RetrievalDebug, RAG Console tab) | evidence preview + confidence UI (6) |
| No over-engineering | SATISFIED today | freeze in proposal; gate each addition |

---

## B. Upgrade path (recommended order, ties to phases)

1. **Phase 2 — freeze the baseline**: fix drift + junk, remove dead code, pin deps/models, add lint/typecheck/CI, archive the §4 numbers, seed regression fixtures (weak cases).
2. **Phase 3 — retrieval/evidence layer**: build the ablation harness (dense-only / bm25-only / hybrid / ±reranker / ±routing / ±expansion), then make retrieval decisions from measurements, not hunches. Add index metadata & per-doc filtering only if a measured experiment calls for it.
3. **Phase 4 — validation + abstention + security**: claim extractor, citation-level groundedness scoring, abstention gate (exercised via harness cases), answer-level guard and per-claim redaction, API auth/rate caps.
4. **Phase 5 — evaluation/ablation**: expand gold set (add ids 1-3 + abstention cases), K-sweep + method-sweep, regression gate in CI.
5. **Phase 6 — UI/evidence console**: surface evidence previews, per-claim confidence, and the ablation/comparison results (reuse the 4 existing tabs).
6. **Phase 7 — hardening & docs**: timeouts/retries/backoff, docker least-privilege + health-gated readiness, EVIDRA README and architecture docs.

---

## C. Known open questions (do not block Phase 2)

- Should ids 1-3 be authored and added to the gold set (Phase 5)? (User brief says they "may be appended later".)
- Is the target deployment a network-visible API, a local Streamlit tool, or both? (Drives Phase 4 auth/caps and Phase 6 UI scope.)
- Will a funded generative LLM be used (OpenAI or local) — or should EVIDRA stay deterministic-by-default? (Drives faithfulness measurement and Phase 7 reliability work.)