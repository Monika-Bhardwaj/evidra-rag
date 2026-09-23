# RAG Failure Taxonomy

How EVIDRA can fail, how each failure is detected, what it does, and the measured evidence that
the behaviour holds. Every entry is fail-closed: the system prefers "I don't know" over a wrong
confident answer.

## A. Retrieval failures

| Failure | Detection | Behaviour | Evidence |
|---|---|---|---|
| **Out-of-knowledge query** (not answerable from the PDF) | raw-query dense cosine below `min_similarity=0.35` | Exact abstention sentence | Abstention set 6/6 exact-string 1.0; golden min cosine 0.473 vs abstain max 0.300 |
| **Lexical phantom** (sparse overlap without semantic support) | sparse-only candidate; gate ignores sparse scores entirely after the fallback removal | Same as above | ABS-6 leaked at 0.45 in 2026-09-23T14:59 run; regression-test: `test_low_dense_cosine_fails_closed_despite_sparse_overlap` |
| **Expansion leakage** (expansion terms lift an off-topic query) | gate always uses the original query embedding | Same as above | ABS-6 (expanded 0.45 → raw 0.214); `test_sufficiency_gate_uses_raw_query_cosine_not_expanded` |
| **Cache/top-k divergence** | cache stores the verdict `{sufficient, chunks}`; `_sufficient` is a staticmethod | Cache-hit path identical to fresh path | `test_sufficiency_must_use_full_candidate_set_not_topk_slice` |
| **Gold chunk ranked low** | Recall@6 / nDCG@6 | Retrieval metrics report it; no silent fix | Recall@6 0.8148, MRR 0.7398 (20260923-153131) |
| Embeddings unavailable | exception at embed time | BM25-only path | code path; no failure observed in runs |
| Vector DB unavailable | exception at load | in-memory FAISS-equivalent fallback | code path |

## B. Generation failures

| Failure | Detection | Behaviour | Evidence |
|---|---|---|---|
| Provider error / outage | `LLMResult.failed=True` | Evidence-first answer, "(Generation failed; no invented content returned.)" | unit tests; pipeline `_generate` |
| Provider hang | per-request `timeout_seconds` (30 s default) | Same as above | `test_provider_breaker_...` plumbing asserts timeout reaches the client |
| Repeated upstream failures | circuit breaker opens after 3 consecutive failures, 30 s cooldown, half-open probe | Fail fast while open; probe after cooldown | `tests/test_llm.py` (5 tests cover open/short-circuit/half-open/reset) |
| **Fabricated numbers** | `CitationValidator` numeric check + claim numeric grounding | Unsupported → abstention | Numeric recall 1.0 on the golden set |
| **Unsupported claims** | `ClaimVerifier` per-claim verdicts (any `unsupported` → abstain) | Exact abstention sentence | Faithfulness 1.0 on the golden set |

## C. Input/security failures

| Failure | Detection | Behaviour | Evidence |
|---|---|---|---|
| Prompt injection in retrieved chunks | neutralization before prompt composition | Injected token never appears in the answer | Adversarial 3/3 (INJ-1..3) |
| Direct jailbreak prompt | `JailbreakGuard` | PDF-grounded answer + neutralization note | `tests/test_security.py` |
| History injection / prompt smuggling | bounded history + delimited evidence | treated as data | same suite |

## D. Platform failures

| Failure | Detection | Behaviour | Evidence |
|---|---|---|---|
| PDF missing | file check in entrypoint/scripts | `--download` from arXiv, else clear error | script docs |
| PDF extraction fails on a page | PyMuPDF exception | pypdf fallback per page, failures logged | `src/ingestion/loader.py` |
| API readiness | `/api/ready` returns 503 until index loaded | compose gates UI start on healthy API | `tests/test_api.py` |
| Non-root permissions | named volumes owned by `evidra` (uid 10001) | bootstrap PDF/index inside volumes once | Dockerfile/compose |

## The invariant

Across all classes the invariant is single: **when in doubt, emit the exact abstention sentence
— never a fabricated number, page, or answer.** The evaluation suite pins this with exact-string
checks, not fuzzy "did it refuse somehow" checks.