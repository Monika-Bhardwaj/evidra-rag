# EVIDRA — Architecture Proposal (Phase 1 deliverable)

Date: 2026-09-20
Revision: v1 (awaiting approval at the Phase 2 gate)
Companion: `docs/engineering_audit.md` (as-built findings this proposal is based on).

---

## 1. Objective

Repurpose the existing RAG chatbot for "Agent-as-a-Judge: Evaluate Agents with Agents" into **EVIDRA** — a production-grade, evidence-centric retrieval system whose contract is:

> **claim → evidence → source → validation** — every claim is traceable to retrieved evidence, a source citation is verifiable against the document, and if validation fails the system **abstains** rather than inventing an answer.

Nothing in this proposal adds complexity without a measured purpose (brief, "do not over-engineer"). The unit of truth stays the golden-question harness; every phase ships with its own status block (`IMPLEMENTED/TESTED/BENCHMARKED/PARTIALLY_IMPLEMENTED/NOT_IMPLEMENTED/BLOCKED`) and a benchmarked before/after comparison against the Phase 2 baseline (§4 of the audit).

---

## 2. Target architecture (claim → evidence → source → validation)

The current as-built architecture already has the right bones: `pipeline.py` orchestrates retrieval → generation → citation → validation, with a fail-closed abstention path and a single eval harness driving the exact prod code path. EVIDRA adds the *claim-level* layer and hardens security/reliability/reproducibility around that same spine.

```
                        ┌────────────────────────────────────────────────┐
   QUESTION ───────────►│ src/pipeline.py (RagPipeline.answer)           │
                        │  1. security guard — question-level injection  │  EXISTING
                        │  2. query understanding / expansion / routing  │  EXISTING
                        │  3. hybrid retrieval (dense + section@BM25)    │  EXISTING
                        │  4. sufficiency gate → abstain if insufficient │  EXISTING
                        │  5. generation (deterministic extractive; API  │  EXISTING
                        │     optional)                                  │
                        │  6. citation + numeric validation              │  EXISTING
                        └───────────────┬────────────────────────────────┘
                                        │ answer + evidence + scores + debug
                                        ▼
   NEW[Phase 4]  claim extractor (linear, deterministic)  ──► claims[] {text, evidence ids}
   NEW[Phase 4]  claim scorer (verification via evidence,  ──► per-claim confidence 0..1
                 item-by-item, incl. numbers & page)
   NEW[Phase 4]  abstention gate (below-threshold claims    ──► EVIDRA abstain message
                 → decline answer, return evidence only)        (fixed, exact string)
   NEW GUARD     per-claim redaction + answer-level defense ──► adversarial text neutralized
                 (retrieved text = data)                        before it reaches user
                        │
                        ▼
   RESPONSE {answer | abstain, citations, per-claim evidence, confidence, debug}
        │
        ├─► src/api/main.py      /query /evidence /stats + readiness-aware /health  (EXISTING +hardening)
        ├─► app/streamlit_app.py RAG Console + Evaluation tabs get:                (EXISTING +Phase 6)
        │                        evidence preview, per-claim confidence, comparison runway
        └─► scripts/ask.py       CLI with evidence + verdict verbosity               (EXISTING)
```

New/modified system blocks and their phase:

| Block | Phase | Type | Why this exists / why it is *not* over-engineering |
|---|---|---|---|
| `ClaimExtractor` | 4 | NEW | Turns an answer into discrete, checkable claims; deterministic, no LLM dependency. Without it there is no claim-level grounding. |
| `ClaimVerifier` (item-level incl. numeric + page) | 4 | NEW | Scores each claim against the exact cited chunks (reuses `citation.py` machinery). This is the "validation" in claim→evidence→source→**validation**. |
| Abstention gate | 4 | NEW gate on existing INSF path | Same fail-closed philosophy already in `pipeline.py`; now also fires when claims fail verification, per the exact brief message. |
| Answer/claim redaction (security) | 4 | NEW defense in depth | Retrieved text is data; the existing guard covers the question, not the content. Cheapest reliable layer: keep adversarial marker text out of the final answer. |
| `ResultsArchive` / baseline pinning | 2 | NEW small util | Overwrite-in-place of `evaluation_results.json` makes changes unreproducible. This is a ~30-line versioned writer, not a DB. |
| Ablation harness (`scripts/ablate.py`) | 3 | NEW script | The only way to legitimately answer "which retriever/fusion/expansion?" without hunch-driven tuning (see Audit §7). |
| Index metadata (model+dim+hash in index dir) | 3 | NEW small change | Prevents silent index/embedding-model mismatch. |
| Timeouts/retries/backoff in `llm.py` | 7 | NEW on API path | Production reliability for the optional API provider; offline path is unaffected. |
| Metrics/abstention in eval harness | 5 | EXT | Measure what Phase 4 guarantees: abstain-correctness, abstain-precision on out-of-knowledge queries. |

Explicitly **not** proposed (over-engineering guard): Graph RAG, Neo4j/knowledge graphs, Kubernetes, microservices, a separate vector database, LangChain/LlamaIndex, async message queues, multi-tenant auth. If any of these ever becomes necessary it will be because a measured experiment demands it, and it will require a new approval gate.

---

## 3. Phase plan (2 → 7)

Each phase is independently shippable, ends with a status block, and (where it changes behaviour) a measured before/after against the Phase 2 baseline (Audit §4: 15/15, recall@6 0.8944, ndcg@6 0.7081, citation 1.0, page 1.0, offline ~20 ms).

### Phase 2 — Freeze the baseline (reposability + hygiene)
Exit: identical 15/15 + 40 tests on a fresh checkout; drift gone.
- Fix README/`.env.example`/`architecture.md` drift (tests=40, alpha=0.5, k=48/48/5/6), delete the phantom "18-step workflow" claim or replace with the real 8-step `answer()`.
- Delete `page5.txt`; add to `.gitignore`.
- Remove dead code: `APIEmbedder` + phantom `embedder_backend` hook, `ChromaVectorStore` (or add `chromadb` back only if a measured experiment uses it), unused deps (`httpx`, `plotly`, `cachetools`, `python-dotenv`, `pytest-asyncio`).
- Add `ruff` + a minimal `pyproject.toml` (lint + format + `mypy`-lite/`pyright` optional), CI (GitHub Actions: pytest + lint on 3.10/3.11).
- Pin `requirements.txt` to tested versions (commit a lockfile if practical); record embedding model + dim in `data/processed/index/` metadata.
- Make scripts cwd-independent (`scripts/*.py` already fix `sys.path`; drop the `insert(0, ".")` pattern).
- New: `ResultsArchive` — copy `evaluation_results.json` + report to `docs/eval_results/<run_id>/` on each run, and write the §4 table as the committed baseline.
- Regression fixtures: seed tests with weak-case gold pairs (Q5 `p003-0008`, Q6 `p011-0045`, Q16 `p005-0015`) so fusion/expansion regressions fail loudly.
- Verdict: **no retrieval or generation behaviour changes in this phase.**

### Phase 3 — Retrieval / evidence layer (measure, then touch)
Exit: a reproducible ablation run that *justifies* the current retrieval stack or changes it.
- Build `scripts/ablate.py`: sweeps dense-only / bm25-only / hybrid(weighted, α∈{0.3,0.5,0.7}) / hybrid+reranker(offline `Noop` vs `Score` vs cross-encoder-if-available) / routing on-off / expansion on-off, at k∈{4,6,8}, on the golden set + seeded weak cases; per-run metrics from `src/evaluation/metrics.py`.
- Add `recall@cited` / `precision@cited` (precision of evidence actually cited) metrics — Audit §4 shows raw Precision@6 is a poor driver.
- Decide: keep `weighted`, log the decision table; enable cross-encoder only if measured lift justifies the download/runtime (currently `USE_RERANKER=false`).
- Add index metadata + store-readout validation; per-document filtering only if a multi-document experiment is approved.
- Verdict: report → harmlessly keep current config if it wins; change only what the data says wins.

**Phase 3 status: DONE (2026-09-20).** `scripts/ablate.py` runs retrieval sweep + window sweep +
end-to-end passes with `recall@cited` / `precision@cited` (added to `src/evaluation/metrics.py`
as `cited_metrics`, wired through the `Evaluator` row/summary/report). Measured result: `weighted`
α=0.5, routing+expansion on, k=6, rerankers OFF is the best config; both `Score` and `CrossEncoder`
rerankers degrade cited metrics/page/completeness (score drops a pass to 14/15), so the default
flipped to `use_reranker=false`. Full data in
`docs/eval_results/ablation/20260920-185423/REPORT.md` and `docs/engineering_audit.md` §4.x.
No further retrieval changes without a measured experiment.

### Phase 4 — Validation + abstention + security (the EVIDRA core)
Exit: claim-level validation in harness + first security controls.
- `ClaimExtractor` (deterministic, rule+position based on `LocalExtractiveProvider` output; list format), `ClaimVerifier` (per-claim evidence support incl. numeric item match and page match reusing `citation.py`).
- Abstention gate with the exact brief string; extend `Evaluator` with abstain/out-of-knowledge cases so the path is *measured* (Phase 5 hardens the metric).
- Answer-level injection handling: neutralize/redact markers found in retrieved text before composing the answer; keep question-level guard.
- API hardening: input-size caps, optional basic auth + rate limit (backend: simple in-process token bucket — not a new service), readiness-aware `/health` (fail until index present, `api/main.py:67`).
- Verdict: JSONL/CSV claims+verdicts per question; harness shows abstain-correctness on the new cases  and no regression on the 15.

**Phase 4 status: DONE (2026-09-20).** `ClaimExtractor` + `ClaimVerifier` shipped in `src/validation/claims.py` (bullet/multiline + page-marker parsing, debris/PDF-noise filtering, hard-fail verdicts for page-cite mismatch and ungrounded numbers; low-support is recorded but does not trip the gate). The pipeline has three fail-closed exits: retrieval sufficiency gate (now with a `knowledge_boundary_min_sim` floor so sparse-token fallback cannot mark out-of-knowledge queries sufficient), provider-abstention detection, and the post-generation claim gate. Retrieved-text injection markers are neutralized before prompt composition (`src/generation/security.py`). Evaluator logs per-question claims/verdicts to `data/processed/claims.jsonl` and runs 4 out-of-knowledge abstention cases. API hardening shipped (`src/api/rate_limit.py`, auth/caps/readiness in `src/api/main.py`). Benchmarked on the golden set: **15/15 pass (no regression)** with baseline metrics unchanged and **abstain-correctness 4/4 (1.0), exact-string rate 1.0**; 180/180 golden claims verdicts = `supported` (archive `docs/eval_results/20260920-193334`). Remaining Phase 5 scope: abstain-precision/recall metrics and generative-provider claim hardening.

### Phase 5 — Evaluation / ablation lab
Exit: expanded, honest harness.
- Add ids 1–3 questions (user said "may be appended") + ≥5 out-of-knowledge abstention cases + a few adversarial-chunk cases (injection-in-retrieved-content fixture).
- Metrics: abstain-precision, abstain-recall, claim faithfulness (now meaningful only when a generative provider is on), retrieval K-curves.
- Wire harness into CI as a nightly/PR regression gate (fast subset locally, full set in CI).
- Verdict: full run → a committed `docs/eval_results/<run>/` archive.

### Phase 6 — UI / evidence console
Exit: users can see *why* an answer was returned (or declined).
- RAG Console tab: evidence preview per cited chunk (page/section/text + scores), per-claim → evidence mapping, confidence gauge, jump-to-page.
- Evaluation tab: run harness from the UI, show per-question rows + the run archive comparisons; Architecture tab updated with EVIDRA dataflow.
- Chat tab: show abstention reason inline; keep stream of key-value detail consistent with v1 metrics (no metric regressions).
- Verdict: UI scrapes `evaluation_results.json` fields it already renders (no schema break).

### Phase 7 — Hardening + documentation
Exit: operable, documented system.
- `llm.py`: timeouts, retries with backoff, circuit breaker for the API path (offline provider path unaffected).
- Docker: non-root user, readiness-gated healthcheck, compose `api` service using the entrypoint (currently bypasses index build), pin container Python to 3.10 or bless 3.11 explicitly in README.
- README rewritten for EVIDRA (claim→evidence→source→validation narrative, statuses per feature, run+eval+ablate+UI recipes); `docs/architecture.md` refreshed to match as-built + EVIDRA blocks.
- Verdict: full end-to-end smoke (ingest→index→query→eval→UI) from clean clone.

---

## 4. Design decisions frozen at the Phase 2 gate

1. **Determinism-by-default stays.** Offline extractive provider remains the reference path; API providers are opt-in and their faithfulness is only trusted after Phase 5 remeasurement.
2. **Fail-closed abstention string is fixed** and unit-tested: `I could not find sufficient evidence for this answer in the provided document.`
3. **Golden set + index artifacts are the source of truth**; `data/processed/*` stays gitignored, but every meaning-changing run is archived under `docs/eval_results/` and the baseline table is pinned.
4. **Weighted fusion remains the default** unless Phase 3 ablation overrides it with data.
5. **No new heavy dependencies** for Phases 2–4 (ruff/CI/archive are tooling, not runtime deps).

---

## 5. Out of scope (explicit)

- Multi-document / arbitrary-corpus routing (single-PDF corpus today; revisit only with a measured need).
- Generative LLM as the default answerer; knowledge-graph/graph-RAG; vector-DB migration; async/microservice architecture; tenant/platform auth.
- Any change justified by aesthetics rather than a measured defect.

---

## 6. Approval gate

Awaiting approval to begin **Phase 2 (baseline)** as described above. Approval confirms:
- the target architecture and phase order,
- the frozen decisions in §4,
- that Phase 3 retrieval changes will be driven by the ablation harness rather than intuition.