# Architecture

This document explains how the EVIDRA system is structured and why each component exists.
See `rag_architecture.png` / `retrieval_workflow.png` for the visual pipelines.

## High-level data flow

```
PDF  →  ingestion (loader → parser → chunker)  →  embeddings  →  FAISS index + BM25
                                                                │
                  query → classification → expansion → hybrid fusion → sufficiency gate
                                                                │  (fail-closed)
                                                                ▼
              grounded answer  ←  claim/evidence verification  ←  LLM generation
                                        │  unsupported claims
                                        ▼
                          exact abstention sentence (no invented content)
```

Everything is wired through `src/pipeline.py`, a single orchestrator reused by the CLI, the
FastAPI server and the Streamlit UI.

## Ingestion (`src/ingestion/`)

| Module | Responsibility |
|---|---|
| `loader.py` | Opens the PDF with PyMuPDF (fallback: pypdf). Emits one `PageDocument` per page with page number preserved. |
| `parser.py` | Normalises whitespace, de-hyphenates line breaks, strips repeated headers/footers, detects headings (`2.2`, `Appendix K`) and figure/table captions, extracts tables via PyMuPDF `find_tables` (rows joined with `|`), and classifies narrative vs definition units. |
| `chunker.py` | `SemanticChunker` groups units by `(section, page)` so page metadata is never lost, packs narrative text into chunks of ~`chunk_size_tokens` with `chunk_overlap` overlap, and keeps tables whole (splitting long tables at row boundaries while repeating the header row). Each chunk is classified (`table`, `result`, `cost-analysis`, `methodology`, `definition`, `narrative`, ...). |

Chunk metadata: `chunk_id`, `page`, `section`, `chunk_type`, `document_id`, `source`, `text`.

## Retrieval (`src/retrieval/`)

- `embeddings.py` — `Embedder` interface. `SentenceTransformerEmbedder` (local, CPU) is the
  default; embeddings are L2-normalised so FAISS inner-product == cosine similarity.
- `vector_store.py` — `VectorStore` interface: `FAISSVectorStore` (default, persisted) and
  `InMemoryVectorStore` (fallback when FAISS is unavailable).
- `bm25.py` — lexical retrieval with `rank_bm25` (`BM25Okapi`); guarantees exact terms like
  `6.38`, `97.72%`, `cn9o` match.
- `hybrid.py` — `HybridRetriever` fuses dense + sparse:
  - **weighted** (default, `alpha=0.5`): min-max-normalised dense and sparse scores combined as
    `alpha * dense + (1-alpha) * sparse`.
  - **rrf**: Reciprocal Rank Fusion.
  - Applies metadata filters and `type_boost`, then computes `evidence_sufficient` with a
    **fail-closed sufficiency gate** `_sufficient` (staticmethod): at least one candidate chunk
    must clear `min_similarity` on the **original question embedding** (`raw_dense`). Two fixes
    landed here after ABS-6 (an off-topic question leaked): the sparse-token fallback was removed,
    and the gate is evaluated on the raw query — expansion terms are ranking-only and can never
    pull an off-topic query over the bar. See `docs/failure_analysis.md`.
- `reranker.py` — optional `CrossEncoderReranker` (`ms-marco-MiniLM-L-6-v2`) with graceful
  fallback to hybrid ordering. Off by default: the Phase 3 ablation measured no lift
  (`docs/eval_results/ablation/`).
- `query_understanding.py` — rule-based `QueryClassifier` (factual / numerical / table /
  comparison / definition / methodology / multi-hop / jailbreak).
- `query_expansion.py` — controlled query expansion (domain suffixes such as "average cost",
  "table") used only to improve **recall**, never to change the sufficiency decision.
- `routing.py` — `QueryRouter` emits type boosts (table/result/cost-analysis for numeric
  questions, appendix for search-method questions, etc.) and guarantees at least one
  table/result chunk is selected for numeric questions.

## Generation (`src/generation/`)

- `prompts.py` — the strict grounded `SYSTEM_PROMPT` and the user prompt builder that places
  retrieved evidence inside explicit `>>RETRIEVED_EVIDENCE_START<< ... END<<` delimiters so the
  LLM can separate instructions from data.
- `llm.py` — `LLMProvider` interface with `OpenAIProvider`, `GroqProvider`, `GeminiProvider`
  (all OpenAI-compatible) and `LocalExtractiveProvider` (offline, deterministic, extracts and
  quotes evidence sentences — never invents content). Failure modes are first-class:
  - rate-limit / connection errors retry with exponential backoff (tenacity);
  - a **circuit breaker** (`_CircuitState`) opens after 3 consecutive failures for a 30 s cooldown
    and fails fast while open (half-open probe after cooldown); the offline provider never trips
    it;
  - every provider takes `timeout_seconds` (`settings.llm_timeout_seconds`, default 30.0) and
    returns `LLMResult(failed=True, error=...)` instead of fabricating text.
- `citation.py` — `CitationValidator` checks claim sentences against evidence: numeric values in
  the answer must appear in the retrieved chunks, and non-numeric claims require token overlap
  with evidence. Unsupported claims lower grounding confidence.
- `security.py` — `JailbreakGuard` detects prompt-injection attempts (ignore-the-PDF,
  reveal-system-prompt, invent numbers, don't-cite, ...) and keeps the response grounded.

## Validation (`src/validation/claims.py`)

Evidence-centric validation is a distinct stage, not a side effect of generation:

- `ClaimExtractor` splits the answer into bullet/line claims (skipping meta lines like "According
  to the paper:"), segments by `[Page X, Section Y]` markers, and filters debris/non-alpha rows.
- `ClaimVerifier` checks each claim against the retrieved evidence: token overlap, numeric
  grounding (every number claimed must appear in the evidence), and page consistency. Each claim
  gets a `verdict`: **supported** / **low-support** / **unsupported**, with a reason.
- Any `unsupported` claim forces the whole answer to abstain (fail-closed).
- The verdicts are surfaced on `RAGResponse.claims` and rendered in the UI (claim → evidence).

## Orchestration (`src/pipeline.py`)

`RagPipeline.answer()` runs:

1. Empty-query guard
2. Jailbreak assessment
3. Query classification + expansion (ranking-only)
4. Retrieval (dense + BM25 → fusion → rerank → evidence selection) with a query→chunks disk
   cache that **persists the verdict** (`{sufficient, chunks}`), so cache hits and fresh paths
   enforce the same sufficiency gate
5. Fail-closed sufficiency check (raw-query dense cosine; below-bar → exact abstention sentence)
6. Grounded context construction (token-budget aware)
7. LLM generation (system prompt + history + delimited evidence); on `LLMResult.failed` the
   response returns the best supporting evidence verbatim — never invented content
8. Citation validation + claim-by-claim verification; unsupported claims → abstention
9. Structured JSON logging of query, chunk ids, scores, model, tokens, latency, cost, verdicts
   and failures — never secrets.

`PipelineStats` tracks queries, cache hits, insufficient-evidence events, generation failures,
latency, tokens and estimated cost for dashboards.

## Evaluation (`src/evaluation/`)

- `questions.json` — 18 golden questions (id 1–18) with expected answers, required evidence
  phrases, expected pages and key numeric values. Q2 was replaced with a verbatim abstract
  question after the original was shown structurally unanswerable (gold chunk dense 0.119,
  rank 86/139); rationale in `questions.json` notes and `docs/failure_analysis.md`.
- `abstain_questions.json` — 6 out-of-knowledge questions for the abstention harness.
- `adversarial_cases.json` — 3 prompt-injection cases embedded in retrieved chunks.
- `metrics.py` — retrieval metrics (Recall@k, Precision@k, MRR, nDCG@k), generation metrics
  (numeric correctness/recall, citation correctness, faithfulness, completeness, page accuracy),
  computed deterministically without an LLM judge.
- `evaluator.py` — runs every question end-to-end, archives each run under
  `docs/eval_results/<timestamp>/` (`evaluation_results.json`, `claims.jsonl`,
  `abstention_results.json`, `evaluation_report.md`).

## API & UI

- `src/api/main.py` — FastAPI: `/api/query`, `/api/health` (liveness), `/api/ready` (readiness;
  **503 until the pipeline index is loaded**, used by the compose healthcheck), `/api/evidence/{chunk_id}`,
  `/api/stats`, `/api/evaluation`, `/api/ingest`. Length cap, per-IP token-bucket rate limit,
  optional HTTP Basic auth.
- `app/streamlit_app.py` — Chat UI with confidence gauge, claim→evidence verdict cards with
  page jump, abstention banner, retrieval debugger, evaluation runner ("Run evaluation now") and
  archived-run comparison, plus the architecture viewer.

## Failure handling paths

Failure modes are first-class, fail-closed, and measured (`docs/rag_failure_taxonomy.md`):

| Case | Behaviour |
|---|---|
| PDF missing | `--download` script or clear error |
| PDF extraction fails | pypdf fallback |
| Empty query | prompt for a question |
| No relevant chunks / below dense bar | exact abstention sentence (fail-closed) |
| Embeddings unavailable | BM25-only |
| Vector DB unavailable | in-memory fallback |
| LLM API failure / rate limit / timeout | retry with backoff → circuit breaker → evidence-first response, never fabricated content |
| Unsupported / low-support claims | exact abstention sentence (fail-closed) |
| Prompt-injection in retrieved text | neutralized before prompt composition; note appended |
| Context over token budget | retrieve → rerank → budget-aware context |

## Deployment

`Dockerfile` builds a `python:3.11-slim` image that runs as a dedicated non-root user
(`evidra`, uid 10001); `scripts/docker_entrypoint.sh` (invoked via `/bin/bash` so it needs no
executable bit) bootstraps the PDF/index on first start and then runs the API, evaluation or the
Streamlit app. `docker-compose.yml` mounts persistent named volumes `evidra_data`/`evidra_index`
(the non-root user can write to them), health-checks the API with `curl http://localhost:8000/api/ready`
and starts the UI only after the API is healthy. The index/embedding download happens at most
once per volume.