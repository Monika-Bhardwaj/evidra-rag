# EVIDRA — Evidence-Centric RAG Assistant

*Agent-as-a-Judge: Evaluate Agents with Agents* (arXiv:2410.10934, ICML 2025), evidence-first.

EVIDRA is a production-shaped, **document-grounded RAG assistant** for a single source of truth —
the paper PDF. It ingests the document, builds a persistent local index, and answers **strictly
from retrieved evidence with traceable citations**. When the evidence does not support an answer,
the system **fails closed**: it returns the exact abstention sentence instead of inventing content.

> Core principle: **claim → evidence → source → validation, in that order.** No evidence, no claim.

Latest measured state (offline-extractive, `docs/eval_results/20260923-153131`):

| Harness | Result |
|---|---|
| Golden questions (18/18) | **100% pass** · Recall@6 **0.8148** · nDCG@6 **0.6949** · citation/page/faithfulness **1.0** |
| Abstention (6/6 out-of-knowledge) | **100%** correct abstention · exact abstention sentence **100%** |
| Injection defense (3/3 adversarial chunks) | **100%** neutralized (injected instructions never followed) |
| Automated test suite | **91 tests pass** · `ruff check` + `ruff format --check` clean |

---

## Why evidence-centric?

RAG systems that answer "confidently and wrong" are worse than systems that say "I don't know".
EVIDRA is engineered around the abstention boundary:

1. **Retrieval** — hybrid dense + BM25 over page-aware semantic chunks.
2. **Sufficiency gate** — fail-closed: evidence is sufficient only if a candidate chunk clears the
   dense bar on the **original (unexpanded) question embedding**. Query expansion is ranking-only
   and can never lift an off-topic query over the bar (ABS-6 post-mortem:
   [`docs/failure_analysis.md`](docs/failure_analysis.md)).
3. **Generation** — the LLM sees only delimited retrieved evidence; no general-knowledge
   supplementation. Offline extractive provider available with zero API cost.
4. **Grounding validation** — citations checked against retrieved chunks, every numeric claim must
   appear in the evidence, and every answer claim is verified claim-by-claim (supported /
   low-support / unsupported). Unsupported claims → abstention.
5. **Security** — retrieved text is **data, not instructions**; prompt-injection attempts are
   neutralized before prompt composition.

---

## Quick start (local)

Requires **Python 3.10 or 3.11**.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

# 1. Ingestion + index (one-time; downloads the ~90 MB embedder on first run)
python scripts/ingest.py --download
python scripts/ingest.py
python scripts/build_index.py

# 2. Run
streamlit run app/streamlit_app.py          # conversational UI + evidence console
uvicorn src.api.main:app --port 8000        # FastAPI backend
python scripts/ask.py "What is the average cost and average time for OpenHands?"
```

Run the evaluation harness:

```bash
python scripts/evaluate.py
```

Writes `docs/evaluation_report.md` and an archived run under `docs/eval_results/<timestamp>/`
(claims, per-question results, abstention and adversarial verdicts for reproducible comparison).

---

## Docker (non-root, ready-gated)

```bash
docker compose up --build
```

- `api` service loads the pipeline before it is considered healthy (`/api/ready` returns **503**
  until the index is loaded; compose gates `app` on `api` being healthy).
- Everything runs as a dedicated **non-root user** (`evidra`, uid 10001).
- The PDF and index are written to persistent **named volumes** `evidra_data` / `evidra_index`, so
  the first boot downloads the paper and builds the index once; later boots start instantly.
- LLM keys are read from the container environment, never baked into the image.

---

## Configuration

Copy `.env.example` to `.env`. Every value is optional for the fully-offline path.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `auto` | `auto` \| `openai` \| `gemini` \| `groq` \| `local` |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | — / `gpt-4o-mini` | OpenAI-compatible chat completions |
| `GEMINI_API_KEY` / `GROQ_API_KEY` | — | alternate OpenAI-compatible providers |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` | `0.0` / `512` | generation behaviour |
| `LLM_TIMEOUT_SECONDS` | `30.0` | per-request provider timeout |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | local embedder |
| `DENSE_TOP_K` / `BM25_TOP_K` | `48` / `48` | candidate depth per retriever |
| `RERANK_TOP_K` / `FINAL_TOP_K` | `5` / `6` | rerank pool / evidence sent to the LLM |
| `HYBRID_ALPHA` / `HYBRID_METHOD` | `0.5` / `weighted` | dense weight / fusion (`weighted` \| `rrf`) |
| `MIN_SIMILARITY` | `0.35` | fail-closed sufficiency bar (raw-query dense cosine) |
| `USE_QUERY_EXPANSION` / `USE_QUERY_ROUTING` | `true` / `true` | ranking-only expansion; conservative routing |
| `USE_RERANKER` / `RERANKER_KIND` | `false` / `cross` | **off by default** — the Phase 3 ablation measured no lift (see `docs/eval_results/ablation/`) |
| `CONTEXT_MAX_TOKENS` | `4000` | token budget for the evidence context |
| `API_MAX_QUESTION_CHARS` / `API_RATE_LIMIT_PER_MIN` / `API_AUTH_USERNAME/PASSWORD` | `4000` / `60` / empty | API hardening (length cap, token-bucket rate limit, HTTP Basic auth) |

No API key → the system runs fully offline with the **extractive evidence generator**
(deterministic grounded quotes). Add a key for free-form synthesized answers; the per-request
timeout and the **circuit breaker** (3 consecutive failures → 30 s cooldown, half-open probe)
protect the request path against a degraded upstream.

---

## System architecture

![RAG architecture](docs/rag_architecture.png)

```
PDF → loader → parser → semantic chunker → embeddings → FAISS + BM25 index
                                                        │
USER QUERY → classification → ranking-only expansion → dense + BM25 → fusion → top-k
        │                       │
        └──► sufficiency gate (raw-query dense cosine, fail-closed)
                    │ sufficient
                    ▼
        context construction → grounded LLM → claim/evidence verification → citations
                    │ insufficient / unsupported claims / LLM failure
                    ▼
        exact abstention sentence (no invented content)
```

Component map:

| Component | What it does |
|---|---|
| `src/ingestion/` | PyMuPDF extraction (pypdf fallback), heading/table detection, page- and section-aware semantic chunking; tables kept whole |
| `src/retrieval/` | Sentence-transformer embeddings + FAISS, BM25, hybrid fusion, optional reranking, query understanding/expansion/routing |
| `src/generation/` | Strict grounded prompts, provider abstraction (+ circuit breaker), citation validator, injection guard |
| `src/validation/claims.py` | Claim extractor + verifier: every answer claim gets a supported/low-support/unsupported verdict |
| `src/pipeline.py` | Single orchestrator shared by CLI, API and UI |
| `src/api/main.py` | FastAPI: `/api/query`, `/api/health` (liveness), `/api/ready` (readiness, 503 until loaded), `/api/stats`, `/api/evaluation`, `/api/ingest` |
| `app/streamlit_app.py` | Chat UI with confidence gauge, claim→evidence verdicts, evidence cards with page jump, eval runner + archive comparison |

See [`docs/architecture.md`](docs/architecture.md) for design rationale and supporting documents:
[retrieval](docs/retrieval_design.md), [validation](docs/evaluation_methodology.md),
[security](docs/security_model.md), [failure taxonomy](docs/rag_failure_taxonomy.md),
[reproducibility](docs/reproducibility.md), [future work](docs/future_work.md).

---

## Anti-hallucination: the fail-closed chain

1. **Sufficiency gate** — `HybridRetriever._sufficient` requires at least one retrieved chunk to
   clear `MIN_SIMILARITY` on the **original question embedding**. Sparse token overlap alone can
   never mark a question answerable.
2. **Evidence-only generation** — the model receives retrieved chunks inside explicit
   `>>RETRIEVED_EVIDENCE_START<< … >>RETRIEVED_EVIDENCE_END<<` delimiters; the system prompt
   forbids general-knowledge supplementation.
3. **Verification** — `CitationValidator` (numeric + token grounding) and `ClaimVerifier`
   (per-claim verdicts) must both pass, else the answer is replaced with the exact abstention
   sentence.
4. **LLM failure** — outages/timeouts/rate limits return the best supporting evidence verbatim
   ("Generation failed; no invented content returned"), never a plausible guess.

---

## Security model

- **Prompt-injection defense**: retrieved text is neutralized before prompt composition
  (`src/generation/security.py`); `JailbreakGuard` flags "ignore the PDF / reveal your prompt /
  invent a number / do not cite" attempts and keeps the response PDF-grounded.
- **Adversarial harness**: 3 cases embed injection instructions inside retrieved chunks and
  assert the injected token never appears in the answer — 3/3 passing.
- **API hardening**: length cap, per-IP token-bucket rate limiting, optional HTTP Basic auth.
- **LLM circuit breaker**: 3 consecutive provider failures open the circuit for 30 s (half-open
  probe after cooldown), so a degraded upstream cannot wedge the request path.
- **No secrets in the image**: keys come from the environment; `.env` is gitignored.
- Full details: [`docs/security_model.md`](docs/security_model.md).

---

## Evaluation methodology

Three deterministic harnesses, no LLM judge, fully reproducible:

- **Golden set (18 questions)** — facts with `required_evidence` phrases, expected values and
  pages. Relevance labels are phrase-based; retrieval metrics (Recall@k, Precision@k, MRR,
  nDCG@k), generation metrics (numeric correctness, citation accuracy, faithfulness, page
  accuracy) and abstention correctness are all computed programmatically.
- **Abstention set (6)** — out-of-knowledge questions that must produce the exact abstention
  sentence and never an answer.
- **Adversarial set (3)** — retrieval-level injection cases that must not leak.

`scripts/evaluate.py` archives each run under `docs/eval_results/<timestamp>/`
(`evaluation_results.json`, `claims.jsonl`, `abstention_results.json`, `evaluation_report.md`),
and the `EVIDRA` UI compares archived runs side by side. Detail:
[`docs/evaluation_methodology.md`](docs/evaluation_methodology.md).

---

## Repository layout

```
.
├── src/
│   ├── ingestion/           loader · parser · chunker
│   ├── retrieval/           embeddings · vector_store · bm25 · hybrid · reranker · query_* · routing
│   ├── generation/          prompts · llm (providers + circuit breaker) · citation · security
│   ├── validation/          claims (extractor + verifier)
│   ├── evaluation/          questions.json · abstain_questions.json · adversarial_cases.json · metrics.py · evaluator.py
│   ├── api/main.py          FastAPI server (/api/ready, /api/health, /api/query, …)
│   ├── pipeline.py          orchestration (query → answer)
│   ├── config.py            settings + pricing table
│   ├── schemas.py           dataclasses (RAGResponse includes claims)
│   └── cache.py             retrieval disk cache (persists verdicts)
├── app/streamlit_app.py     chat + evaluation console
├── scripts/                 ingest · build_index · ask · evaluate
├── tests/                   91 tests (unit / integration / security / regression)
├── docs/                    architecture + design + evaluation + failure post-mortem
└── data/raw/Agent-as-a-Judge.pdf
```

---

## Limitations

- The extractive fallback quotes evidence rather than synthesizing; open-ended reasoning quality
  is a function of the configured LLM.
- Relevance labels are phrase-based — deterministic and reproducible, but approximate graded
  human judgement.
- Single-document corpus; multi-document retrieval is future work.
- Chunking is structural; heavily multi-column PDF layouts can occasionally merge spans
  (tracked in `data/processed/ingestion_stats.json`).

See [`docs/future_work.md`](docs/future_work.md) for the prioritized roadmap.

---

## License / notes

Code is provided for evaluation; the paper and dataset belong to their respective authors. No API
keys are committed — secrets live only in `.env`.