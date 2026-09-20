# Architecture

This document explains how the RAG system is structured and why each component exists.
See `rag_architecture.png` / `retrieval_workflow.png` for the visual pipelines.

## High-level data flow

```
PDF  →  ingestion (loader → parser → chunker)  →  embeddings  →  FAISS index  →  query pipeline  →  grounded answer
```

Everything is wired through `src/pipeline.py`, a single orchestrator reused by the CLI, the
FastAPI server and the Streamlit UI.

## Ingestion (`src/ingestion/`)

| Module | Responsibility |
|---|---|
| `loader.py` | Opens the PDF with PyMuPDF (fallback: pypdf). Emits one `PageDocument` per page with page number preserved. |
| `parser.py` | Normalises whitespace, de-hyphenates line breaks, strips repeated headers/footers, detects headings (`2.2`, `Appendix K`), detects figures/tables captions, extracts tables via PyMuPDF `find_tables` (rows joined with `|`), and classifies narrative vs definition units. |
| `chunker.py` | `SemanticChunker` groups units by `(section, page)` so page metadata is never lost, packs narrative text into chunks of ~`chunk_size_tokens` with `chunk_overlap` overlap, and keeps tables whole (splitting long tables at row boundaries while repeating the header row). Each chunk is classified (`table`, `result`, `cost-analysis`, `methodology`, `definition`, `narrative`, ...). |

Chunk metadata: `chunk_id`, `page`, `section`, `chunk_type`, `document_id`, `source`, `text`.

## Retrieval (`src/retrieval/`)

- `embeddings.py` — `Embedder` interface. `SentenceTransformerEmbedder` (local, CPU) is the
  default; `APIEmbedder` (OpenAI-compatible) is available for swap. Embeddings are
  L2-normalised so FAISS inner-product == cosine similarity.
- `vector_store.py` — `VectorStore` interface with three backends: `FAISSVectorStore`
  (default, persisted), `InMemoryVectorStore` (fallback when FAISS is unavailable), and
  `ChromaVectorStore` (alternative persistent backend).
- `bm25.py` — lexical retrieval with `rank_bm25` (`BM25Okapi`).
- `hybrid.py` — `HybridRetriever` fuses dense + sparse:
  - **weighted** (default, `alpha=0.7`): min-max-normalised dense and sparse scores combined
    as `alpha * dense + (1-alpha) * sparse`.
  - **rrf**: Reciprocal Rank Fusion.
  - Applies metadata filters, `type_boost`, a retrieval confidence threshold, and returns a
    `(evidence_sufficient)` flag for the anti-hallucination gate.
- `reranker.py` — optional `CrossEncoderReranker` (`ms-marco-MiniLM-L-6-v2`) with graceful
  fallback to hybrid ordering when the model is unavailable.
- `query_understanding.py` — rule-based `QueryClassifier` (factual / numerical / table /
  comparison / definition / methodology / multi-hop / jailbreak).
- `query_expansion.py` — controlled query expansion (domain suffixes such as “average cost”,
  “table”) used only to improve recall; never alters the user's intent.
- `routing.py` — `QueryRouter` emits type boosts (table/result/cost-analysis for numeric
  queries, appendix for search-method questions, etc.) and guarantees at least one
  table/result chunk is selected for numeric questions.

## Generation (`src/generation/`)

- `prompts.py` — the strict grounded `SYSTEM_PROMPT` and the user prompt builder that places
  retrieved evidence inside explicit `>>RETRIEVED_EVIDENCE_START<< ... END<<` delimiters so
  the LLM can separate instructions from data.
- `llm.py` — `LLMProvider` interface with `OpenAIProvider`, `GroqProvider`,
  `GeminiProvider` (all OpenAI-compatible), and `LocalExtractiveProvider` (offline,
  deterministic, extracts and quotes evidence sentences — never invents content). Rate-limit
  and connection errors retry with exponential backoff. Token counts and estimated costs are
  computed per call.
- `citation.py` — `CitationValidator` checks every claim sentence against the evidence:
  numeric values in the answer must appear in the retrieved chunks, and non-numeric claims
  require token overlap with evidence. Unsupported claims lower the grounding confidence.
- `security.py` — `JailbreakGuard` detects prompt-injection attempts (ignore-the-PDF,
  reveal-system-prompt, invent numbers, don't-cite, ...) and keeps the response grounded.

## Orchestration (`src/pipeline.py`)

`RagPipeline.answer()` runs:

1. Empty-query guard
2. Jailbreak assessment
3. Query classification + expansion
4. Retrieval (dense + BM25 → fusion → rerank → evidence selection) with a query→chunks
   disk cache
5. Grounded context construction (token-budget aware)
6. LLM generation (system prompt + history + delimited evidence) with an offline fallback
7. Citation validation and low-confidence marking
8. Structured JSON logging of query, chunk ids, scores, model, tokens, latency, cost,
   citations, and failures — never secrets.

`PipelineStats` tracks queries, cache hits, insufficient-evidence events, failures, latency,
tokens and estimated cost for dashboards.

## Evaluation (`src/evaluation/`)

- `questions.json` — the 15 golden questions (id 4–18) with expected answers, required
  evidence phrases, expected pages, and key numeric values.
- `metrics.py` — retrieval metrics (Recall@k, Precision@k, MRR, nDCG@k), generation metrics
  (numeric correctness/recall, citation correctness, faithfulness, completeness,
  page accuracy), computed deterministically without an LLM judge.
- `evaluator.py` — runs every question end-to-end, writes `evaluation_results.json`,
  `docs/evaluation_report.md`, and plots in `docs/eval_plots/`.

Relevance labels are derived from the gold `required_evidence` phrases: a chunk is relevant to
a question if it contains any of its required phrases. This is deterministic and reproducible.

## API & UI

- `src/api/main.py` — FastAPI: `/api/query`, `/api/health`, `/api/evidence/{chunk_id}`,
  `/api/stats`, `/api/evaluation`, `/api/ingest`.
- `app/streamlit_app.py` — Chat UI with evidence panel, retrieval debugger, evaluation
  dashboard and architecture viewer.

## Failure handling paths

| Case | Behaviour |
|---|---|
| PDF missing | `--download` script or clear error |
| PDF extraction fails | pypdf fallback |
| Empty query | prompt for a question |
| No relevant chunks / below threshold | “I could not find sufficient evidence…” |
| Embeddings unavailable | BM25-only |
| Vector DB unavailable | in-memory fallback |
| LLM API failure / rate limit | retry with backoff, else evidence-first response |
| Context over token budget | retrieve → rerank → compress (budget-aware context) |
| Conflicting evidence | both pieces returned with their sections (never merged silently) |