# Reproducibility

EVIDRA is reproducible by design: pinned dependencies, deterministic evaluation, archived runs,
and cache keys that encode the configuration.

## Exact environment

- **Python 3.10 or 3.11** (pinned; the repo is tested on both).
- All dependencies pinned in `requirements.txt` (e.g. `sentence-transformers==3.4.1`,
  `faiss-cpu==1.10.0`, `streamlit==1.49.1`, `openai==1.78.1`, `tenacity==9.1.2`,
  `pydantic==2.11.4`). Install with `pip install -r requirements.txt`.
- The Docker image is `python:3.11-slim` (non-root `evidra`, uid 10001).

## Data pipeline (once)

1. `python scripts/ingest.py --download` — fetch the paper PDF from arXiv 2410.10934
   (`data/raw/Agent-as-a-Judge.pdf`).
2. `python scripts/ingest.py` — extraction + parsing + chunking → `data/processed/chunks.json`,
   `pages.json`, `ingestion_stats.json`.
3. `python scripts/build_index.py` — embeddings + FAISS index + BM25 corpus +
   `data/processed/index/`. Embedder download (~90 MB `all-MiniLM-L6-v2`) is the only network
   dependency; after that everything is offline. Reranker (~90 MB) only if `USE_RERANKER=true`.

`data/processed/` is gitignored; the regression suite (91 tests) skips index-dependent tests when
the index is absent.

## Deterministic evaluation

- Relevance labels are derived from gold `required_evidence` phrases — a fixed function, no
  model, no randomness.
- No seed-dependent sampling anywhere in the query path; the LLM runs greedy
  (`LLM_TEMPERATURE=0` default) or offline-extractive (deterministic).
- Therefore: **two runs of the same commit + same settings must produce identical numbers.** A
  deviation between identical runs is treated as a bug.

## Cache correctness

- `retrieval_config_fingerprint(settings)` hashes every setting that changes retrieval or
  sufficiency (fusion weights/method, top-k values, `min_similarity`, expansion/routing/rerank
  toggles, embedder). Cache entries are only read under a matching fingerprint.
- `tests/test_config.py` pins the fingerprint's sensitivity and insensitivity (generation/ops
  settings like temperature or log level must NOT change the retrieval fingerprint).
- The pipeline additionally keys on the index signature (chunk set + embedder) so a rebuilt
  index invalidates stale entries without manual cache clearing.

## Archived runs as evidence

Every `scripts/evaluate.py` run is archived under `docs/eval_results/<timestamp>/` with four
files (`evaluation_results.json`, `claims.jsonl`, `abstention_results.json`,
`evaluation_report.md`) — committed to the repo so experiments can be diffed. Two committed
archives document the ABS-6 before/after:

- `docs/eval_results/20260923-145944/` — pre-fix: ABS-6 leak (abstention 5/6).
- `docs/eval_results/20260923-153131/` — fixed: golden 18/18, abstention 6/6, adversarial 3/3.

The UI's evaluation tab lists archives and compares runs side by side.

## Commands that reproduce everything

```bash
python scripts/ingest.py --download && python scripts/ingest.py
python scripts/build_index.py
pytest -q                     # 91 pass
ruff check . && ruff format --check .   # clean
python scripts/evaluate.py    # fresh archived run under docs/eval_results/<now>/
streamlit run app/streamlit_app.py
docker compose up --build     # detached, non-root, readiness-gated
```

## Fingerprint pinning

`test_fingerprint_known_string` pins the default-config fingerprint to a literal hash. If it
changes, either a retrieval-affecting default changed (then it must ship with a fresh eval
archive) or the fingerprint function drifted silently (which would poison cache keys) — either
way the test makes the change visible.