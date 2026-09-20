#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f "data/raw/Agent-as-a-Judge.pdf" ]; then
  echo "Downloading paper PDF from arXiv (2410.10934)..."
  python scripts/ingest.py --download
fi

if [ ! -f "data/processed/index/chunks.json" ]; then
  echo "Building index..."
  python scripts/build_index.py
fi

MODE="${1:-app}"
case "$MODE" in
  api)
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    ;;
  eval)
    python scripts/evaluate.py
    ;;
  *)
    streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
    ;;
esac