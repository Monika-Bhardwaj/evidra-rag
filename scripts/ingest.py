from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings, get_settings
from src.ingestion.chunker import SemanticChunker
from src.ingestion.loader import PDFLoader
from src.ingestion.parser import Parser
from src.logging_utils import get_logger
from src.schemas import DocumentChunk

logger = get_logger(__name__)


def download_pdf(settings: Settings) -> None:
    import urllib.request

    url = "https://arxiv.org/pdf/2410.10934v2"
    dest = settings.pdf_path_resolved
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Agent-as-a-Judge v2 from arXiv (%s) ...", url)
    urllib.request.urlretrieve(url, str(dest))
    logger.info("Saved PDF to %s", dest)


def run_ingestion(settings: Settings) -> List[DocumentChunk]:
    settings.ensure_dirs()
    if not settings.pdf_path_resolved.exists():
        logger.error(
            "PDF not found at %s. Run `python scripts/ingest.py --download` to fetch it from arXiv.",
            settings.pdf_path_resolved,
        )
        sys.exit(1)

    loader = PDFLoader(settings.pdf_path_resolved)
    pages = loader.load()

    parser = Parser(pdf_path=settings.pdf_path_resolved)
    try:
        units = parser.parse(pages)
    finally:
        parser.close()

    chunker = SemanticChunker(
        chunk_size_tokens=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap,
        document_id="doc-1",
        source=settings.document_name,
    )
    chunks = chunker.chunk(units)

    processed = settings.processed_dir_path
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "chunks.json").write_text(
        json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    (processed / "pages.json").write_text(
        json.dumps([p.__dict__ for p in pages], ensure_ascii=False),
        encoding="utf-8",
    )
    types: dict = {}
    for c in chunks:
        types[c.chunk_type] = types.get(c.chunk_type, 0) + 1
    pages_covered = sorted({c.page for c in chunks})
    stats = {
        "pages": len(pages),
        "chunks": len(chunks),
        "chunk_types": types,
        "pages_with_chunks": pages_covered,
        "avg_chunk_chars": round(sum(len(c.text) for c in chunks) / max(1, len(chunks))),
    }
    (processed / "ingestion_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Ingestion complete: %s", json.dumps(stats))
    return chunks


def main() -> None:
    settings = get_settings()
    if "--download" in sys.argv:
        download_pdf(settings)
    run_ingestion(settings)


if __name__ == "__main__":
    main()
