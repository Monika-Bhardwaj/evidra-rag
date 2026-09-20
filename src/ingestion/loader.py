from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PageDocument:
    page: int
    text: str
    source: str


class PDFLoader:
    def __init__(self, pdf_path: Path, source_name: Optional[str] = None) -> None:
        self.pdf_path = Path(pdf_path)
        self.source_name = source_name or self.pdf_path.name

    def load(self) -> List[PageDocument]:
        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found at {self.pdf_path}. Download the paper and place it "
                f"in data/raw/ (arXiv 2410.10934), or run `python scripts/ingest.py --download`."
            )
        pages = self._try_pymupdf()
        if pages is None:
            pages = self._fallback_pypdf()
        if not pages:
            raise ValueError(f"No extractable text found in {self.pdf_path}.")
        logger.info("Loaded %d pages from %s", len(pages), self.pdf_path.name)
        return pages

    def _try_pymupdf(self) -> Optional[List[PageDocument]]:
        try:
            import pymupdf
        except ImportError:
            logger.warning("PyMuPDF unavailable; trying pypdf fallback.")
            return None
        try:
            doc = pymupdf.open(self.pdf_path)
            pages: List[PageDocument] = []
            for i in range(doc.page_count):
                text = doc[i].get_text("text")
                pages.append(PageDocument(page=i + 1, text=text, source=self.source_name))
            doc.close()
            return pages
        except Exception as exc:  # pragma: no cover - failure path
            logger.error("PyMuPDF extraction failed: %s", exc)
            return None

    def _fallback_pypdf(self) -> List[PageDocument]:
        from pypdf import PdfReader

        reader = PdfReader(str(self.pdf_path))
        pages: List[PageDocument] = []
        failed: List[int] = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                logger.error("pypdf failed on page %d: %s", i + 1, exc)
                failed.append(i + 1)
                text = ""
            pages.append(PageDocument(page=i + 1, text=text, source=self.source_name))
        if failed:
            logger.warning("pypdf could not extract pages: %s", failed)
        return pages