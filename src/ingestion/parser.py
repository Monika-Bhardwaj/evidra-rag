from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from src.ingestion.loader import PageDocument
from src.logging_utils import get_logger

logger = get_logger(__name__)

SECTION_RE = re.compile(r"^(\d+(?:\.\d+){0,2})\s+([A-Z][A-Za-z0-9'&\- ]{1,80})$")
APPENDIX_RE = re.compile(r"^(Appendix\s+[A-Z][A-Za-z0-9'&\- ]{1,80})$")
FIGURE_RE = re.compile(r"^\s*Figure\s+\d+[.:\s]")
TABLE_RE = re.compile(r"^\s*Table\s+\d+[.:\s]")


@dataclass
class Unit:
    kind: str
    text: str
    page: int
    section: str = ""
    source: str = "Agent-as-a-Judge.pdf"
    metadata: dict = field(default_factory=dict)


class TextPreprocessor:
    def normalize(self, text: str) -> str:
        text = text.replace("\u00a0", " ")
        text = text.replace("\u2010", "-")
        text = text.replace("\u2013", "-")
        text = re.sub(r"([a-z])-\s*\n\s*([a-z])", r"\1\2", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class TableExtractor:
    def __init__(self) -> None:
        self._doc = None

    def open(self, pdf_path: Path) -> None:
        import pymupdf

        self._doc = pymupdf.open(str(pdf_path))

    def close(self) -> None:
        if self._doc is not None:
            try:
                self._doc.close()
            except Exception:
                pass
            self._doc = None

    def extract(self, page_number: int, source_name: str) -> List[Unit]:
        if self._doc is None or page_number - 1 >= self._doc.page_count:
            return []
        try:
            page = self._doc[page_number - 1]
            tables: List[Unit] = []
            for tb in page.find_tables().tables:
                rows = []
                for row in tb.extract():
                    cleaned = [str(c).replace("\n", " ").strip() if c else "" for c in row]
                    if any(cleaned):
                        rows.append(" | ".join(cleaned))
                body = "\n".join(rows)
                if self._is_valid_table(body, rows):
                    tables.append(
                        Unit(
                            kind="table",
                            text=body,
                            page=page_number,
                            source=source_name,
                            metadata={"table_rows": len(rows)},
                        )
                    )
            return tables
        except Exception as exc:
            logger.warning("Table extraction failed on page %d: %s", page_number, exc)
            return []

    @staticmethod
    def _is_valid_table(body: str, rows: List[str]) -> bool:
        if not body or len(rows) < 2:
            return False
        letters = sum(1 for ch in body if ch.isalpha())
        if letters / max(1, len(body)) < 0.40:
            return False
        return max(len(r.split("|")) for r in rows) >= 2


class Parser:
    HEADING_MIN_SIZE = 11.5

    def __init__(self, pdf_path: Optional[Path] = None) -> None:
        self.preprocessor = TextPreprocessor()
        self.pdf_path = pdf_path
        self.table_extractor = TableExtractor()
        self._doc = None
        if pdf_path is not None and pdf_path.exists():
            import pymupdf

            self._doc = pymupdf.open(str(pdf_path))
            self.table_extractor.open(pdf_path)

    def close(self) -> None:
        self.table_extractor.close()
        if self._doc is not None:
            try:
                self._doc.close()
            except Exception:
                pass
            self._doc = None

    def parse(self, pages: List[PageDocument]) -> List[Unit]:
        units: List[Unit] = []
        for page in pages:
            text = self.preprocessor.normalize(page.text)
            if not text:
                continue
            units.extend(self._parse_page(page, text))
            units.extend(self.table_extractor.extract(page.page, page.source))
        units.sort(key=lambda u: (u.page, _unit_order(u.kind)))
        return units

    def _span_lines(self, page_number: int) -> List[Tuple[str, float, bool]]:
        if self._doc is None or page_number - 1 >= self._doc.page_count:
            return []
        try:
            data = self._doc[page_number - 1].get_text("dict")
        except Exception:
            return []
        lines: List[Tuple[str, float, bool]] = []
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans)
                max_size = max(round(s["size"], 1) for s in spans)
                bold = any(s["flags"] & 16 for s in spans)
                lines.append((text, max_size, bold))
        return lines

    def _parse_page(self, page: PageDocument, text: str) -> List[Unit]:
        span_lines = self._span_lines(page.page) or [
            (line, 10.0, False) for line in text.split("\n")
        ]
        repeats = self._repeated_short_lines(span_lines)
        heading_buffer: List[str] = []
        buffer: List[str] = []
        current_section = ""
        units: List[Unit] = []

        def flush() -> None:
            nonlocal current_section
            if heading_buffer:
                current_section = " ".join(h.strip() for h in heading_buffer).strip()
                units.append(
                    Unit(
                        kind="heading",
                        text=current_section,
                        page=page.page,
                        section=current_section,
                        source=page.source,
                    )
                )
                heading_buffer.clear()
            joined = self.preprocessor.normalize("\n".join(buffer))
            if joined:
                kind = "narrative"
                if TABLE_RE.match(joined) or FIGURE_RE.match(joined):
                    kind = "caption"
                elif _looks_like_definition(joined):
                    kind = "definition"
                units.append(
                    Unit(
                        kind=kind,
                        text=joined,
                        page=page.page,
                        section=current_section,
                        source=page.source,
                    )
                )
            buffer.clear()

        for raw_line, size, bold in span_lines:
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("arXiv:"):
                continue
            if stripped in repeats:
                continue
            if _is_page_marker(stripped):
                continue
            if self._is_heading(stripped, size, bold):
                flush()
                heading_buffer.append(stripped)
                continue
            buffer.append(stripped)
        flush()
        return units

    @staticmethod
    def _repeated_short_lines(span_lines: List[Tuple[str, float, bool]]) -> set:
        from collections import Counter

        counter = Counter(
            s.strip() for s, size, bold in span_lines if s.strip() and len(s.strip()) < 60
        )
        return {s for s, count in counter.items() if count >= 3}

    def _is_heading(self, line: str, size: float, bold: bool) -> bool:
        if not line or len(line) > 70:
            return False
        if (
            FIGURE_RE.match(line)
            or TABLE_RE.match(line)
            or line.startswith("Metric")
            and len(line) < 20
        ):
            return False
        if size >= self.HEADING_MIN_SIZE:
            return True
        if (
            bold
            and size >= 9.5
            and not line.endswith((".", ",", ":"))
            and not re.search(r"^(Table|Figure|Algorithm)\s+\d", line)
        ):
            return True
        if SECTION_RE.match(line) or APPENDIX_RE.match(line):
            return True
        return False


def _is_page_marker(line: str) -> bool:
    return len(line) <= 3 and line.replace(".", "").isdigit()


def _looks_like_definition(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "we introduce",
        "we define",
        "is defined as",
        "refers to",
        "refers here to",
        "in this work, we",
    )
    return any(marker in lowered for marker in markers)


def _unit_order(kind: str) -> int:
    order = {"heading": 0, "table": 1, "caption": 2, "narrative": 3}
    return order.get(kind, 4)
