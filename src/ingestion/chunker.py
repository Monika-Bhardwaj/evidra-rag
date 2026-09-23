from __future__ import annotations

import re
from typing import Dict, List, Tuple

from src.ingestion.parser import Unit
from src.logging_utils import get_logger
from src.schemas import DocumentChunk

logger = get_logger(__name__)

TABLE_RE = re.compile(r"^\s*Table\s+\d+")
FIGURE_RE = re.compile(r"^\s*Figure\s+\d+")
DIGIT_TOKEN_RE = re.compile(r"\S*\d\S*")


class SemanticChunker:
    def __init__(
        self,
        chunk_size_tokens: int = 500,
        overlap: float = 0.15,
        document_id: str = "doc-1",
        source: str = "Agent-as-a-Judge.pdf",
    ) -> None:
        self.max_tokens = chunk_size_tokens
        self.max_words = chunk_size_tokens
        self.overlap = overlap
        self.document_id = document_id
        self.source = source

    def chunk(self, units: List[Unit]) -> List[DocumentChunk]:
        grouped = self._group_by_section_page(units)
        chunks: List[DocumentChunk] = []
        seq = 0
        for (section, page), group_units in grouped.items():
            for unit in group_units:
                if unit.kind == "heading":
                    continue
                if unit.kind in ("table", "caption"):
                    pieces = self._chunk_table(unit.text)
                else:
                    pieces = self._chunk_narrative(unit.text)
                for piece in pieces:
                    kind = self.classify_chunk(section, unit.kind, piece)
                    chunk = DocumentChunk(
                        chunk_id=f"p{page:03d}-{seq:04d}",
                        text=piece,
                        page=page,
                        section=section,
                        chunk_type=kind,
                        document_id=self.document_id,
                        source=self.source,
                    )
                    chunks.append(chunk)
                    seq += 1
        logger.info("Produced %d chunks", len(chunks))
        return chunks

    def _group_by_section_page(self, units: List[Unit]) -> Dict[Tuple[str, int], List[Unit]]:
        grouped: Dict[Tuple[str, int], List[Unit]] = {}
        heading = ""
        for unit in units:
            if unit.kind == "heading":
                heading = unit.text
                continue
            section = unit.section or heading
            key = (section, unit.page)
            grouped.setdefault(key, []).append(unit)
        return grouped

    def _chunk_narrative(self, text: str) -> List[str]:
        words = text.split()
        budget = self.max_words
        if len(words) <= budget:
            return [text]
        overlap_n = max(1, int(budget * self.overlap))
        step = max(1, budget - overlap_n)
        pieces: List[str] = []
        i = 0
        while i < len(words):
            end = min(i + budget, len(words))
            pieces.append(" ".join(words[i:end]))
            if end >= len(words):
                break
            i += step
        deduped: List[str] = []
        for piece in pieces:
            if not deduped or deduped[-1] != piece:
                deduped.append(piece)
        return deduped

    def _chunk_table(self, text: str) -> List[str]:
        lines = text.split("\n")
        if len(lines) <= self.max_words // 12:
            return [text]
        header = lines[0] if lines else ""
        rows = lines[1:]
        budget = max(1, self.max_words - len(header.split()))
        pieces: List[str] = []
        current = [header]
        current_words = len(header.split())
        for row in rows:
            row_words = len(row.split())
            if current_words + row_words > budget and current:
                pieces.append("\n".join(current))
                current = [header, row] if row_words <= budget else [header]
                current_words = len(header.split()) + (row_words if row in current else 0)
            else:
                current.append(row)
                current_words += row_words
        if current:
            pieces.append("\n".join(current))
        return pieces

    def classify_chunk(self, section: str, kind: str, text: str) -> str:
        sl = section.lower()
        if kind == "table":
            return "table"
        if kind == "caption":
            return "table" if TABLE_RE.match(text) else "figure"
        if kind == "definition":
            return "definition"
        if "cost" in sl or "budget" in sl:
            return "cost-analysis"
        if "reference" in sl or "bibliography" in sl or sl.startswith("references"):
            return "reference"
        if sl.startswith("appendix"):
            return "appendix"
        if any(
            m in sl
            for m in ("method", "framework", "approach", "system overview", "implementation")
        ):
            return "methodology"
        if self._numeric_density(text) >= 0.05 or text.count("%") >= 3:
            return "result"
        return "narrative"

    @staticmethod
    def _numeric_density(text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        numeric = sum(1 for w in words if DIGIT_TOKEN_RE.search(w))
        return numeric / len(words)
