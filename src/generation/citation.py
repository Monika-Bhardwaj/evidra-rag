from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

from src.schemas import Citation, DocumentChunk, RetrievedChunk

NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)(%|\sand\s|\s|,|;)?")
PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
DOLLAR_RE = re.compile(r"\$?(\d+(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")


@dataclass
class ValidationResult:
    claim_supported: bool
    unsupported_claims: List[str] = field(default_factory=list)
    missing_numbers: List[str] = field(default_factory=list)
    cited_chunk_ids: Set[str] = field(default_factory=set)
    level: str = "high"


def _normalize_number_sequence(text: str) -> List[str]:
    cleaned = text.replace(",", "")
    return [m.group(0) for m in re.finditer(r"\$?\d+(?:\.\d+)?%?", cleaned)]


def _strip_num(value: str) -> str:
    return value.replace("$", "").replace("%", "").replace(",", "").strip()


class CitationValidator:
    def __init__(self, epsilon: float = 0.001) -> None:
        self.epsilon = epsilon

    def validate(
        self,
        answer: str,
        evidence: List[RetrievedChunk],
        numeric_question: bool,
        expected_numbers: Optional[List[str]] = None,
    ) -> ValidationResult:
        supported, unsupported = self._sentence_support(answer, evidence)
        missing = self._numeric_verification(answer, evidence, numeric_question, expected_numbers)
        cited = self._extract_cited_chunks(answer, evidence)
        level = "high"
        if unsupported or missing:
            level = "low"
        return ValidationResult(
            claim_supported=supported and not missing,
            unsupported_claims=unsupported,
            missing_numbers=missing,
            cited_chunk_ids=cited,
            level=level,
        )

    def _sentence_support(
        self, answer: str, evidence: List[RetrievedChunk]
    ) -> tuple[bool, List[str]]:
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        evidence_blocks: List[DocumentChunk] = [e.chunk for e in evidence]
        union_numbers: List[str] = _normalize_number_sequence(
            " ".join(self._strip_citations(c.text) for c in evidence_blocks)
        )
        unsupported: List[str] = []
        for sentence in sentences:
            sentence_clean = self._strip_citations(sentence.strip())
            if len(sentence_clean) < 4:
                continue
            if "could not find sufficient evidence" in sentence_clean.lower():
                continue
            sentence_numbers = _normalize_number_sequence(sentence_clean)
            if sentence_numbers:
                supported = self._numbers_subset(sentence_numbers, union_numbers) and any(
                    _token_overlap(sentence_clean, c.text) >= 1 for c in evidence_blocks
                )
            else:
                supported = any(
                    _token_overlap(sentence_clean, c.text) >= 1 for c in evidence_blocks
                )
            if not supported:
                unsupported.append(sentence_clean)
        return (not unsupported, unsupported)

    def _numeric_verification(
        self,
        answer: str,
        evidence: List[RetrievedChunk],
        numeric_question: bool,
        expected_numbers: Optional[List[str]],
    ) -> List[str]:
        missing: List[str] = []
        if not numeric_question:
            return missing
        answer_clean = self._strip_citations(answer)
        answer_numbers = {_strip_num(n) for n in _normalize_number_sequence(answer_clean)}
        answer_numbers.discard("")
        evidence_numbers = {
            _strip_num(n)
            for c in evidence
            for n in _normalize_number_sequence(self._strip_citations(c.chunk.text))
        }
        if expected_numbers:
            norm_answer = _strip_num(answer_clean.replace(",", ""))
            for num in expected_numbers:
                clean = _strip_num(num)
                if clean not in norm_answer or clean not in evidence_numbers:
                    missing.append(num)
        elif answer_numbers and not answer_numbers.issubset(evidence_numbers):
            missing.extend(sorted(answer_numbers - evidence_numbers))
        return missing

    @staticmethod
    def _strip_citations(text: str) -> str:
        return re.sub(r"\[Page \d+,[^\]]*\]", "", text)

    @staticmethod
    def _numbers_subset(small: List[str], big: List[str]) -> bool:
        smaller = [s.replace("%", "") for s in small]
        bigger = [b.replace("%", "") for b in big]
        return all(s in bigger for s in smaller)

    def _extract_cited_chunks(self, answer: str, evidence: List[RetrievedChunk]) -> Set[str]:
        pages: Set[int] = set()
        for match in re.finditer(r"Page (\d+)", answer):
            pages.add(int(match.group(1)))
        cited = set()
        for rc in evidence:
            if rc.chunk.page in pages:
                cited.add(rc.chunk.chunk_id)
        return cited


def _token_overlap(sentence: str, text: str) -> int:
    a = set(re.findall(r"[a-z0-9]+", sentence.lower()))
    b = set(re.findall(r"[a-z0-9]+", text.lower()))
    a.discard("the")
    a.discard("")
    return len(a & b)


def build_citations(evidence: List[RetrievedChunk]) -> List[Citation]:
    return [
        Citation(
            page=rc.chunk.page,
            section=rc.chunk.section or "(unknown)",
            chunk_id=rc.chunk.chunk_id,
            source=rc.chunk.source,
        )
        for rc in evidence
    ]
