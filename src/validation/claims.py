from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.schemas import RetrievedChunk

PAGE_MARKER_RE = re.compile(r"\[Page\s+(\d+),\s*Section\s+[^\]]*\]")
BULLET_RE = re.compile(r"^(?:[-*•])\s+|^\d+[.)]\s+")
BULLET_EDGE_RE = re.compile(r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+")
META_PREFIXES = (
    "according to the paper",
    "the language model could not be reached",
    "returning the best supporting evidence",
    "generation failed",
    "please provide a question",
)
INSF_PHRASE = "could not find sufficient evidence"
_STOP_TOKENS = {"the", "a", "an", "of", "to", "and", "in", "for", "is", "are", "was", "were"}
MIN_TOKEN_OVERLAP = 3
MIN_MEANINGFUL_TOKENS = 3
MIN_ALPHA_RATIO = 0.5

VERDICT_SUPPORTED = "supported"
VERDICT_LOW_SUPPORT = "low-support"
VERDICT_UNSUPPORTED = "unsupported"


@dataclass
class EvidenceClaim:
    text: str
    page: Optional[int]
    supported: bool
    page_matched: bool
    numeric_matched: bool
    token_overlap: int
    verdict: str
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "text": self.text,
            "page": self.page,
            "supported": self.supported,
            "page_matched": self.page_matched,
            "numeric_matched": self.numeric_matched,
            "token_overlap": self.token_overlap,
            "verdict": self.verdict,
            "reason": self.reason,
        }


def _meaningful_tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP_TOKENS and len(t) > 1
    }


def _numbers(text: str) -> set[str]:
    cleaned = re.sub(r"\(\d+\)", "", text.replace(",", ""))
    return {m.group(0) for m in re.finditer(r"\$?\d+(?:\.\d+)?%?", cleaned)}


class ClaimExtractor:
    def extract(self, answer: str) -> List[Dict[str, Optional[int]]]:
        claims: List[Dict[str, Optional[int]]] = []
        seen: set[str] = set()
        for unit in self._units(answer):
            if self._is_meta(unit):
                continue
            for text, page in self._segment_line(unit):
                if self._is_debris(text):
                    continue
                if len(text) < 4 or text in seen:
                    continue
                seen.add(text)
                claims.append({"text": text, "page": page})
        return claims

    def _units(self, answer: str) -> List[str]:
        starts = [m.start() for m in BULLET_EDGE_RE.finditer(answer)]
        if not starts:
            text = re.sub(r"\s+", " ", answer).strip()
            return [text] if text else []
        units: List[str] = []
        cursor = 0
        for i, pos in enumerate(starts):
            if pos > cursor:
                pre = re.sub(r"\s+", " ", answer[cursor:pos]).strip()
                if pre:
                    units.append(pre)
            end = starts[i + 1] if i + 1 < len(starts) else len(answer)
            units.append(re.sub(r"\s+", " ", answer[pos:end]).strip())
            cursor = end
        return units

    def _segment_line(self, line: str) -> List[tuple[str, Optional[int]]]:
        line = BULLET_RE.sub("", line, count=1).strip()
        if not line:
            return []
        segments: List[tuple[str, Optional[int]]] = []
        pos = 0
        for match in PAGE_MARKER_RE.finditer(line):
            text = line[pos : match.start()].strip()
            if text:
                segments.append((text, int(match.group(1))))
            pos = match.end()
        tail = line[pos:].strip()
        if tail:
            segments.append((tail, None))
        if not segments and line:
            segments.append((line, None))
        return segments

    @staticmethod
    def _is_debris(text: str) -> bool:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        alpha_chars = sum(len(t) for t in tokens)
        non_space = len(re.sub(r"\s", "", text))
        if non_space == 0:
            return True
        if alpha_chars / non_space < MIN_ALPHA_RATIO:
            return True
        return len([t for t in tokens if t not in _STOP_TOKENS]) < MIN_MEANINGFUL_TOKENS

    @staticmethod
    def _is_meta(text: str) -> bool:
        lowered = text.lower().strip()
        if INSF_PHRASE in lowered or lowered.endswith(":"):
            return True
        return any(lowered.startswith(prefix) for prefix in META_PREFIXES)


class ClaimVerifier:
    def __init__(self, min_token_overlap: int = MIN_TOKEN_OVERLAP) -> None:
        self.min_token_overlap = min_token_overlap

    def verify(self, answer: str, evidence: List[RetrievedChunk]) -> List[EvidenceClaim]:
        claims = ClaimExtractor().extract(answer)
        evidence_pages = {c.chunk.page for c in evidence}
        evidence_text = "\n".join(c.chunk.text for c in evidence)
        evidence_tokens = _meaningful_tokens(evidence_text)
        evidence_numbers = _numbers(evidence_text)

        verdicts: List[EvidenceClaim] = []
        for claim in claims:
            text = claim["text"]
            page = claim["page"]
            claim_tokens = _meaningful_tokens(text)
            overlap = len(claim_tokens & evidence_tokens)
            claim_numbers = _numbers(text)
            numeric_claim = bool(claim_numbers)
            numeric_matched = (not numeric_claim) or bool(claim_numbers & evidence_numbers)
            page_matched = page is None or page in evidence_pages

            page_violation = page is not None and not page_matched
            numeric_violation = numeric_claim and not numeric_matched
            hard_violation = page_violation or numeric_violation

            has_token_support = overlap >= self.min_token_overlap
            has_numeric_support = numeric_claim and numeric_matched and overlap >= 2
            if hard_violation:
                verdict = VERDICT_UNSUPPORTED
                supported = False
            elif has_token_support or has_numeric_support:
                verdict = VERDICT_SUPPORTED
                supported = True
            else:
                verdict = VERDICT_LOW_SUPPORT
                supported = False

            reason = self._reason(overlap, page, page_matched, numeric_matched, numeric_claim)
            verdicts.append(
                EvidenceClaim(
                    text=text,
                    page=page,
                    supported=supported,
                    page_matched=page_matched,
                    numeric_matched=numeric_matched,
                    token_overlap=overlap,
                    verdict=verdict,
                    reason=reason,
                )
            )
        return verdicts

    @staticmethod
    def _reason(
        overlap: int,
        page: Optional[int],
        page_matched: bool,
        numeric_matched: bool,
        numeric_claim: bool,
    ) -> str:
        if not page_matched and page is not None:
            return f"cited page {page} not present in answer evidence"
        if numeric_claim and not numeric_matched:
            return "numbers claimed but not grounded in cited evidence"
        if overlap < MIN_TOKEN_OVERLAP:
            return "insufficient token overlap with cited evidence (low-support)"
        return "supported by cited evidence"
