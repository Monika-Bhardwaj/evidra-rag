from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from src.logging_utils import get_logger

logger = get_logger(__name__)

INJECTION_MARKERS = (
    "ignore the pdf",
    "ignore all previous",
    "ignore previous instructions",
    "ignore prior instructions",
    "forget everything",
    "forget all instructions",
    "reveal your system prompt",
    "system prompt",
    "show your instructions",
    "print your instructions",
    "what are your instructions",
    "invent a result",
    "invent a number",
    "make up a number",
    "make up data",
    "pretend the pdf says",
    "pretend the paper says",
    "do not cite",
    "don't cite",
    "without citing",
    "use your internal knowledge",
    "use your own knowledge",
    "using your internal knowledge",
    "as a senior engineer",
    "you are now",
    "new instructions",
    "assume the paper says",
    "assume the pdf says",
    "believe openhands costs",
    "act as a",
    "convincing but false",
    "false answer",
    "hallucinate an answer",
    "hallucinate a result",
    "fabricate",
    "make up an answer",
    "inaccurate answer",
)


@dataclass
class SecurityAssessment:
    flagged: bool
    matched_markers: List[str] = field(default_factory=list)
    safe_message: str = ""

    def severity(self) -> str:
        return "high" if self.flagged else "safe"


class JailbreakGuard:
    def assess(self, query: str) -> SecurityAssessment:
        q = query.lower().strip()
        matched = [m for m in INJECTION_MARKERS if m in q]
        if matched:
            return SecurityAssessment(
                flagged=True,
                matched_markers=matched,
                safe_message=(
                    "This query attempted to override grounding rules. The assistant remains "
                    "document-grounded; only content found in the PDF will be returned."
                ),
            )
        return SecurityAssessment(flagged=False)

    def sanitize_query(self, query: str) -> str:
        sanitized = re.sub(r"^(ignore|forget|pretend|assume|do not|don't|act as)\b[^,.;?!]*[,.;?!]\s*", "", query)
        sanitized = re.split(
            r"\s+(?:and|then)\s+",
            sanitized.strip(),
            maxsplit=1,
        )
        if (
            isinstance(sanitized, list)
            and len(sanitized) == 2
            and re.match(r"^(ignore|forget|disregard|pretend|assume|do not|don't)\b", sanitized[0], re.I)
        ):
            return sanitized[1].strip()
        result = sanitized[0] if isinstance(sanitized, list) else sanitized
        result = result.strip()
        return result if result else query


def strip_suspicious_instructions(query: str) -> bool:
    q = query.lower()
    return any(m in q for m in INJECTION_MARKERS)