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
        sanitized = re.sub(
            r"^(ignore|forget|pretend|assume|do not|don't|act as)\b[^,.;?!]*[,.;?!]\s*", "", query
        )
        sanitized = re.split(
            r"\s+(?:and|then)\s+",
            sanitized.strip(),
            maxsplit=1,
        )
        if (
            isinstance(sanitized, list)
            and len(sanitized) == 2
            and re.match(
                r"^(ignore|forget|disregard|pretend|assume|do not|don't)\b", sanitized[0], re.I
            )
        ):
            return sanitized[1].strip()
        result = sanitized[0] if isinstance(sanitized, list) else sanitized
        result = result.strip()
        return result if result else query


RETRIEVED_INJECTION_PATTERNS = (
    "ignore all previous instructions",
    "ignore the above",
    "ignore previous instructions",
    "ignore prior instructions",
    "forget everything",
    "forget all instructions",
    "disregard all previous",
    "disregard the instructions",
    "reveal your system prompt",
    "you must now",
    "you are now",
    "new instructions:",
    "system prompt:",
    "act as if",
    "pretend you are",
    "output only",
    "respond only with",
    "do not mention",
    "answer this instead",
    "override your instructions",
)

NEUTRAL_PLACEHOLDER = "[REDACTED: embedded-instruction]"


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Second line of defense beyond the verbatim marker list: flag any sentence that
# *starts* with an imperative second-person instruction verb. The corpus was
# checked for false positives — every use of ignore/reveal/disregard in the paper
# is first-person or third-person prose ("we found", "our experiments revealed"),
# never a bare imperative to the model.
_IMPERATIVE_LEADER = re.compile(
    r"^\s*(?:"
    r"ignore|forget|disregard|pretend|assume|invent|reveal|output|respond|"
    r"answer|now|never|always|instead|override|disobey|"
    r"do not|don'?t|you must|you will|you are|you should|"
    r"from now on|act as if|from now"
    r")\b",
    re.IGNORECASE,
)


def _sentence_is_suspicious(part: str) -> bool:
    lowered = part.lower()
    if any(m in lowered for m in RETRIEVED_INJECTION_PATTERNS):
        return True
    return bool(_IMPERATIVE_LEADER.match(part))


def neutralize_retrieved_text(text: str, placeholder: str = NEUTRAL_PLACEHOLDER) -> str:
    parts = _SENTENCE_SPLIT.split(text.strip())
    if len(parts) <= 1:
        if _sentence_is_suspicious(text):
            return placeholder
        return text
    flagged: List[int] = []
    for i, part in enumerate(parts):
        if _sentence_is_suspicious(part):
            flagged.append(i)
    if not flagged:
        return text
    out = "\n".join(placeholder if i in flagged else part for i, part in enumerate(parts))
    return out


def strip_suspicious_instructions(query: str) -> bool:
    q = query.lower()
    return any(m in q for m in INJECTION_MARKERS)
