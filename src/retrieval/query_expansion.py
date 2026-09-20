from __future__ import annotations

import re
from typing import List

from src.logging_utils import get_logger
from src.retrieval.query_understanding import QueryIntent

logger = get_logger(__name__)

SUFFIX_POOL = {
    "cost": ["average cost", "cost", "cost analysis"],
    "time": ["time", "average time", "duration", "how long"],
    "alignment": ["alignment rate", "alignment"],
    "openhands": ["openhands", "open hands", "devai"],
    "metagpt": ["metagpt", "devai"],
    "gpt-pilot": ["gpt-pilot", "gpt pilot", "devai"],
    "task solve": ["task solve rate", "task solve", "solve rate"],
    "requirements met": ["requirements met", "requirement evaluation"],
    "search": ["search module", "bm25", "sentence-bert", "fuzzy search", "search"],
    "percentage": ["save", "savings", "percentage saves", "compared"],
    "drawback": ["drawback", "time-consuming", "manual effort", "expertise"],
    "requirement": ["requirement criteria", "criteria details", "listed requirement"],
}

_DIGIT_REQUIREMENT_RE = re.compile(r"\br(\d+)\b")


class QueryExpander:
    def expand(self, query: str, intent: QueryIntent, max_variants: int = 6) -> List[str]:
        q = query.lower()
        variants: List[str] = []
        for key, suffixes in SUFFIX_POOL.items():
            if key in q and len(suffixes) > 1:
                variants.extend(suffix for suffix in suffixes if suffix not in q)
        m = _DIGIT_REQUIREMENT_RE.search(q)
        if m:
            variants.append(f"criteria r{m.group(1)}")
        if intent.is_numeric:
            variants.append("results table")
            variants.append("numbers")
        seen = set()
        out: List[str] = []
        for candidate in [query, *variants]:
            key = candidate.strip().lower()
            if key in seen or not key:
                continue
            seen.add(key)
            out.append(candidate)
            if len(out) >= max_variants:
                break
        logger.debug("Expanded '%s' -> %s", query, out)
        return out