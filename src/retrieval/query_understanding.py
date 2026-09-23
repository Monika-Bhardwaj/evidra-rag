from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

NUMERIC_TERMS = (
    "how much",
    "cost",
    "price",
    "how long",
    "time",
    "seconds",
    "minutes",
    "hours",
    "dollar",
    "$",
    "%",
    "percentage",
    "rate",
    "average",
    "alignment",
    "percent",
)
PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")
MONEY_RE = re.compile(r"\$\s?\d")
TIME_RE = re.compile(r"\d+(?:\.\d+)?\s*(seconds|minutes|hours|days|ms)")

FRAMEWORKS = ("metagpt", "gpt-pilot", "gpt pilot", "openhands", "open hands")
JUDGES = ("agent-as-a-judge", "llm-as-a-judge", "human-as-a-judge", "human-as-judge")


@dataclass
class QueryIntent:
    query: str
    category: str
    is_numeric: bool = False
    has_number: bool = False
    entities: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    table_hint: bool = False

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "is_numeric": self.is_numeric,
            "has_number": self.has_number,
            "entities": self.entities,
            "frameworks": self.frameworks,
            "table_hint": self.table_hint,
        }


class QueryClassifier:
    def classify(self, query: str) -> QueryIntent:
        q = query.lower()
        has_number = bool(
            PERCENT_RE.search(query)
            or MONEY_RE.search(query)
            or TIME_RE.search(query)
            or re.search(r"\d", query)
        )
        is_numeric = bool(
            any(term in q for term in NUMERIC_TERMS)
            and (
                has_number
                or any(
                    term in q
                    for term in (
                        "cost",
                        "time",
                        "average",
                        "rate",
                        "%",
                        "alignment",
                        "how much",
                        "how long",
                    )
                )
            )
        )
        entities = self._extract_entities(q)
        frameworks = [f for f in ("MetaGPT", "GPT-Pilot", "OpenHands") if f.lower() in q]
        table_hint = is_numeric or any(t in q for t in ("table", "ablations", "ablation", "figure"))

        if self._is_jailbreak(q):
            category = "jailbreak"
        elif any(
            term in q for term in ("compare", "versus", "vs", "difference", "instead of", "than")
        ):
            category = "comparison"
        elif any(
            term in q
            for term in (
                "cost",
                "price",
                "how much",
                "how long",
                "average",
                "time",
                "rate",
                "savings",
                "percentage",
            )
        ):
            category = "numerical_lookup"
        elif any(
            term in q
            for term in (
                "what is",
                "what are",
                "define",
                "definition",
                "refers to",
                "is a framework",
            )
        ):
            category = "definition"
        elif any(
            term in q
            for term in (
                "how does",
                "how do",
                "method",
                "approach",
                "framework",
                "algorithm",
                "pipeline",
                "architecture",
                "workflow",
                "why",
            )
        ):
            category = "methodology"
        elif any(term in q for term in ("table", "figure", "row", "column")):
            category = "table_lookup"
        elif len(frameworks) >= 2 or len(entities) >= 2:
            category = "multi_hop"
        else:
            category = "factual_lookup"

        return QueryIntent(
            query=query,
            category=category,
            is_numeric=is_numeric or category in ("numerical_lookup", "table_lookup"),
            has_number=has_number,
            entities=entities,
            frameworks=frameworks,
            table_hint=table_hint,
        )

    def _extract_entities(self, q: str) -> List[str]:
        found = []
        for f in FRAMEWORKS:
            if f in q:
                found.append(f.title())
        for j in JUDGES:
            if j in q:
                found.append(j.title())
        return found

    def _is_jailbreak(self, q: str) -> bool:
        markers = (
            "ignore the pdf",
            "ignore previous",
            "ignore all previous",
            "forget",
            "system prompt",
            "reveal",
            "invent a result",
            "invent a number",
            "pretend",
            "do not cite",
            "don't cite",
            "use your internal knowledge",
            "make up",
            "internal knowledge",
        )
        return any(m.lower() in q for m in markers)
