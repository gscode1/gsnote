"""Cheap regex-based intent detection (PRD §6 step 1). GENERAL or WHEN."""
import re

_WHEN_PATTERNS = re.compile(
    r"\b(when|what time|which day|how long ago|last week|last month|yesterday|"
    r"today|this week|this month|recently|date|schedule)\b",
    re.IGNORECASE,
)

GENERAL = "GENERAL"
WHEN = "WHEN"


def detect_intent(query: str) -> str:
    return WHEN if _WHEN_PATTERNS.search(query) else GENERAL


# Fusion weights per intent (vector, keyword, recency, graph)
FUSION_WEIGHTS = {
    GENERAL: {"vector": 1.0, "keyword": 1.0, "recency": 0.3, "graph": 0.4},
    WHEN: {"vector": 0.6, "keyword": 0.6, "recency": 1.2, "graph": 0.8},
}
