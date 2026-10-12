"""
Rule-based query translator — a stand-in for app/voice/translator.py's local
LLM version. Same output contract (StructuredQuery), so swapping in the real
LLM later requires no changes downstream (app/skg/engine.py, app/api/query.py).

Handles a small set of phrasings:
    "path from X to Y" / "route from X to Y" / "how do I get from X to Y"
    "area of X" / "how big is X" / "square footage of X"
    "egress from X" / "evacuation route from X" / "nearest exit from X"

Deliberately simple: no NLU, just pattern matching. Good enough to demo the
full loop (upload -> graph -> query -> answer) before the LLM is wired up.
"""

import re

from app.api.query import StructuredQuery

_PATH_PATTERNS = [
    r"(?:path|route|way)\s+from\s+(?P<from>.+?)\s+to\s+(?P<to>.+)",
    r"how\s+do\s+i\s+get\s+from\s+(?P<from>.+?)\s+to\s+(?P<to>.+)",
]

_AREA_PATTERNS = [
    r"area\s+of\s+(?:the\s+)?(?P<room>.+)",
    r"how\s+big\s+is\s+(?:the\s+)?(?P<room>.+)",
    r"square\s+footage\s+of\s+(?:the\s+)?(?P<room>.+)",
]

_EGRESS_PATTERNS = [
    r"egress\s+from\s+(?:the\s+)?(?P<room>.+)",
    r"evacuation\s+route\s+from\s+(?:the\s+)?(?P<room>.+)",
    r"nearest\s+exit\s+from\s+(?:the\s+)?(?P<room>.+)",
]


def _clean(text: str) -> str:
    return text.strip().strip("?.!").lower()


def parse(text: str) -> StructuredQuery:
    """Match the input text against known phrasings. Returns intent="unknown"
    if nothing matches — never guess silently."""
    cleaned = _clean(text)

    for pattern in _PATH_PATTERNS:
        m = re.search(pattern, cleaned)
        if m:
            return StructuredQuery(
                intent="find_path",
                room_from=m.group("from").strip(),
                room_to=m.group("to").strip(),
            )

    for pattern in _AREA_PATTERNS:
        m = re.search(pattern, cleaned)
        if m:
            return StructuredQuery(intent="compute_area", room=m.group("room").strip())

    for pattern in _EGRESS_PATTERNS:
        m = re.search(pattern, cleaned)
        if m:
            return StructuredQuery(intent="check_egress", room=m.group("room").strip())

    return StructuredQuery(intent="unknown")
