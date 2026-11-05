"""Rule-based text -> StructuredQuery translator (regex). Same output contract as
the LLM translator, so it can be swapped or used as a fallback."""

import re

from app.models import StructuredQuery

_PATH = [
    r"(?:path|route|way)\s+from\s+(?:the\s+)?(?P<a>.+?)\s+to\s+(?:the\s+)?(?P<b>.+)",
    r"how\s+(?:do\s+i|to)\s+get\s+from\s+(?:the\s+)?(?P<a>.+?)\s+to\s+(?:the\s+)?(?P<b>.+)",
]
_AREA = [
    r"area\s+of\s+(?:the\s+)?(?P<r>.+)",
    r"how\s+big\s+is\s+(?:the\s+)?(?P<r>.+)",
    r"size\s+of\s+(?:the\s+)?(?P<r>.+)",
]
_EGRESS = [
    r"egress\s+from\s+(?:the\s+)?(?P<r>.+)",
    r"evacuation\s+route\s+from\s+(?:the\s+)?(?P<r>.+)",
    r"(?:nearest\s+)?exit\s+from\s+(?:the\s+)?(?P<r>.+)",
]


def parse(text: str) -> StructuredQuery:
    t = text.strip().strip("?.!").lower()
    for p in _PATH:
        if m := re.search(p, t):
            return StructuredQuery(intent="find_path", room_from=m["a"].strip(), room_to=m["b"].strip())
    for p in _AREA:
        if m := re.search(p, t):
            return StructuredQuery(intent="compute_area", room=m["r"].strip())
    for p in _EGRESS:
        if m := re.search(p, t):
            return StructuredQuery(intent="check_egress", room=m["r"].strip())
    return StructuredQuery(intent="unknown")
