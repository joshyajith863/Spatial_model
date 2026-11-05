"""Voice/text -> StructuredQuery translator.

Rule-based regex parser by default; a small local LLM (Ollama) can be
opted into with USE_LLM=1 as an intent classifier + slot-filler. Either way
it only selects one of a fixed set of query templates and extracts room
names — it never computes distances, areas, or routes (those numbers come
from app/skg/engine.py). Any LLM failure falls back to the rules;
never guess silently.
"""

import json
import os

from app.models import StructuredQuery
from app.voice import rule_based_translator

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
USE_LLM = os.getenv("USE_LLM", "0") == "1"  # opt-in; off by default

SYSTEM_PROMPT = (
    "Convert the user's command into JSON with keys: "
    '"intent" (find_path | compute_area | check_egress | unknown), '
    '"room_from", "room_to", "room" (strings or null). '
    "Output JSON only. Extract names only; never compute anything."
)


def _llm_parse(text: str) -> StructuredQuery:
    import urllib.request

    body = json.dumps(
        {"model": OLLAMA_MODEL, "system": SYSTEM_PROMPT, "prompt": text, "format": "json", "stream": False}
    ).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = json.loads(resp.read())["response"]
    return StructuredQuery(**json.loads(raw))


def parse(text: str) -> StructuredQuery:
    """Translate `text` to a StructuredQuery. Falls back to intent="unknown"
    (which the API turns into a 422) rather than guessing."""
    if USE_LLM:
        try:
            q = _llm_parse(text)
            if q.intent != "unknown":
                return q
        except Exception:
            pass  # fall through to rules
    return rule_based_translator.parse(text)
