"""
Voice/text -> structured query translator.

Uses a small, local, quantized text-only LLM purely as an intent classifier +
slot-filler. It selects one of a fixed set of query templates and extracts
room names from the command — it never computes distances, areas, or routes.

Suggested model: a quantized 3B-8B instruct model run locally (e.g. via
llama.cpp / Ollama) so the whole demo works offline on the local network
with no API costs.
"""

from app.api.query import StructuredQuery

SYSTEM_PROMPT = """You convert a spoken facilities command into JSON matching this schema:
{"intent": "find_path" | "compute_area" | "check_egress", "room_from": str | null, "room_to": str | null, "room": str | null}
Only output JSON. Do not compute anything — only extract intent and room names."""


def parse(text: str) -> StructuredQuery:
    """Send `text` to the local LLM with SYSTEM_PROMPT, parse its JSON output
    into a StructuredQuery. Falls back to intent="unknown" on any parse
    failure — never guess silently."""
    raise NotImplementedError("Local LLM call not yet wired up.")
