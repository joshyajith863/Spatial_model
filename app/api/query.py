"""
Query endpoint — accepts a natural-language command, translates it via the
small local LLM (app.voice.translator) into one of a fixed set of structured
queries, then executes it deterministically against the SKG (app.skg.engine).

The LLM NEVER computes areas, distances, or paths itself — it only selects
a query template and fills its slots. All math happens in app.skg.
"""

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class QueryRequest(BaseModel):
    blueprint_id: str
    text: str  # transcribed voice command or typed text


class StructuredQuery(BaseModel):
    """The fixed vocabulary of graph operations the LLM is allowed to emit."""
    intent: Literal["find_path", "compute_area", "check_egress", "unknown"]
    room_from: str | None = None
    room_to: str | None = None
    room: str | None = None


class QueryResponse(BaseModel):
    structured_query: StructuredQuery
    result: dict


@router.post("/", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    # TODO: app.voice.translator.parse(req.text) -> StructuredQuery
    # TODO: app.skg.engine.execute(req.blueprint_id, structured_query) -> dict
    structured_query = StructuredQuery(intent="unknown")
    result = {"error": "Pipeline not yet wired up."}

    if structured_query.intent == "unknown":
        raise HTTPException(status_code=501, detail="Query translation not implemented yet.")

    return QueryResponse(structured_query=structured_query, result=result)
