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
# Imported lazily inside handle_query to avoid a circular import
# (app.voice.rule_based_translator imports StructuredQuery from this module).


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
    # Local imports to avoid a circular import at module load time
    # (rule_based_translator imports StructuredQuery from this file).
    from app.voice.rule_based_translator import parse as translate
    from app.skg import store as skg_store
    from app.skg.engine import execute as execute_query

    structured_query = translate(req.text)

    if structured_query.intent == "unknown":
        raise HTTPException(
            status_code=422,
            detail=f"Couldn't understand '{req.text}'. Try phrasing like "
                   f"'path from kitchen to hallway', 'area of kitchen', "
                   f"or 'egress from bedroom'.",
        )

    skg = skg_store.get(req.blueprint_id)
    if skg is None:
        raise HTTPException(
            status_code=404,
            detail=f"No graph found for blueprint_id '{req.blueprint_id}'. "
                   f"Upload a DXF/vector-PDF via /api/mesh/to-glb first.",
        )

    try:
        result = execute_query(skg, structured_query)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return QueryResponse(structured_query=structured_query, result=result)
