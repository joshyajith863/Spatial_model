"""
Query endpoint — accepts a natural-language command, translates it via
app.voice (regex rules; Ollama LLM if USE_LLM=1) into one of a fixed set of
structured queries, then executes it deterministically against the SKG.

The translator NEVER computes areas, distances, or paths — it only selects
a query template and fills its slots. All math happens in app.skg.engine.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models import StructuredQuery
from app.skg import store
from app.skg.engine import execute
from app.voice.translator import parse

router = APIRouter()


class QueryRequest(BaseModel):
    blueprint_id: str
    text: str  # transcribed voice command or typed text


class QueryResponse(BaseModel):
    structured_query: StructuredQuery
    result: dict


@router.post("/", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    query = parse(req.text)
    if query.intent == "unknown":
        raise HTTPException(
            422,
            "Couldn't understand that. Try 'path from kitchen to hallway', "
            "'area of kitchen', or 'egress from bedroom'.",
        )
    skg = store.get(req.blueprint_id)
    if skg is None:
        raise HTTPException(404, f"No graph for blueprint_id '{req.blueprint_id}'.")
    try:
        result = execute(skg, query)
    except (ValueError, KeyError) as e:
        raise HTTPException(422, str(e))
    return QueryResponse(structured_query=query, result=result)
