"""
Blueprint ingestion endpoints.

Flow:
    POST /api/blueprints/upload   -> save file, run perception pipeline, return draft graph
    POST /api/blueprints/{id}/confirm -> lock in human-corrected graph as ground truth
    GET  /api/blueprints/{id}     -> fetch current graph state
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# In-memory store for the demo. Fine for a local single-session presentation;
# swap for sqlite if you need persistence across restarts.
BLUEPRINTS: dict[str, dict] = {}


class BlueprintSummary(BaseModel):
    id: str
    filename: str
    status: str  # "processing" | "draft_ready" | "confirmed"


@router.post("/upload", response_model=BlueprintSummary)
async def upload_blueprint(file: UploadFile = File(...)):
    blueprint_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{blueprint_id}_{file.filename}"
    dest.write_bytes(await file.read())

    # TODO: kick off CV pipeline (app.cv.pipeline.run) here, ideally as a
    # background task so upload returns immediately and the client polls
    # or listens on a websocket for "draft_ready".
    BLUEPRINTS[blueprint_id] = {
        "filename": file.filename,
        "path": str(dest),
        "status": "processing",
        "graph": None,
    }

    return BlueprintSummary(id=blueprint_id, filename=file.filename, status="processing")


@router.get("/{blueprint_id}", response_model=BlueprintSummary)
def get_blueprint(blueprint_id: str):
    bp = BLUEPRINTS.get(blueprint_id)
    if not bp:
        raise HTTPException(status_code=404, detail="Blueprint not found")
    return BlueprintSummary(id=blueprint_id, filename=bp["filename"], status=bp["status"])


@router.post("/{blueprint_id}/confirm")
def confirm_blueprint(blueprint_id: str):
    """Lock the (possibly human-corrected) draft graph in as ground truth."""
    bp = BLUEPRINTS.get(blueprint_id)
    if not bp:
        raise HTTPException(status_code=404, detail="Blueprint not found")
    bp["status"] = "confirmed"
    return {"id": blueprint_id, "status": "confirmed"}
