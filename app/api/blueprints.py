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

from app.models import NormalizedFloorplan
from app.parsers.dxf_parser import parse_dxf

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
    floorplan: NormalizedFloorplan | None = None


@router.post("/upload", response_model=BlueprintSummary)
async def upload_blueprint(file: UploadFile = File(...)):
    blueprint_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{blueprint_id}_{file.filename}"
    dest.write_bytes(await file.read())

    suffix = Path(file.filename).suffix.lower()
    if suffix != ".dxf":
        dest.unlink(missing_ok=True)
        raise HTTPException(415, "Only .dxf is supported for now (raster/PDF not implemented).")
    try:
        floorplan = parse_dxf(str(dest))
    except Exception as exc:
        raise HTTPException(422, f"Could not parse DXF: {exc}")

    BLUEPRINTS[blueprint_id] = {
        "filename": file.filename,
        "path": str(dest),
        "status": "draft_ready",
        "floorplan": floorplan,
        "graph": None,
    }
    return BlueprintSummary(id=blueprint_id, filename=file.filename,
                            status="draft_ready", floorplan=floorplan)


@router.get("/{blueprint_id}", response_model=BlueprintSummary)
def get_blueprint(blueprint_id: str):
    bp = BLUEPRINTS.get(blueprint_id)
    if not bp:
        raise HTTPException(status_code=404, detail="Blueprint not found")
    return BlueprintSummary(id=blueprint_id, filename=bp["filename"], status=bp["status"])


@router.get("/{blueprint_id}/floorplan", response_model=NormalizedFloorplan)
def get_floorplan(blueprint_id: str):
    bp = BLUEPRINTS.get(blueprint_id)
    if not bp or not bp.get("floorplan"):
        raise HTTPException(status_code=404, detail="Floorplan not found")
    return bp["floorplan"]


@router.post("/{blueprint_id}/confirm")
def confirm_blueprint(blueprint_id: str):
    """Lock the (possibly human-corrected) draft graph in as ground truth."""
    bp = BLUEPRINTS.get(blueprint_id)
    if not bp:
        raise HTTPException(status_code=404, detail="Blueprint not found")
    bp["status"] = "confirmed"
    return {"id": blueprint_id, "status": "confirmed"}
