"""
Blueprint ingestion endpoints.

Flow:
    POST /api/blueprints/upload   -> save file, run perception pipeline, return draft graph
    POST /api/blueprints/{id}/confirm -> lock in human-corrected graph as ground truth
    GET  /api/blueprints/{id}     -> fetch current graph state
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel

from app.cv.pipeline import read_text
from app.cv.raster import parse_raster
from app.models import NormalizedFloorplan
from app.parsers.dxf_parser import parse_dxf
from app.skg import store
from app.skg.from_floorplan import build_skg

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Raster plans (Phase 1): images and single-page PDFs go through the OpenCV
# wall extractor instead of the DXF parser.
RASTER_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}
SUPPORTED_EXTS = RASTER_EXTS | {".dxf"}

# In-memory store for the demo. Fine for a local single-session presentation;
# swap for sqlite if you need persistence across restarts.
BLUEPRINTS: dict[str, dict] = {}


class BlueprintSummary(BaseModel):
    id: str
    filename: str
    status: str  # "processing" | "draft_ready" | "confirmed"
    floorplan: NormalizedFloorplan | None = None


@router.post("/upload", response_model=BlueprintSummary)
async def upload_blueprint(
    file: UploadFile = File(...),
    pixels_per_meter: float | None = Query(
        None, gt=0,
        description="Raster only: real-world scale. Omit to let the parser "
                    "estimate it from wall thickness (uncalibrated).",
    ),
):
    blueprint_id = str(uuid.uuid4())
    # Never trust the client filename for paths: keep basename only so a name
    # like "../../evil.png" cannot escape data/uploads.
    safe_name = Path(file.filename or "upload").name
    dest = UPLOAD_DIR / f"{blueprint_id}_{safe_name}"
    dest.write_bytes(await file.read())

    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_EXTS:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            415,
            f"Unsupported file type {suffix or '(none)'}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTS))}",
        )

    try:
        if suffix == ".dxf":
            floorplan = parse_dxf(str(dest))
        else:
            floorplan = parse_raster(str(dest), pixels_per_meter=pixels_per_meter)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        kind = "DXF" if suffix == ".dxf" else "image/PDF"
        raise HTTPException(422, f"Could not parse {kind}: {exc}")

    BLUEPRINTS[blueprint_id] = {
        "filename": file.filename,
        "path": str(dest),
        "status": "draft_ready",
        "floorplan": floorplan,
        "graph": None,
    }
    return BlueprintSummary(id=blueprint_id, filename=file.filename,
                            status="draft_ready", floorplan=floorplan)


class OCRDetection(BaseModel):
    text: str
    confidence: float
    box: list[list[float]]  # 4 (x, y) corners, original image pixels
    center: list[float]


class OCRResponse(BaseModel):
    image_size: tuple[int, int]
    detections: list[OCRDetection]


@router.post("/ocr", response_model=OCRResponse)
async def ocr_blueprint(file: UploadFile = File(...)):
    """Stage 3 of the CV pipeline: OCR a raster blueprint for room labels and
    dimension strings. Returns text + confidence + pixel-space boxes, so the
    verification overlay can draw them straight onto the source image."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        raise HTTPException(415, "OCR expects a raster image (png/jpg/bmp/tif/webp).")

    tmp = UPLOAD_DIR / f"ocr_{uuid.uuid4()}{suffix}"
    tmp.write_bytes(await file.read())
    try:
        return read_text(str(tmp))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"OCR failed: {exc}")
    finally:
        tmp.unlink(missing_ok=True)


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
    """Lock the (possibly human-corrected) draft graph in as ground truth:
    build the SKG from the parsed floorplan and register it in the graph
    store so /api/query/ can answer against it."""
    bp = BLUEPRINTS.get(blueprint_id)
    if not bp:
        raise HTTPException(status_code=404, detail="Blueprint not found")
    if not bp.get("floorplan"):
        raise HTTPException(status_code=409, detail="Nothing to confirm: blueprint has no parsed floorplan.")
    graph = build_skg(bp["floorplan"])
    store.save(blueprint_id, graph)
    bp["graph"] = graph
    bp["status"] = "confirmed"
    return {"id": blueprint_id, "status": "confirmed"}
