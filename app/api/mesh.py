"""
CAD/PDF -> 3D GLB endpoint.

Routes based on uploaded file extension:
    .dxf        -> app.mesh.dxf_parser (exact vectors, no CV needed)
    .pdf        -> app.mesh.pdf_vector_parser if vector-based, else flagged
                   for the raster CV pipeline instead
    .png/.jpg   -> not handled here; use /api/blueprints/upload (CV pipeline)
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.mesh.dxf_parser import parse_dxf
from app.mesh.pdf_vector_parser import parse_pdf_vector, is_vector_pdf
from app.mesh.builder import floorplan_to_glb
from app.skg.from_floorplan import build_skg
from app.skg import store as skg_store

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
GLB_DIR = Path("app/static/models")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GLB_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/to-glb")
async def convert_to_glb(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    file_id = str(uuid.uuid4())
    src_path = UPLOAD_DIR / f"{file_id}{suffix}"
    src_path.write_bytes(await file.read())

    if suffix == ".dxf":
        floorplan = parse_dxf(str(src_path))
    elif suffix == ".pdf":
        if not is_vector_pdf(str(src_path)):
            raise HTTPException(
                status_code=422,
                detail="This PDF appears to be a scanned raster image, not vector CAD "
                       "data. Use /api/blueprints/upload instead (CV pipeline).",
            )
        floorplan = parse_pdf_vector(str(src_path))
    else:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format '{suffix}'. Use .dxf or a vector .pdf. "
                   f"Raster images go through /api/blueprints/upload.",
        )

    if not floorplan.walls:
        raise HTTPException(
            status_code=422,
            detail="No wall geometry found. Check layer names (DXF) or confirm "
                   "the file actually contains drawn vector lines.",
        )

    glb_path = GLB_DIR / f"{file_id}.glb"
    floorplan_to_glb(floorplan, str(glb_path))

    # Populate a queryable graph from the same parsed geometry, keyed by the
    # same file_id, so /api/query/ can answer find_path/compute_area/etc.
    # against this upload without a separate confirm step. Adjacency is a
    # proximity-based placeholder (see app/skg/from_floorplan.py) until real
    # door data is available per format.
    skg = build_skg(floorplan)
    skg_store.save(file_id, skg)

    return {
        "blueprint_id": file_id,
        "model_url": f"/static/models/{file_id}.glb",
        "wall_count": len(floorplan.walls),
        "room_count": len(floorplan.rooms),
        "source_format": floorplan.source_format,
        "queryable": len(floorplan.rooms) > 0,
    }
