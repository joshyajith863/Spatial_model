"""
3D mesh endpoints.

    POST /api/mesh/to-glb          -> build a GLB from an uploaded blueprint's
                                       NormalizedFloorplan (or a synthetic sample)
    GET  /api/mesh/sample/glb      -> serve the built-in sample GLB (no upload needed)

The GLB is written to app/static/glb/ and also returned inline, so the WebXR
viewer can fetch it by URL without CORS friction.
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app.mesh.builder import (
    MeshOptions,
    build_glb,
    sample_floorplan,
    DEFAULT_WALL_HEIGHT_M,
    DEFAULT_WALL_THICKNESS_M,
)
# Import the shared blueprint store from the ingestion module.
from app.api.blueprints import BLUEPRINTS

router = APIRouter()

GLB_DIR = Path("app/static/glb")
GLB_DIR.mkdir(parents=True, exist_ok=True)


class MeshRequest(BaseModel):
    blueprint_id: str | None = None  # build from an uploaded blueprint
    sample: bool = False  # or just build the synthetic sample
    wall_height_m: float = DEFAULT_WALL_HEIGHT_M
    wall_thickness_m: float = DEFAULT_WALL_THICKNESS_M
    include_floors: bool = True
    include_ceiling: bool = False


class MeshResponse(BaseModel):
    id: str
    url: str  # /static/glb/<id>.glb
    triangle_count: int
    wall_area_m2: float
    floor_area_m2: float
    bounds: tuple[float, float, float, float, float, float]
    warnings: list[str]


def _build_and_store(floorplan, opts: MeshOptions) -> MeshResponse:
    try:
        result = build_glb(floorplan, opts)
    except ValueError as exc:
        raise HTTPException(422, f"Could not build mesh: {exc}")

    mesh_id = str(uuid.uuid4())
    rel = f"/static/glb/{mesh_id}.glb"
    (GLB_DIR / f"{mesh_id}.glb").write_bytes(result.glb)

    return MeshResponse(
        id=mesh_id, url=rel, triangle_count=result.triangle_count,
        wall_area_m2=result.wall_area_m2, floor_area_m2=result.floor_area_m2,
        bounds=result.bounds, warnings=result.warnings,
    )


@router.post("/to-glb", response_model=MeshResponse)
def to_glb(req: MeshRequest):
    opts = MeshOptions(
        wall_height_m=req.wall_height_m, wall_thickness_m=req.wall_thickness_m,
        include_floors=req.include_floors, include_ceiling=req.include_ceiling,
    )

    if req.sample:
        return _build_and_store(sample_floorplan(), opts)

    if not req.blueprint_id:
        raise HTTPException(400, "Provide a blueprint_id, or set sample=true.")

    bp = BLUEPRINTS.get(req.blueprint_id)
    if not bp or not bp.get("floorplan"):
        raise HTTPException(404, "Blueprint not found or has no parsed floorplan.")

    return _build_and_store(bp["floorplan"], opts)


@router.get("/sample/glb")
def sample_glb():
    """Build (and cache) the sample GLB; return the raw bytes for the viewer."""
    cache = GLB_DIR / "_sample.glb"
    if not cache.exists():
        result = build_glb(sample_floorplan(), MeshOptions())
        cache.write_bytes(result.glb)
    return Response(content=cache.read_bytes(), media_type="model/gltf-binary")
