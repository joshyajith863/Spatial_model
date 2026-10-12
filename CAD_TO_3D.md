# CAD / PDF → 3D GLB — Quick Start

Converts existing DXF or vector-PDF floorplans directly into a viewable
`.glb` mesh, skipping the CV/raster pipeline entirely (no YOLO, no OCR,
no scale calibration guesswork — DXF/vector PDF coordinates are exact).

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Try it

1. Open `http://<your-local-ip>:8000/static/viewer.html` in a browser
2. Drop in a `.dxf` file, or a **vector** PDF (exported from CAD, not scanned)
3. The model loads in-browser via `<model-viewer>` — rotate/zoom with mouse or touch

## API directly

```bash
curl -X POST http://localhost:8000/api/mesh/to-glb \
  -F "file=@/path/to/plan.dxf"
```

Returns:
```json
{
  "model_url": "/static/models/<id>.glb",
  "wall_count": 42,
  "room_count": 0,
  "source_format": "dxf"
}
```

## Important: DXF layer names

DXF wall layers vary by CAD tool/firm. The parser defaults to checking
`WALLS`, `A-WALL`, `0` — if your file uses a different layer name, no walls
will be detected. Inspect your file first:

```python
from app.mesh.dxf_parser import list_layers
print(list_layers("your_file.dxf"))
```

Then pass the correct layer(s) to `parse_dxf(file_path, wall_layers=[...])`
(currently hardcoded in `app/api/mesh.py` — wire up a layer-selection step
in the UI if your files vary a lot).

## Vector vs. scanned PDFs

Only **vector** PDFs work here — ones exported directly from CAD software,
where lines are real vector paths. A scanned/photographed PDF has no vector
data to extract; the endpoint detects this and returns a 422 telling you to
use `/api/blueprints/upload` (the raster CV pipeline) instead.

## What's NOT implemented yet (this module specifically)

- **Room polygons from DXF** — currently only walls are extruded. Room
  floor slabs require either a dedicated DXF layer or deriving closed
  room boundaries from the wall network (same technique the CV pipeline
  needs for raster input — can share code later).
- **Doors/windows as openings** — walls currently extrude as solid boxes;
  boolean-subtracting door/window openings needs `manifold3d`'s boolean
  ops layered on top of `builder.py`.
- **WebXR mode** — `viewer.html` is a plain desktop/mobile 3D viewer via
  `<model-viewer>`. True WebXR (phone AR/VR session, voice teleportation)
  is a separate follow-up — `<model-viewer>` itself supports a basic AR
  button on supported phones as a first step toward that.

## How this connects to the rest of the project roadmap

This module is one input path into the system — it does not replace the
SKG, query engine, or raster CV pipeline described in `README.md` and
`WORKPLAN.md`. Where it fits:

| Area | Status | Notes |
|---|---|---|
| DXF → 3D GLB | ✅ Working | This module |
| Vector PDF → 3D GLB | ✅ Working | This module |
| Mesh output feeding the SKG | ✅ Working | `app/skg/from_floorplan.py` builds a `SpatialKnowledgeGraph` from the same `NormalizedFloorplan`, stored by `blueprint_id` (== the mesh upload's `file_id`) in `app/skg/store.py` |
| Voice/text query → graph | ✅ Working (rule-based) | `app/voice/rule_based_translator.py` — regex-based stand-in for the LLM, same `StructuredQuery` output contract. `find_path`, `compute_area`, `check_egress` all wired through `/api/query/` |
| Room adjacency / doors | ⚠️ Approximate | No parser yet tags which rooms a door actually connects. `from_floorplan.py` currently infers adjacency by polygon proximity (rooms within 0.3m are assumed connected) — works for simple layouts, will misfire on complex ones. Replace with real DXF door-layer data when available |
| Raster image → 3D GLB + graph | ❌ Not started | Needs the CV pipeline (`app/cv/pipeline.py`, YOLO-seg + OCR) to first produce a `NormalizedFloorplan`-compatible output, then reuses `builder.py` and `from_floorplan.py` unchanged |
| Voice/text query via local LLM | ❌ Not started | `app/voice/translator.py` stub still unimplemented; rule-based parser above is the working stand-in until Ollama is wired up — swapping it in requires no changes to `app/api/query.py` or `app/skg/engine.py` |
| Full WebXR (teleportation, AR session) | ❌ Not started | Current viewer is desktop/mobile 3D only |
| SDG Advisory Engine (egress, WWR, daylight, HVAC) | ❌ Not started | `check_egress` now works at the pathfinding level, but exit-room tagging is a placeholder (any leaf node in the graph is treated as a candidate exit) — real SDG 11 compliance checks need actual exit/fire-rating data, not inferred from topology alone |
| Scale calibration (manual two-point fallback) | ❌ Not started | Needed for raster CV path; DXF/vector-PDF paths don't need it since their coordinates are already exact (PDF path still needs a points→meters conversion) |

**Net effect:** uploading a DXF now gets you both a 3D view *and* an
answerable graph — `POST /api/mesh/to-glb` returns a `blueprint_id`, which
you then pass to `POST /api/query/` with a typed question like `"path from
kitchen to hallway"` and get a real, graph-computed answer back. This is
the first version of the full loop end-to-end on the CAD/vector-PDF input
path. The two biggest remaining gaps are real door/adjacency data (currently
inferred by proximity) and the raster-image path (still needs the CV
pipeline before it can reuse any of this).
