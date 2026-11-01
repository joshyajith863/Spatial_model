# Spatial Copilot — API Reference

Base URL (local demo): `http://<host>:8000` (or `https://<host>:8000` when TLS is on — required for WebXR on phones).

- **Interactive docs (Swagger UI):** `GET /docs` — served from local static assets, no CDN needed.
- **Machine-readable spec:** `GET /openapi.json`.
- **Auth:** none. This is a local-network demo build; CORS is wide open (`*`). Do not expose it publicly.
- **Content types:** JSON unless noted. File upload uses `multipart/form-data`. GLB responses use `model/gltf-binary`.

## Conventions

- All coordinates and lengths are normalised to **metres**. Plan geometry lives in 2D (`x`, `y`); the 3D mesh maps plan `y` onto world `z` and extrudes height along world `y`, with the floor pinned to `y = 0`.
- Every response that carries a warning list (`warnings: string[]`) surfaces parser/mesh caveats inline — read them; they flag guessed units, inferred rooms, missing room layers, etc.
- Error responses use FastAPI's standard shape: `{"detail": "<message>"}` with an appropriate HTTP status (see each endpoint).

---

## Core data models

These Pydantic shapes appear across endpoints (`app/models.py`).

### `Point`
```json
{ "x": 1.23, "y": 4.56 }
```

### `Wall`
```json
{ "start": { "x": 0.0, "y": 0.0 }, "end": { "x": 8.0, "y": 0.0 } }
```

### `Room`
| Field | Type | Notes |
|---|---|---|
| `id` | string | e.g. `"room_0"` |
| `label` | string \| null | OCR/text label if found (e.g. `"Kitchen"`), else null |
| `polygon` | `Point[]` | exterior ring, counter-clockwise, no repeated closing point |
| `area_m2` | float | polygon area in m² |
| `inferred` | bool | true if derived from wall loops rather than a dedicated rooms layer |

### `NormalizedFloorplan`
The single intermediate representation every parser converges to.
| Field | Type | Notes |
|---|---|---|
| `source_format` | string | e.g. `"dxf"`, `"sample"` |
| `units` | string | always `"m"` after normalisation |
| `bounds` | `[minx, miny, maxx, maxy]` | 4-tuple, metres |
| `walls` | `Wall[]` | |
| `rooms` | `Room[]` | may be empty (rooms not yet inferred — see Phase 3) |
| `warnings` | `string[]` | parser caveats |

---

## System

### `GET /`
Redirects to the 3D/WebXR viewer (`/static/viewer.html`). The API docs remain at `/docs`.

- **200** → HTML redirect (307) to the viewer.

### `GET /health`
Liveness check.
- **200** → `{"status": "ok"}`

---

## Blueprints — ingestion (`/api/blueprints`)

### `POST /api/blueprints/upload`
Upload a blueprint file, parse it into a `NormalizedFloorplan`, and store it as a draft.

- **Input:** `multipart/form-data`
  - `file` (required) — the blueprint file. **Currently only `.dxf` is accepted.** Raster/PDF is planned (Phase 1) and will be auto-detected by extension.
- **Output (200):** `BlueprintSummary`
  ```json
  {
    "id": "2f0b0ec8-...",
    "filename": "floorplan.dxf",
    "status": "draft_ready",
    "floorplan": { /* NormalizedFloorplan, or null */ }
  }
  ```
  `status` is one of `"processing" | "draft_ready" | "confirmed"`.
- **Errors:**
  - **415** — unsupported file type (anything not `.dxf` today).
  - **422** — file parsed but no walls/rooms found, or malformed DXF.

### `GET /api/blueprints/{blueprint_id}`
Fetch the current state/summary of a stored blueprint.
- **Path param:** `blueprint_id` (UUID string from upload).
- **Output (200):** `BlueprintSummary` (the `floorplan` field is omitted/`null` on this route — use `/floorplan` for geometry).
- **Errors:** **404** — unknown blueprint id.

### `GET /api/blueprints/{blueprint_id}/floorplan`
Fetch the parsed geometry for a blueprint.
- **Path param:** `blueprint_id`.
- **Output (200):** `NormalizedFloorplan`.
- **Errors:** **404** — unknown id, or no floorplan parsed.

### `POST /api/blueprints/{blueprint_id}/confirm`
Lock the (human-verified) draft as ground truth. Queries are intended to run only against confirmed graphs.
- **Path param:** `blueprint_id`.
- **Output (200):** `{"id": "<blueprint_id>", "status": "confirmed"}`
- **Errors:** **404** — unknown id.

> **Planned (Phase 5):** a verification endpoint that accepts *edits* before confirm — e.g. `PATCH /api/blueprints/{id}` to rename/delete rooms, correct door links, and set `north_offset_deg` — plus an `openings` collection. The current `/confirm` only flips status.

---

## Mesh / 3D (`/api/mesh`)

### `POST /api/mesh/to-glb`
Extrude a stored blueprint's floorplan (or a built-in sample) into a 3D GLB, save it under `app/static/glb/`, and return its URL + geometry metadata.

- **Input:** JSON `MeshRequest`
  | Field | Type | Default | Notes |
  |---|---|---|---|
  | `blueprint_id` | string \| null | null | build from an uploaded blueprint |
  | `sample` | bool | false | build the synthetic sample instead |
  | `wall_height_m` | float | `2.7` | extrusion height for walls |
  | `wall_thickness_m` | float | `0.15` | wall centrelines are buffered by half this |
  | `include_floors` | bool | true | extrude room polygons as floor slabs |
  | `include_ceiling` | bool | false | add a ceiling slab at wall height |

  Provide either `blueprint_id` **or** set `sample: true`.
- **Output (200):** `MeshResponse`
  ```json
  {
    "id": "c52491d1-...",
    "url": "/static/glb/c52491d1-....glb",
    "triangle_count": 348,
    "wall_area_m2": 16.33,
    "floor_area_m2": 0.0,
    "bounds": [minx, miny, minz, maxx, maxy, maxz],
    "warnings": ["No room polygons; no floor slab generated."]
  }
  ```
  `bounds` is a 6-tuple in world metres (floor sits at `miny = 0`).
- **Errors:**
  - **400** — neither `blueprint_id` nor `sample` provided.
  - **404** — unknown blueprint id, or it has no parsed floorplan.
  - **422** — nothing to extrude (no walls and no rooms).

### `GET /api/mesh/sample/glb`
Build (and cache) the synthetic sample GLB and return the raw bytes. Used by the viewer as its default scene — no upload required.
- **Output (200):** raw GLB bytes, `Content-Type: model/gltf-binary`.

---

## Query (`/api/query`)

### `POST /api/query/`
Accept a natural-language (or transcribed voice) command, translate it into one of a fixed set of structured graph queries, and execute it deterministically against the SKG. **The LLM never computes geometry** — it only picks a template and fills slots; all math happens in `app/skg/engine.py`.

- **Input:** JSON `QueryRequest`
  ```json
  { "blueprint_id": "2f0b0ec8-...", "text": "find the path from kitchen to bedroom" }
  ```
- **Output (200):** `QueryResponse`
  ```json
  {
    "structured_query": {
      "intent": "find_path",
      "room_from": "kitchen",
      "room_to": "bedroom",
      "room": null
    },
    "result": { /* intent-specific; see below */ }
  }
  ```
  `intent` ∈ `"find_path" | "compute_area" | "check_egress" | "unknown"`. Slot fields (`room_from`, `room_to`, `room`) are filled per intent; unused slots are `null`.
  - `find_path` → `result` carries a `path` (ordered room ids) and hop count.
  - `compute_area` → `result` carries `area_m2` for the target room.
  - `check_egress` → `result` carries nearest-exit `path` and `hops`.
- **Status:** **501** today — the translator is not yet wired (returns `intent: "unknown"`), so this endpoint responds `501 {"detail": "Query translation not implemented yet."}`. **Planned in Phase 6** to be backed by Ollama with a rule-based fallback.

---

## Static assets (`/static`)

Mounted from `app/static/`. Notable paths:

- `/static/viewer.html` — the Three.js / WebXR viewer (orbit on desktop, tap-to-teleport in VR). Accepts query params:
  - `?src=<glb-url>` — load a specific GLB (e.g. the `url` from `POST /api/mesh/to-glb`).
  - `?name=<label>` — display label for the loaded model.
  - No params → loads the sample via `/api/mesh/sample/glb`.
- `/static/glb/<id>.glb` — generated meshes written by `POST /api/mesh/to-glb`. Regenerated on demand; gitignored.
- `/static/swagger-ui/…` — vendored Swagger UI assets for `/docs`.

---

## Planned endpoints (roadmap)

These are **not implemented yet**; they document the intended contract so clients can be written against them. Tracked in `WORKPLAN.md`.

| Phase | Method & path | Purpose | Likely input → output |
|---|---|---|---|
| 1 | `POST /api/blueprints/upload` (extended) | Accept raster PNG/JPG and PDF (rasterised at ~200 DPI) in addition to DXF; return pixel-unit floorplan + scale factor. | file → `BlueprintSummary` |
| 2 | `POST /api/blueprints/{id}/scale` | Set calibration from two clicked points + a known real distance → derives `pixels_per_meter`. | `{p1, p2, real_distance_m}` → updated floorplan |
| 3 | `POST /api/blueprints/{id}/rooms` | Derive room polygons via `polygonize` + OCR labels; attach door-gap adjacency edges to the graph. | — → populated graph |
| 4 | `GET /api/blueprints/{id}/openings` | Door/window openings detected from gaps and symbols. | — → `Opening[]` |
| 5 | `PATCH /api/blueprints/{id}` | Human verification edits: rename/delete rooms, correct doors, mark exits, set `north_offset_deg`. | edit ops → updated graph |
| 6 | `GET /api/blueprints/{id}/hud` | 2D HUD overlay payload: highlighted path/room over the source image for the last query. | — → overlay JSON |
| 8 | `GET /api/blueprints/{id}/glb` | Convenience alias returning the GLB bytes for a confirmed blueprint directly (vs. the `/api/mesh` POST round-trip). | — → GLB bytes |
| 9 | `GET /api/blueprints/{id}/egress` | SDG-11 egress report: distance to nearest exit, door widths below threshold, cited rule basis. | — → report JSON |
| 10 | `POST /api/site/report` | Site-aware daylight / ventilation / solar-screening report from lat/lon + north angle (see `WORKPLAN.md`, Phase 10). | site record → ranked findings |
