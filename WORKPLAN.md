# Spatial Copilot — Raster Floor Plan → 3D + Queryable Graph

**Goal:** turn a real blueprint (raster image or PDF) into a scaled 3D shell plus a
queryable Spatial Knowledge Graph, with a human-verification step that makes the
"deterministic reasoning" claim honest.

> Upload a raster/vector plan → extract walls → scale it → derive rooms + doors →
> confirm the graph → ask a question → get a geometrically correct answer drawn on a
> 2D/3D view.

This plan **builds on what already exists**: the DXF/vector-PDF path
(`app/parsers/dxf_parser.py`, `NormalizedFloorplan`), the SKG
(`app/skg/engine.py`), the mesh/GLB export, and the rule-based query parser
(`app/voice/translator.py`). Estimates assume **one person, part-time-to-full-time**;
they are rough.

---

## Scope

**In scope (Phases 0–6 = the demo)**
- Verify the existing foundation actually runs end-to-end
- Raster → walls parsing with OpenCV (no ML required)
- Scale calibration (manual, with a safe default)
- Room extraction + real adjacency from wall gaps (doors)
- Openings, floors, optional ceilings in the 3D mesh
- Human verification UI (the trust boundary)
- Query layer upgrade (Ollama translator + fuzzy room matching + 2D HUD path)

**Optional / later (Phases 7–10 — drop before rushing 0–6)**
- YOLO fine-tuning for perception
- WebXR 3D walkthrough
- SDG 11 egress check
- **Phase 10 — Site-aware daylight, ventilation, and solar screening** (full spec inlined below)

---

## Phase 0: Verify the foundation (1 day)

Nothing here has run outside the authoring sandbox — start here.

- [ ] Install requirements locally and start the server.
- [ ] Run one real DXF through `POST /api/mesh/to-glb` and open the result in `viewer.html`.
- [ ] Fix whatever breaks (likely: DXF layer names, `trimesh`/`manifold3d` install, the
      circular import in `app/api/query.py`).
- [ ] Push to GitHub on a clean branch; trim the README to match what's actually built.

**Done when:** a DXF upload shows a 3D model in the browser.

---

## Phase 1: Raster walls with OpenCV, no ML (3–4 days)

- [ ] **Ingest:** rasterize PDFs at ~200 DPI with PyMuPDF; load PNG/JPG directly.
- [ ] **Preprocess:** grayscale → adaptive threshold → deskew if needed.
- [ ] **Isolate walls:** estimate typical wall thickness via distance transform; use
      morphological opening sized to that thickness to keep thick strokes (walls) and drop
      thin ones (dimension lines, furniture, hatching). Remove text-sized connected components.
- [ ] **Vectorize:** extract contours → simplify with Shapely `simplify` → snap angles to
      0°/90° → merge collinear segments.
- [ ] **Output:** a `NormalizedFloorplan` in pixel units, fed to the existing mesh/graph builder.
- [ ] **Test set:** collect 8–10 plans (clean exports, scans, a few messy ones). Keep the
      same set for the whole project.
- [ ] **Set up the Evaluation table** (below) now; track it throughout.

**Done when:** ≥6 of 10 plans produce a recognizable 3D shell.

---

## Phase 2: Scale calibration (1–2 days)

- [ ] Viewer: click two points, enter the real distance → derive `pixels_per_meter`.
- [ ] API accepts that value; parser applies it **before** mesh and graph are built.
- [ ] Add a sensible default so the demo never produces a 40-meter bedroom.
- [ ] Later improvement: OCR a dimension string (`12'-6"`) and propose scale automatically.

**Done when:** a known room measures within 5% of its true area.

---

## Phase 3: Rooms and real adjacency (3–4 days)

This unblocks queries on real files — right now `rooms` is empty.

- [ ] Close small wall gaps; run Shapely `polygonize` on the wall network to get room
      faces; discard the outer face.
- [ ] PaddleOCR (lite) for room labels; assign each label to its containing polygon via
      point-in-polygon; name unlabeled rooms `room_3`, etc.
- [ ] Replace proximity adjacency with **door-gap detection**: a gap in a shared wall = a
      door = a graph edge (more accurate than the 0.3 m guess).
- [ ] Do the same for DXF via room layers or the same polygonize step.

**Done when:** `area of kitchen` and `path from X to Y` return correct answers on ≥5 test plans.

---

## Phase 4: Openings, floors, and ceilings in 3D (3–4 days)

- [ ] Extend the schema with an `Opening` type (position, width, door/window, owning wall).
- [ ] Detect doors/windows from gaps + simple symbols (door-swing arcs); add YOLO (Phase 7)
      only if this isn't enough.
- [ ] Cut openings with `manifold3d` booleans; add floor slabs from room polygons; make the
      ceiling optional.

**Done when:** the GLB visibly has doorways you can see through.

---

## Phase 5: Human verification UI (2–3 days)

This is what makes the "deterministic" claim honest — it moves the trust boundary to a
point a person has checked.

- [ ] Overlay detected walls/rooms on the original image in a canvas.
- [ ] Let the user rename/delete rooms, flag wrong doors, and re-click the scale.
- [ ] **Confirm** button locks the graph; queries are only allowed on confirmed graphs.
- [ ] Log corrections to disk (free training data later).

---

## Phase 6: Query layer upgrade (3–4 days)

- [ ] Wire Ollama with a small instruct model as the translator; keep the rule-based parser
      as a fallback when the model fails or is offline.
- [ ] Fuzzy-match room names with `rapidfuzz` ("the kitchen", "kitchen", "Kitchenette").
- [ ] Tag exits explicitly (user marks them during verification) instead of treating leaf
      nodes as exits.
- [ ] Draw the returned path on a 2D HUD over the plan.
- [ ] Optional: speech-to-text via the browser Web Speech API (a few lines on top).

**Done when:** a typed or spoken question highlights the correct route on screen.

---

## Phase 7: YOLO fine-tuning (optional, 4–6 days, parallelizable)

Only if Phases 1 and 4 aren't accurate enough on the test set.

- [x] **Convert CubiCasa5K → YOLO segmentation format**
      (`scripts/convert_cubicasa_yolo.py`). Pulls the `phungpx/cubicassa5k-coco` Parquet
      from HuggingFace (4,228 train / 748 valid), maps the source's 8 classes onto our
      `floor/wall/door/window` taxonomy (`--taxonomy segmenter`, default) or keeps the
      raw 8 (`--taxonomy source`), writes `images/` + `labels/` + `data.yaml`. Verified
      on a `--split valid --limit 6` smoke run.
- [ ] **Fine-tune YOLO11n-seg / YOLOv8n-seg** — script built
      (`scripts/train_yolo.py`, `yolo11n-seg.pt` default; *not YOLOv10*, which has no
      segmentation variant in Ultralytics). `--dry-run` validates config; CPU-only box so
      a real run is `--epochs` small + `--limit` slice first. **Not yet run.**
- [ ] **Evaluate vs. the existing segmenter, keep the better one per element type**
      (`scripts/eval_yolo.py`). Harness verified: per-class IoU + winner table, runs both
      the UNet (`app/cv/segmenter.py`) and a YOLO checkpoint on the same val images.
      Needs a trained YOLO `best.pt` to produce a real comparison (blocked on the step
      above).

---

## Phase 8: WebXR (4–5 days) — *3D pipeline built; HTTPS + Three.js vendoring open*

- [x] **Mesh builder** (`app/mesh/builder.py`): `NormalizedFloorplan` → extruded walls +
      room floor slabs → GLB (trimesh + shapely). Handles the y-flip in one place, centres
      the footprint on the origin, pins the floor to y=0 so the teleport plane lines up.
      Tested: sample plan and a real DXF both produce valid GLB.
- [x] **Endpoints** (`app/api/mesh.py`): `POST /api/mesh/to-glb` (from an uploaded
      blueprint or `sample=true`) and `GET /api/mesh/sample/glb`. Verified end-to-end.
- [x] **Viewer** (`app/static/viewer.html`): Three.js orbit + zoom for desktop, **Enter VR**
      with tap-to-teleport in a headset, model auto-framed, mesh/warning readout.
- [ ] **HTTPS** — set up `mkcert` (or a tunnel) **in week 1**, not on demo day. `navigator.xr`
      is undefined on insecure origins, so a phone on `http://192.168.x.x` silently falls
      back to desktop-only. See the *Running on your local network* section of the README.
- [ ] **Vendor Three.js** — the viewer pulls from the jsDelivr CDN; if the demo network
      blocks CDNs, download the four modules into `app/static/vendor/three/` and repoint the
      import map.

**Done when:** a DXF upload shows a 3D model in the browser, and a phone on HTTPS enters VR
and teleports.

---

## Phase 9: Egress check, SDG 11 only (2–3 days)

- [ ] Use explicit exits, door widths, and path lengths from the graph.
- [ ] Report distance to nearest exit; flag door widths below a threshold. **Cite the rule**
      you use (e.g. a travel-distance limit from your local code) so it isn't an invented number.
- [ ] Leave energy, daylight, and HVAC as stated roadmap items.

---

## Phase 10: Site-aware daylight, ventilation, and solar screening (2–3 weeks, optional)

**Depends on Phase 3 (rooms + windows) and Phase 4 (openings)** — windows are the core input.
Given a plan, a lat/lon, and a north angle, report where the house gains unwanted heat, where
it wastes daylight, and where ventilation is weak, then rank fixes that cut cooling and
lighting energy. **Every number comes from deterministic code; the LLM only explains them.**

**Inputs**

| Input | Source | Default if missing |
|---|---|---|
| Floor plan with rooms, walls, windows, doors | Existing parsers + verification UI | None (required) |
| Latitude / longitude | City search or map pin | None (required) |
| `north_offset_deg` | Compass dial or typed angle | 0, with a visible warning |
| Window sill and head heights | User or defaults | 0.9 m / 2.1 m, labeled as assumptions |
| Wall, roof, glass properties | Preset by construction type | Labeled as assumptions |
| Weather file | Typical-year dataset for the location | Required for Tier 2 |

**Deliverables:** site record (lat, lon, tz, `north_offset_deg`); facade summary (glazing area
+ window count per compass direction); per-room metrics (window-to-floor ratio, openable area,
exterior walls with openings, cross-ventilation flag); sun exposure (hours of direct sun per
window on the hottest/coolest/equinox days); shading check (overhang/fin blocks summer, admits
winter); and a ranked recommendation report. Every default is labeled; no savings figure
appears without a range and a stated basis.

- [ ] **Tier 1 — geometry rules:** site record; facade summary; per-room metrics. Apply the
      default sill/head heights and construction presets, all **labeled as assumptions**.
- [ ] **Tier 2 — sun + airflow:** sun-path math for hours of direct sun per window on the
      hottest/coolest/equinox days; shading check; simple ventilation estimates. Pull a
      typical-year weather file (verify the data source early; fall back to nearest station).
- [ ] **Report + LLM explanation:** ranked findings, each with its number, an impact
      **range**, and the assumption it rests on. LLM explanation restricted to the report's
      figures.
- [ ] **Stretch (Tier 3):** validate against EnergyPlus or Radiance on 2–3 houses and report
      the gap between estimates and simulation.

**Excluded:** certified compliance, guaranteed rupee/dollar savings, HVAC sizing, and anything
that needs a full dynamic simulation to be honest.

**Done when:** facade orientation correct on 3 plans (hand-checked), sun exposure within 10%
of a reference tool (e.g. Sun Surveyor or a published sun-path chart) on 5 windows,
cross-ventilation flags match manual review on 3 plans, and the full report runs in under 10 s
per plan.

---

## Evaluation (set up in Phase 1, track throughout)

| Metric | How |
|---|---|
| Wall detection | Hand-trace 3 plans, compare overlap |
| Room area error | Compare against known areas; target < 10% |
| Room count | Detected vs. actual |
| Path correctness | 10 queries on 3 plans, checked by hand |
| Latency | Seconds per upload; seconds per query |

Report these numbers in the results section, **including the failures** — more credible
than "it works."

---

## Timeline

| Weeks | Phases | Milestone |
|---|---|---|
| 1 | 0–2 | Raster plan → scaled 3D shell |
| 2–3 | 3–4 | Rooms, doors, working queries on real plans |
| 4 | 5–6 | **Demo-ready:** upload, verify, ask, see the path |
| 5–6 | 7–9 | Optional: YOLO, WebXR, egress check |

Week 4 is the demo. Everything after is polish / extra scope — drop it before rushing the
earlier phases.

---

## Main risks

| Risk | Mitigation |
|---|---|
| Messy scans break the heuristics | Scope the demo to clean plans and say so openly |
| Door detection is unreliable | Let the verification UI correct doors by hand |
| Ollama is slow or fails | Rule-based fallback is already built |
| WebXR fails over HTTP | Set up HTTPS in week 1, not on demo day |
| Scope creep | Anything outside Phases 0–6 is optional |

---

## Appendix — Deferred: Sustainability Advisor spec (carried from prior plan)

**Goal:** given a *confirmed* floorplan, tell the user how to make it more environmentally
sustainable. Output is a **concrete, ranked, actionable** list — not just a score. This is
the "SDG advisory" idea made explicit. Deferred; not part of Phases 0–9 above.

**Why it fits the architecture.** The LLM never does geometry — it only selects a query
template and fills slots. This model reads **deterministic metrics computed by the SKG**
and turns them into plain-language advice. Environmental math stays hallucination-free;
the LLM only explains and prioritizes.

### Inputs (all computed deterministically by `app/skg`, never guessed)

| Metric | Source | Sustainability angle |
|---|---|---|
| Window-to-wall ratio (WWR), per orientation | window + wall polygons | SDG 7/13 — daylight vs. heat loss/gain |
| Gross floor area, room areas | `compute_area()` | embodied + operational energy |
| Shape factor (surface-area-to-volume) | exterior wall polygon | thermal efficiency of the envelope |
| Egress path length, bottlenecks | `check_egress()` | SDG 11 — safety + accessibility |
| Daylight autonomy proxy (glazing per room) | window polygons ÷ room area | SDG 3/7 — lighting energy, wellbeing |
| Circulation-to-usable-area ratio | room adjacency graph | SDG 11 — wasted, conditioned space |

### Model role (decide on measured quality)

1. **Small local LLM via Ollama** — prompt with computed metrics + a fixed catalogue of
   improvement levers; have it select + rank + justify. No free-form geometry.
2. **Rule-based first** — deterministic mapping (e.g. `WWR < 0.15` on a south face →
   "increase glazing for passive solar gain") with templated phrasing. Cheaper, testable,
   zero-hallucination, and a fallback if the LLM adds friction.

**Recommendation:** ship rule-based first (just lookups over SKG metrics), then layer the
LLM on top for phrasing and cross-metric trade-off reasoning.

### Suggested improvement levers to encode

- **Envelope** — raise WWR on useful orientations; flag over-glazing on poor ones.
- **Orientation / passive solar** — south-facing glazing for free heat; shade hot faces.
- **Daylight** — deepen daylight into the floor plate; flag deep, artificially-lit rooms.
- **Compactness** — high shape factor → more envelope heat exchange per usable m².
- **Circulation** — long corridors / dead space are conditioned but unusable.
- **Egress / accessibility** — SDG 11: multiple exits, short travel distances.
- **Material / embodied-carbon hook** — stub for later; surface area feeds an embodied-CO₂
  estimate once a per-material factor exists.

### Definition of done for this model

One confirmed plan in → a ranked list of suggestions out, each tagged with the triggering
metric and the SDG it serves (7, 11, 13, 3), rendered on the 2D HUD.

### Out of scope for this model

- No HVAC duct routing or pressure-drop simulation (needs engineering data we lack).
- No generative redesign / auto-move-walls — advisory only; the human edits.
- No energy-code compliance certification — advisory, not a permit tool.
