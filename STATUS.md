# Spatial Copilot — Status

_Last updated: 2026-11-05. Source of truth for "what works right now" — keep it in sync with
`README.md` (user-facing status table) and `WORKPLAN.md` (the full plan)._

**Verify any claim here yourself:**

```bash
pytest            # 27 tests
uvicorn app.main:app --port 8000   # then open http://localhost:8000/docs
```

---

## ✅ Completed

### Foundation
- [x] **App boots** — `uvicorn app.main:app` starts; verified `/health`, `/docs`, `/openapi.json`,
      `/static/viewer.html` all return 200.
- [x] **Test suite runs** — `tests/` + `pytest.ini`, **27 passing**: engine, mesh, translator,
      floorplan→SKG bridge, API round-trip (incl. error paths).
- [x] **Circular import fixed** — `StructuredQuery` moved to `app/models.py`, so `app/voice/`
      no longer imports the HTTP layer (was flagged in `WORKPLAN.md` Phase 0).
- [x] **`requirements.txt` regrouped** (web / 3D / perception / YOLO / tests) + `.gitignore`
      extended (`.pytest_cache`, `.ruff_cache`, `.mypy_cache`, `.env`).
- [x] **Docs consolidated** — `README.md`, `API.md`, `WORKPLAN.md` updated to match code
      (the query endpoint is no longer documented as a 501 stub).
- [x] **Nested duplicate project merged in** — the structure from `Spatial_model/Spatial_model`
      (tests, SKG store, rule-based translator, floorplan bridge) folded into this repo.

### Reasoning core (deterministic)
- [x] **SKG engine** (`app/skg/engine.py`) — `add_room`, `add_door`, `mark_exit`,
      `resolve_room` (fuzzy-ish id/label match), `compute_area`, `find_path`, `check_egress`
      (explicit exits, leaf-node fallback flagged `provisional_exits`).
- [x] **Graph store** (`app/skg/store.py`) — in-memory `blueprint_id -> SKG`.
- [x] **Floorplan → SKG bridge** (`app/skg/from_floorplan.py`) — rooms become nodes,
      adjacency by proximity (**placeholder**: no real door detection yet).
- [x] **Query layer end-to-end** (`POST /api/query/`) — text → translator → deterministic SKG
      dispatch. No longer a 501 stub; 404 (no graph) / 422 (unparseable, unknown room,
      no path) behave correctly.
- [x] **Translators** — regex rule-based (`app/voice/rule_based_translator.py`, default, offline)
      and Ollama LLM (`app/voice/translator.py`, opt-in `USE_LLM=1`, always falls back to rules).

### Ingestion → 3D
- [x] **DXF parser** (`app/parsers/dxf_parser.py`) → `NormalizedFloorplan` (the single IR).
- [x] **Mesh builder** (`app/mesh/builder.py`) — extrudes walls + room floors → GLB, one
      centralised y-flip, footprint centred, floor pinned to y=0 for VR teleport. Degenerate
      walls tolerated; empty plan raises.
- [x] **Mesh endpoints** — `POST /api/mesh/to-glb`, `GET /api/mesh/sample/glb`.
- [x] **Blueprint ingestion** — `POST /api/blueprints/upload` (DXF-only, 415 otherwise,
      rejects path-traversal filenames), `GET .../floorplan`, `POST .../confirm`.
- [x] **Confirm builds the graph** — `/confirm` now runs `build_skg()` and registers it in the
      store, so a confirmed blueprint is immediately queryable (409 if nothing to confirm).
- [x] **WebXR viewer** (`app/static/viewer.html`) — orbit/zoom on desktop, Enter VR +
      tap-to-teleport in a headset; sample GLB needs no upload.
- [x] **Swagger UI served locally** (`/docs`) — no CDN dependency.

### Perception (built; CPU-only)
- [x] **Segmenter** (`app/cv/segmenter.py`) — CubiCasa5K ResNet34-UNet, no GPU required.
- [x] **OCR** (`app/cv/ocr.py` + `POST /api/blueprints/ocr`) — PaddleOCR, returns text,
      confidence, and pixel-space boxes for the verification overlay.
- [x] **YOLO dataset prep** — `scripts/convert_cubicasa_yolo.py` (CubiCasa5K → YOLO seg),
      verified on a `--split valid --limit 6` smoke run.
- [x] **Train / eval harnesses built** — `scripts/train_yolo.py` (`--dry-run` validated),
      `scripts/eval_yolo.py` (per-class IoU + winner table).

---

## 🚧 Remaining — required for the demo (WORKPLAN Phases 0–6)

### Phase 0 — Verify the foundation
- [ ] Run **one real DXF** through `POST /api/mesh/to-glb` and open it in `viewer.html`.
      (Code path is tested synthetically; a real file has not been pushed through in this session.)
- [ ] Push to GitHub on a clean branch. **Current work is uncommitted** — see "Open work" below.

### Phase 1 — Raster walls with OpenCV
- [ ] Ingest PNG/JPG directly + rasterise PDFs (PyMuPDF) at ~200 DPI.
- [ ] Preprocess (grayscale → adaptive threshold → deskew).
- [ ] Isolate walls (distance-transform thickness estimate, morphological opening).
- [ ] Vectorize (contours → Shapely simplify → snap 0°/90° → merge collinear).
- [ ] Collect the 8–10 plan **test set**; set up the **evaluation table**.

### Phase 2 — Scale calibration
- [ ] Two-point click calibration in the viewer → `pixels_per_meter`.
- [ ] API accepts it; parser applies it **before** mesh + graph build.
- [ ] Safe default so the demo never produces a 40 m bedroom.

### Phase 3 — Rooms and real adjacency  ← blocks queries on real files
- [ ] Shapely `polygonize` on the wall network → room faces (currently **rooms are empty** on
      real files with no room layer).
- [ ] PaddleOCR labels assigned by point-in-polygon; auto-name the unlabelled ones.
- [ ] Replace proximity adjacency with **door-gap detection** (the current 0.3 m guess).

### Phase 4 — Openings, floors, ceilings in 3D
- [ ] `Opening` schema type (position, width, door/window, owning wall).
- [ ] Detect doors/windows from gaps + swing arcs.
- [ ] Cut openings with `manifold3d` booleans; optional ceiling.

### Phase 5 — Human verification UI
- [ ] Canvas overlay of detected walls/rooms on the source image.
- [ ] Rename/delete rooms, flag wrong doors, re-click scale.
- [ ] **Confirm** gates queries (endpoint exists; the *editing* step does not).
- [ ] Log corrections to disk.

### Phase 6 — Query layer upgrade
- [ ] Wire Ollama as the translator (code exists; **untested against a live Ollama**).
- [ ] Fuzzy room matching via `rapidfuzz`.
- [ ] Tag exits explicitly during verification instead of leaf-node fallback.
- [ ] Draw the returned path on a 2D HUD over the plan.
- [ ] Optional: browser Web Speech API for speech-to-text.

---

## ⬜ Optional / later (Phases 7–10)

- [ ] **Fine-tune YOLO** — script built, **never actually run** (`--epochs` small first).
- [ ] **Evaluate YOLO vs. existing segmenter** — harness verified, blocked on a trained
      `best.pt`.
- [ ] **HTTPS for WebXR** — `mkcert` setup, do it in week 1, not on demo day
      (`navigator.xr` is undefined on insecure origins).
- [ ] **Vendor Three.js** — viewer still pulls from jsDelivr CDN.
- [ ] **SDG-11 egress report** — cite the code basis, don't invent numbers.
- [ ] **Phase 10 — daylight / ventilation / solar screening** — spec written, not started
      (needs Phases 3 + 4 first).
- [ ] **Sustainability Advisor** — deferred spec in `WORKPLAN.md` appendix; ship rule-based
      first, LLM phrasing later.

---

## ⚠️ Known gaps / risks

- [ ] **Rooms are empty on real files** — DXF parser infers rooms from wall loops; test files
      without a room-boundary layer yield 0 rooms. Fix = Phase 3.
- [ ] **Scale is uncalibrated** — one test DXF parsed to a ~1 m footprint. Fix = Phase 2.
- [ ] **Adjacency is a guess** — rooms within 0.3 m are assumed connected; no door data.
- [ ] **Egress exits are provisional** — leaf rooms stand in unless exits are explicitly marked;
      results are labelled `provisional_exits: true`.
- [ ] **Perception model drift** — pretrained segmenter was trained on Nordic residential CAD;
      US-style scans misread (door count high, wall count low).
- [ ] **CPU-only throughput** — segmentation ~7 s/image; fine one-at-a-time, not for batch.
- [ ] **No auth, open CORS, in-memory storage** — local-network demo only, never expose publicly.
- [ ] **`fitz` (PyMuPDF) not installed in this venv** — vector-PDF ingestion needs
      `pip install pymupdf`; raster/PDF ingest is Phase 1 anyway.

---

## 📌 Open work (uncommitted in git)

Consolidation of the nested `Spatial_model/Spatial_model` project into this repo — **all tests
passing, not yet committed**:

- Modified: `.gitignore`, `API.md`, `README.md`, `WORKPLAN.md`, `requirements.txt`,
  `app/api/blueprints.py`, `app/api/query.py`, `app/models.py`, `app/skg/engine.py`,
  `app/voice/translator.py`
- Added: `pytest.ini`, `tests/` (27 tests), `app/skg/store.py`, `app/skg/from_floorplan.py`,
  `app/voice/rule_based_translator.py`
- Untracked leftovers: `Spatial_model/` (the old nested copy — delete once you're happy),
  `normalized-floorplan.patch`
