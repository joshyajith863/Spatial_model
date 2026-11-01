# Spatial Copilot

**A multimodal AI assistant that turns static 2D blueprints into interactive 3D environments and tactical overlays — no expensive hardware, no paid APIs.**

---

## The Problem

Traditional architectural inspection relies on manual spatial analysis and static CAD files. Recent visual LLMs can *inspect* floorplan images, but using a pure LLM for spatial reasoning introduces severe **spatial hallucinations** — wrong room measurements, broken pathfinding, incorrect building-code calculations.

## The Core Idea

Decouple **language understanding** from **spatial mathematics**. The LLM never guesses dimensions or routes. Instead:

1. Computer vision extracts the exact blueprint layout.
2. The layout is stored as a deterministic **Spatial Knowledge Graph (SKG)**.
3. The LLM is used *only* to translate human voice commands into structured graph queries.

```
[ 2D Blueprint PDF / Image ]
            │
            ▼
┌──────────────────────────────┐
│  Perception Pipeline         │ ──► walls, doors, windows, text annotations
│  (YOLOv10-seg + PaddleOCR)   │
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│  Spatial Knowledge Graph     │ ──► ground truth for paths, area math, code rules
│  (NetworkX + Shapely)        │
└───────────┬──────────────────┘
            │
      ┌─────┴─────────────────────────────┐
      ▼                                   ▼
┌─────────────────────┐      ┌─────────────────────────┐
│  2D Tactical HUD     │      │  WebXR 3D Environment   │
│  • Voice queries     │      │  • Auto-extruded GLB    │
│  • A* pathfinding    │      │  • Voice teleportation  │
└─────────────────────┘      └─────────────────────────┘
```

## Why This Works

The reasoning layer (SKG math) is deterministic and hallucination-free by construction — it's plain graph theory and geometry, not model inference. The perception layer (CV extraction) is probabilistic and *can* misread geometry, so extracted layouts are surfaced as an editable overlay for human confirmation before being locked into the graph. This is what makes the "zero hallucination" claim honest: it describes the reasoning step, not the perception step.

## Pipeline

| Stage | What runs | Cost profile |
|---|---|---|
| Ingestion | OpenCV deskew/normalize | Milliseconds, once per upload |
| Perception | YOLOv10-seg (detection + segmentation in one pass) + PaddleOCR (lite model) | Seconds, once per upload |
| Human verification | Editable overlay, confidence-flagged detections | One-time, per upload |
| Graph construction | NetworkX + Shapely | Milliseconds, cached after build |
| Voice → query | Small quantized text LLM, constrained to fixed query templates (no VLM per command) | Sub-second, every interaction |
| Pathfinding | A* over the graph | Microseconds |
| Rendering | Pre-extruded GLB (built once) + 2D HUD overlay | No per-query cost |

The expensive steps (perception, graph build) run once per blueprint. Everything on the interactive path — voice queries, pathfinding, rendering — is cheap, cached, or template-constrained.

## Key Components

- **Perception Engine** — YOLOv10-seg + PaddleOCR convert raster blueprints into vector geometry (walls, doors, windows, room labels, dimension strings).
- **Spatial Knowledge Graph (SKG)** — computes floor areas, door clearances, and egress paths mathematically. Built with `NetworkX` (topology) and `Shapely` (geometry).
- **Voice-Driven Query Translator** — a small local LLM maps natural-language commands (*"find the evacuation route from the kitchen"*) to structured graph queries (`find_path(from, to)`, `compute_area(room)`, `check_egress(room)`). It never touches the math directly.
- **Dual Display System**
  - *2D Tactical HUD* — animated flow paths, safety exits, room highlights over the source plan.
  - *WebXR 3D Pipeline* — auto-extruded `.glb` mesh, viewable via mobile WebXR, with voice-controlled teleportation to avoid vection-induced motion sickness.
- **SDG Advisory Engine** — automated checks against UN Sustainable Development Goals:
  - **SDG 11** (Resilient Cities) — egress routes, fire exits, accessibility.
  - **SDG 7 & 13** (Clean Energy & Climate) — window-to-wall ratio, daylight autonomy, HVAC duct pressure drops.
  - **SDG 3** (Good Health) — ventilation pathways, circulation bottlenecks.

## Scale Calibration

Real-world units are recovered via a fallback chain:
1. Parse an explicit scale annotation (e.g. `1/4" = 1'-0"`) if present.
2. Otherwise, detect a dimension line + its OCR'd length string, measure its pixel length, and derive a pixel-to-unit ratio.
3. Otherwise, prompt the user to confirm one known measurement manually (e.g. a door width).

## Academic Foundations

- **Perception** — *CubiCasa5K* (Kalervo et al., 2021)
- **Graph topology** — *HouseGAN* (Nauata et al., 2020)
- **3D lifting** — *Plan2Scene* (Cruz et al., 2021)
- **Generative benchmarks** — *HouseDiffusion* (Shabani et al., CVPR 2023), *LayoutGPT* (Feng et al., 2023)

## Status

The reasoning core (SKG geometry) and the 3D/WebXR pipeline are built and verified; the
raster-perception and query layers are still being wired up. Contributions and issue reports
welcome.

### What works today
- **SKG geometry engine** (`app/skg/engine.py`) — `add_room`, `compute_area`, `find_path`,
  `check_egress` over NetworkX + Shapely. The deterministic math core.
- **DXF ingestion** (`app/parsers/dxf_parser.py`) — parses walls/rooms/bounds/units into a
  `NormalizedFloorplan`. Verified on a real blueprint (257 walls extracted).
- **3D / WebXR pipeline** — `app/mesh/builder.py` extrudes a floorplan to a GLB;
  `POST /api/mesh/to-glb` + `GET /api/mesh/sample/glb` serve it; `app/static/viewer.html`
  renders it with orbit (desktop) and tap-to-teleport (VR). Verified end-to-end with both the
  sample and a real DXF.
- **Perception (CPU-only)** — `app/cv/segmenter.py` runs a CubiCasa5K-trained
  ResNet34-UNet (MIT, `Yytsi/floorplan-to-3d-walls`) for walls/doors/windows/floor with no
  GPU (~7 s/image on CPU after a one-time ~44 s model load).
- **`/docs`** — Swagger UI served from local assets, so it renders on networks that block CDNs.

### Known gaps / risks
- **Rooms are empty on real files.** The DXF parser infers rooms from wall loops, but test
  files with no room-boundary layer yield 0 rooms. Fix is Phase 3 (polygonize the wall
  network + OCR labels) in `WORKPLAN.md`.
- **Scale is uncalibrated.** One test DXF parses to a ~1 m footprint — units/scale are wrong
  for some inputs. Fix is Phase 2 (manual two-point calibration + a safe default).
- **Query layer is a stub.** `POST /api/query/` returns 501 (translator + SKG dispatch not
  wired yet).
- **Model drift.** The pretrained segmenter was trained on Nordic residential CAD; US-style
  scans misread (door count high, wall count low). Tune thresholds or fine-tune (Phase 7).
- **CPU-only throughput.** Segmentation ~7 s/image — fine one-at-a-time, not for batch.

### Measured environment
Python 3.12.10, Windows, Intel Iris Xe (no CUDA); torch 2.14.1+cpu, torchvision 0.29.1+cpu,
segmentation-models-pytorch 0.5.0.

## Running on your local network

No deployment needed — this runs on one machine; other devices (phone for WebXR, laptop for
the HUD) connect over Wi-Fi.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

`--host 0.0.0.0` is what makes it reachable from other devices on the LAN — `127.0.0.1`
would only work on the host machine. Find your LAN IP with `ipconfig` (Windows) or
`ifconfig` (macOS/Linux); look for `192.168.x.x`.

- Host machine: `http://localhost:8000`
- Other devices: `http://<your-local-ip>:8000`
- API docs: `http://<your-local-ip>:8000/docs`
- 3D viewer: `http://localhost:8000/static/viewer.html`

The viewer loads a GLB automatically — the built-in sample by default, or your own via
`viewer.html?src=<url>` (upload a DXF → `POST /api/mesh/to-glb` → pass the returned `url`).
Desktop shows orbit + zoom; in a headset press **Enter VR**, point at the floor, and pull the
trigger to teleport.

### HTTPS is required for WebXR on phones (not optional)

`navigator.xr` is **undefined on insecure origins**, so a phone hitting
`http://192.168.x.x:8000` silently falls back to desktop-only. This is the most common
demo-day failure — fix it **before** demo day. `localhost` counts as secure, so desktop VR
works over `http://localhost:8000` with no cert. For a phone you need a real secure context:

```bash
# 1. Install mkcert (once) and create a locally-trusted cert for your machine:
choco install mkcert          # or: winget install mkcert   (Windows)
mkcert -install
mkcert 192.168.x.x localhost  # your actual LAN IP + localhost

# 2. Run uvicorn over TLS:
uvicorn app.main:app --host 0.0.0.0 --port 8000 \
  --ssl-certfile cert.pem --ssl-keyfile key.pem
```

The phone must trust the cert — AirDrop `rootCA.pem` to an iOS device and enable it under
Settings → General → About → Certificate Trust Settings. If that's too fiddly, a tunnel
(`cloudflared tunnel --url http://localhost:8000` or `ngrok http 8000`) gives a public HTTPS
URL but needs internet; prefer `mkcert` on a local-only network.

### Demo notes
- All presenting devices must be on the **same Wi-Fi** — this won't work across networks or
  on mobile data. Some corporate/campus/guest networks block device-to-device traffic; a
  phone hotspot or home router is most reliable.
- The 3D viewer pulls Three.js from the jsDelivr CDN. If the demo network blocks external
  CDNs (the same reason Swagger UI is vendored locally), download `three.module.js`,
  `GLTFLoader.js`, `OrbitControls.js`, and `VRButton.js` into `app/static/vendor/three/` and
  point the import map in `viewer.html` at them.

### Repo gotcha: stale `__pycache__` after branch switches
We were once bitten by a "missing" parser that wasn't — after switching branches, git removed
the `.py` source but a leftover `__pycache__/*.pyc` made it look present, while the real code
lived unmerged on another branch. If code seems to exist but won't import, check the actual
branch has the `.py` file (`.pyc` files are gitignored; never rely on them).

## API

See [`API.md`](API.md) for the full REST reference — every endpoint with input/output
formats, plus the planned (not-yet-implemented) routes. Interactive Swagger UI is at `/docs`.

## License

TBD.
