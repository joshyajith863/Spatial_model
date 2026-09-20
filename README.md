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

Early-stage / architecture design. Contributions and issue reports welcome.

## License

TBD.
