# Spatial Copilot — 2-Week MVP Work Plan

**Goal:** a working, demoable core loop — not the full system.

> Upload a blueprint → see a draft graph → confirm it → type a question → get a geometrically correct answer, shown on a 2D overlay.

That's the whole architectural claim (deterministic reasoning over AI-assisted perception) proven in miniature.

---

## Scope

**In scope**
- CV perception — basic, one blueprint style
- Spatial Knowledge Graph (already implemented in `app/skg/engine.py`)
- Human-verification step — simplified (list + confirm, not a full editor)
- Query translator — **text input**, not voice yet
- One query type end-to-end (`find_path`), second if time allows (`compute_area`)
- 2D HUD overlay only

**Out of scope (defer to phase 2)**
- WebXR / 3D pipeline
- SDG advisory engine
- Real voice/speech input
- Multi-style blueprint generalization
- Automatic scale-bar detection (use manual calibration fallback instead)

---

## Week 1 — Perception → Graph

### Days 1–2 (Mon–Tue): CV pipeline, minimum viable
- [ ] Get YOLOv10-seg running (pretrained or fine-tuned on a CubiCasa5K subset — don't train from scratch)
- [ ] Get PaddleOCR (lite model) reading room labels
- [ ] Pick **one** test blueprint, get the pipeline to output *something* — rough room polygons + labels. Accuracy doesn't matter yet, pipeline continuity does.

### Days 3–4 (Wed–Thu): Scale calibration + graph construction
- [ ] Implement manual calibration fallback (click two points, enter known length) — skip automatic scale-bar detection
- [ ] Wire CV output → `SpatialKnowledgeGraph.add_room()` / `add_door()`
- [ ] Test: one blueprint in → populated graph out → `compute_area()` returns plausible numbers

### Day 5 (Fri): Integration checkpoint
- [ ] Wire `app/cv/pipeline.py::run()` into `POST /api/blueprints/upload` for real
- [ ] End-to-end test: upload → draft graph exists, inspectable via `GET /api/blueprints/{id}`
- [ ] **🎯 Milestone: raw pipeline works end-to-end, even if ugly**

---

## Week 2 — Verification → Query → Display

### Days 6–7 (Mon–Tue): Verification overlay (simplified)
- [ ] Simple read-only list view ("Detected: Kitchen, Bedroom, Hallway — confirm?") instead of a draggable-polygon editor
- [ ] Wire `POST /api/blueprints/{id}/confirm` to lock the graph as ground truth

### Days 8–9 (Wed–Thu): Query translator + one query type
- [ ] Implement `app/voice/translator.py::parse()` via a local LLM (Ollama — fastest path, no custom quantization work), **text input only**
- [ ] Support `find_path` end-to-end first; add `compute_area` if time allows
- [ ] Wire `app/skg/engine.py::execute()` to dispatch real queries
- [ ] Test: "find the path from kitchen to hallway" → correct room sequence returned

### Day 10 (Fri): 2D HUD + demo polish
- [ ] Minimal HTML/canvas overlay: show blueprint image, highlight path/room from last query response
- [ ] Dry-run the full demo flow 2–3 times: upload → confirm → query → see result
- [ ] Bug-fix buffer — no new features today

---

## Definition of "done" (Day 10)

One blueprint, uploaded through the browser, produces a graph a human confirms, and a typed natural-language question returns a geometrically correct answer rendered visually on the 2D HUD.

---

## Risk flags

| Risk | Mitigation |
|---|---|
| CV accuracy eats the whole two weeks | If YOLOv10-seg isn't cooperating by end of Day 2, fall back to a manually-traced/hardcoded graph for the demo blueprint. Swap in real CV after the deadline. |
| Local LLM setup friction | Budget a buffer. If Ollama stalls, fall back to a rule-based keyword parser (`"from X to Y"` regex) as an acceptable placeholder. |

---

## Explicitly deferred to post-demo

- WebXR 3D pipeline
- SDG Advisory Engine (egress, WWR, daylight, HVAC, ventilation)
- Real speech-to-text voice input
- Automatic scale-bar/dimension-line detection
- Multi-style blueprint generalization
- Draggable-polygon human verification editor
