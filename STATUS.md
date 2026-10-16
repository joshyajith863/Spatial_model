# Spatial Copilot — Project Status

> Living to-do list. Checked = done and verified. Unchecked = not started or stubbed.
> Last updated: 2026-10-16 (branch: `experiment`)

## What actually works today

- [x] **Floorplan perception — structural segmentation.** `app/cv/segmenter.py` loads a
      CubiCasa5K-trained ResNet34-UNet (MIT, `Yytsi/floorplan-to-3d-walls`) and segments
      walls / doors / windows / floor on CPU. Verified on a real blueprint: 20 walls,
      45 doors, 4 windows, ~7s per image (one-time ~44s model load).
- [x] **Perception model runs on CPU-only hardware** — no CUDA/GPU required. Uses the
      CPU torch wheels (~190 MB, not the 2.5 GB CUDA stack).
- [x] **SKG geometry engine.** `app/skg/engine.py` — working `add_room`, `compute_area`,
      `find_path`, `check_egress` over NetworkX + Shapely. The deterministic math core.
- [x] **Model weights fetched reproducibly.** `scripts/download_weights.py`; weights are
      gitignored (97 MB) with a documented download step.
- [x] **`/docs` fix (all branches).** Swagger UI served locally instead of the jsdelivr
      CDN, so the docs page renders on networks that block external CDNs.
- [x] **Repo hygiene.** Branches consolidated to `main`; `experiment` branch created for
      CV work; `.gitignore` added (this branch previously had none).

## Next up — the critical path to a demoable loop

- [ ] **Room labels via OCR.** PaddleOCR (lite) to read `kitchen`, `great room`,
      `master bedroom`, dimension strings. The segmenter does *semantic* (not instance)
      segmentation, so rooms/labels must come from OCR. **Unblocks everything downstream.**
- [ ] **Preprocess step.** Deskew / normalize / upscale low-res scans in `app/cv/pipeline.py`.
- [ ] **Wire CV output → SKG.** Connect segmenter + OCR into `SpatialKnowledgeGraph.add_room()`
      / `add_door()`. This is `pipeline.run()` — currently a 15-line stub.
- [ ] **Scale calibration.** Manual fallback (click two points, enter known length) so
      `compute_area()` returns real-world units instead of pixels.
- [ ] **Populate the draft graph on upload.** `POST /api/blueprints/upload` should return a
      real draft graph instead of requiring DXF-only input.
- [ ] **Verification step (simplified).** Read-only list ("Detected: Kitchen, Bedroom,
      Hallway — confirm?") wired to `POST /api/blueprints/{id}/confirm`.

## Query + display layer

- [ ] **Query translator.** `app/voice/translator.py::parse()` — local LLM (Ollama), text
      input only. Regex fallback (`"from X to Y"`) is acceptable per the WORKPLAN.
- [ ] **SKG dispatch.** `app/skg/engine.py::execute()` — currently `NotImplementedError`.
- [ ] **Query endpoint end-to-end.** `POST /api/query/` currently returns 501.
- [ ] **One query type working.** `find_path` first; `compute_area` if time allows.
- [ ] **2D HUD overlay.** Minimal HTML/canvas — show blueprint, highlight path/room from
      the last query response.

## Explicitly out of scope for the MVP (phase 2)

- [ ] WebXR / 3D pipeline (auto-extruded GLB)
- [ ] SDG advisory engine — see `WORKPLAN.md` for the sustainability advisory model
- [ ] Real voice/speech input
- [ ] Multi-style blueprint generalization (hand-drawn, commercial)
- [ ] Automatic scale-bar / dimension-line detection
- [ ] Draggable-polygon human verification editor
- [ ] Fine-tuning a floorplan model on our own data (the pretrained model covers the MVP)

## Known gaps / risks

- **DXF path lives on another branch.** `app/parsers/dxf_parser.py` exists only on
  `feature/normalized-floorplan`, not merged into `experiment`/`main`.
- **Model drift.** The pretrained model was trained on Nordic residential CAD plans; our
  test image is a US-style scan. Door count looks high (45) and wall count low (20) —
  tune `CLOSING_KERNEL_PX` / `MIN_POLYGON_AREA_PX`, or fine-tune, if quality matters.
- **OCR compatibility.** numpy is at 2.5.4; PaddlePaddle has a history of numpy-2.x
  friction. May need a version pin when we add OCR.
- **CPU-only throughput.** Segmentation is ~7s/image on CPU. Fine for one blueprint at a
  time; not for batch. Fine-tuning on CPU would be hours/epoch — avoid unless measured.

## Measured environment

- Python 3.12.10, Windows, Intel Iris Xe (no CUDA)
- torch 2.14.1+cpu, torchvision 0.29.1+cpu
- segmentation-models-pytorch 0.5.0, safetensors 0.8.0
- Disk free: ~102 GB
