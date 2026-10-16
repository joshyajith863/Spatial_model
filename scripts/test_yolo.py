"""
Smoke test for YOLOv10-seg on a raster blueprint (WORKPLAN Day 1-2, step 1).

Goal: prove the YOLO toolchain runs end-to-end on CPU -- model downloads,
loads, and emits segmentation masks for an input image.

Honest caveat: pretrained COCO weights know nothing about walls/doors/rooms,
so detections here are plumbing proof only. Real floorplan segmentation needs
fine-tuning on CubiCasa5K (Day 2+). See WORKPLAN risk flags.

Usage:
    python scripts/test_yolo.py <image_path>
"""

import sys
from pathlib import Path


def main() -> int:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/uploads/375f6cc2-b7bf-44b2-b0c7-423e5fc4727f_b9ea330f65ca8a9f10be077a6af1bde7.jpg"
    )
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}")
        return 1

    print(f"Image: {image_path}")
    print(f"Loading YOLOv10-seg (downloads weights on first run)...")

    from ultralytics import YOLO

    # yolov10s-seg.pt = small segmentation model; auto-downloads from GitHub.
    model = YOLO("yolov10s-seg.pt")
    print(f"Model loaded: {model.info()}")

    results = model.predict(source=str(image_path), imgsz=640, conf=0.25, verbose=False)

    total = 0
    for r in results:
        names = r.names  # {0: 'person', ...}
        if r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                xyxy = [round(float(v), 1) for v in b.xyxy[0].tolist()]
                print(f"  {names.get(cls_id, cls_id):>12}  conf={conf:.2f}  box={xyxy}")
                total += 1
        n_masks = 0 if r.masks is None else len(r.masks.data)
        print(f"  -> masks detected: {n_masks}")

    print(f"\nTotal detections: {total}")
    print("NOTE: COCO classes only -- this validates the toolchain, not floorplan parsing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
