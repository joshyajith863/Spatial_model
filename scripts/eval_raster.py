"""
Evaluate the Phase 1 raster wall extractor against CubiCasa5K ground truth.

The converted dataset's YOLO segmentation labels give a wall polygon per image;
we rasterise those to a binary mask and score our OpenCV wall isolation against
it with IoU / precision / recall. This is the "Evaluation table" WORKPLAN Phase
1 asks for — the numbers, including the failures.

Usage:
    python scripts/eval_raster.py --limit 30
    python scripts/eval_raster.py --limit 100 --save-overlays 5
    python scripts/eval_raster.py --images path/to/imgs --labels path/to/lbls

Writes runs/raster_eval/summary.json (+ overlay PNGs) and prints a table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse the YOLO-label -> mask helper (same ground truth the eval harness uses).
sys.path.insert(0, str(ROOT / "scripts"))
from eval_yolo import rasterize_label_txt  # noqa: E402

WALL_CLASS = 1  # floor=0 wall=1 door=2 window=3 in the segmenter taxonomy


def score(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    union = tp + fp + fn
    return {
        "iou": tp / union if union else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


def overlay(image_path: Path, pred: np.ndarray, gt: np.ndarray, out: Path) -> None:
    """Green = ground truth walls, red = predicted walls, yellow = both."""
    img = cv2.imread(str(image_path))
    if img is None:
        return
    if img.shape[:2] != pred.shape:
        img = cv2.resize(img, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA)
    canvas = img.copy()
    canvas[gt & ~pred] = (0, 200, 0)
    canvas[pred & ~gt] = (0, 0, 220)
    canvas[pred & gt] = (0, 215, 230)
    cv2.imwrite(str(out), canvas)


def main() -> int:
    p = argparse.ArgumentParser(description="Score the raster wall extractor vs CubiCasa5K labels")
    p.add_argument("--images", default=str(ROOT / "data" / "yolo" / "images" / "valid"))
    p.add_argument("--labels", default=str(ROOT / "data" / "yolo" / "labels" / "valid"))
    p.add_argument("--limit", type=int, default=30, help="number of images (0 = all)")
    p.add_argument("--out", default=str(ROOT / "runs" / "raster_eval"))
    p.add_argument("--save-overlays", type=int, default=0, help="save N overlay PNGs (0 = none)")
    p.add_argument("--wall-class", type=int, default=WALL_CLASS)
    args = p.parse_args()

    from app.cv.raster import isolate_walls, load_raster, preprocess

    images_dir, labels_dir = Path(args.images), Path(args.labels)
    pairs = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
            continue
        lbl = labels_dir / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        print("no image/label pairs found")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"evaluating wall extraction on {len(pairs)} images\n")
    print(f"{'image':<44} {'IoU':>6} {'Prec':>6} {'Rec':>6} {'ms':>6}")
    print("-" * 72)

    rows = []
    for i, (img_path, lbl_path) in enumerate(pairs):
        t0 = time.time()
        rgb = load_raster(img_path)
        ink, (orig_h, orig_w, factor) = preprocess(rgb)
        wall_mask, _thickness = isolate_walls(ink)
        elapsed = (time.time() - t0) * 1000.0

        # Canonical mask -> original image resolution for comparison.
        pred = cv2.resize(wall_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) > 0
        gt = rasterize_label_txt(lbl_path, orig_h, orig_w, 4)[args.wall_class]

        row = score(pred, gt)
        row.update(image=img_path.name, ms=round(elapsed, 1),
                   gt_px=int(gt.sum()), pred_px=int(pred.sum()))
        rows.append(row)
        print(f"{img_path.name[:44]:<44} {row['iou']:6.3f} {row['precision']:6.3f} "
              f"{row['recall']:6.3f} {row['ms']:6.0f}")

        if args.save_overlays and i < args.save_overlays:
            overlay(img_path, pred, gt, out_dir / f"overlay_{i:02d}.png")

    ious = [r["iou"] for r in rows]
    summary = {
        "images": len(rows),
        "iou_mean": round(float(np.mean(ious)), 4),
        "iou_median": round(float(np.median(ious)), 4),
        "iou_min": round(float(np.min(ious)), 4),
        "iou_max": round(float(np.max(ious)), 4),
        "precision_mean": round(float(np.mean([r["precision"] for r in rows])), 4),
        "recall_mean": round(float(np.mean([r["recall"] for r in rows])), 4),
        "ms_mean": round(float(np.mean([r["ms"] for r in rows])), 1),
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("-" * 72)
    print(f"IoU mean {summary['iou_mean']:.3f}  median {summary['iou_median']:.3f}  "
          f"min {summary['iou_min']:.3f}  max {summary['iou_max']:.3f}")
    print(f"precision {summary['precision_mean']:.3f}  recall {summary['recall_mean']:.3f}  "
          f"~{summary['ms_mean']:.0f} ms/image")
    print(f"\nsummary -> {out_dir / 'summary.json'}")
    if args.save_overlays:
        print(f"overlays -> {out_dir}/overlay_*.png (green=GT, red=predicted, yellow=both)")

    worst = sorted(rows, key=lambda r: r["iou"])[:5]
    print("\nworst images:")
    for r in worst:
        print(f"  {r['iou']:.3f}  {r['image'][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
