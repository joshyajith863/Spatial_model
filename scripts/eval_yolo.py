"""
Evaluate a trained YOLO segmenter against the pretrained floorplan segmenter.

Phase 7 of the WORKPLAN: "Evaluate against the heuristic pipeline on your test
set. Keep whichever is better per element type." This is the harness that
produces that number — it does NOT train anything.

For each model and each class (floor/wall/door/window) it rasterises predicted
polygons to a binary mask and computes IoU against the ground-truth mask derived
from the YOLO labels, then reports per-class IoU + a mean. A model that scores
higher on a class wins that element type.

Ground truth comes from the converted dataset's label files (same source as
training), so scores are directly comparable across models.

Usage:
    # both models, first N val images:
    python scripts/eval_yolo.py --data data/yolo/data.yaml --limit 50

    # YOLO checkpoint only:
    python scripts/eval_yolo.py --data data/yolo/data.yaml --yolo runs/segment/cubicasa/weights/best.pt

    # skip the UNet (e.g. if weights/ absent):
    python scripts/eval_yolo.py --data data/yolo/data.yaml --no-unet

Outputs a per-class IoU table and writes JSON to --out for the results section.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make `app.*` importable when run as `python scripts/eval_yolo.py` (for the UNet).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLASSES = ["floor", "wall", "door", "window"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def rasterize_label_txt(path: Path, h: int, w: int, n_classes: int) -> list[np.ndarray]:
    """YOLO-seg label file -> per-class binary masks (h, w)."""
    masks = [np.zeros((h, w), dtype=bool) for _ in range(n_classes)]
    if not path.exists():
        return masks
    from PIL import Image, ImageDraw

    canvases = [Image.new("1", (w, h), 0) for _ in range(n_classes)]
    drew = [False] * n_classes
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        cls = int(parts[0])
        if not 0 <= cls < n_classes:
            continue
        coords = [float(v) for v in parts[1:]]
        pts = [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords) - 1, 2)]
        if len(pts) >= 3:
            ImageDraw.Draw(canvases[cls]).polygon(pts, fill=1)
            drew[cls] = True
    for i in range(n_classes):
        if drew[i]:
            masks[i] = np.array(canvases[i], dtype=bool)
    return masks


def masks_from_polygons(polys, h: int, w: int) -> list[np.ndarray]:
    """Ultralytics result polygons (list per class, each a list of arrays) -> masks."""
    from PIL import Image, ImageDraw

    out = []
    for cls_polys in polys:
        canvas = Image.new("1", (w, h), 0)
        any_drawn = False
        for poly in cls_polys:
            arr = np.asarray(poly)
            if arr.ndim == 2 and len(arr) >= 3:
                ImageDraw.Draw(canvas).polygon([tuple(p) for p in arr], fill=1)
                any_drawn = True
        out.append(np.array(canvas, dtype=bool) if any_drawn
                   else np.zeros((h, w), dtype=bool))
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else float("nan")


def iter_images(img_dir: Path, limit: int):
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    return files[:limit] if limit else files


def eval_yolo(yolo_path: str, img_dir: Path, lbl_dir: Path, classes: list[str],
              limit: int) -> dict:
    from ultralytics import YOLO

    n_classes = len(classes)
    model = YOLO(yolo_path)
    per_class_iou: dict[str, list[float]] = {name: [] for name in classes}
    n = 0

    for img in iter_images(img_dir, limit):
        from PIL import Image
        w, h = Image.open(img).size
        gt = rasterize_label_txt(lbl_dir / (img.stem + ".txt"), h, w, n_classes)

        res = model.predict(img, verbose=False, imgsz=640)[0]
        # Group masks by predicted class id.
        pred_by_cls: list[list] = [[] for _ in range(n_classes)]
        if res.masks is not None and len(res.masks):
            cls_ids = res.boxes.cls.tolist()
            for poly, cid in zip(res.masks.xy, cls_ids):
                cid = int(cid)
                if 0 <= cid < n_classes:
                    pred_by_cls[cid].append(poly)
        pred = masks_from_polygons(pred_by_cls, h, w)

        for i in range(n_classes):
            name = classes[i]
            # Score only where either side has pixels; absent-on-both is skipped
            # so empty classes don't inflate the mean.
            if gt[i].any() or pred[i].any():
                per_class_iou[name].append(iou(gt[i], pred[i]))
        n += 1
        print(f"  [{n}] {img.name}")

    return summarise(per_class_iou, n)


def eval_unet(weights_dir: str, img_dir: Path, lbl_dir: Path, classes: list[str],
              limit: int) -> dict | None:
    """Run the pretrained ResNet34-UNet (app/cv/segmenter.py) the same way.

    The UNet is a fixed 4-class model (floor/wall/door/window); for a dataset
    converted with a different taxonomy it simply has no polygons for the extra
    classes, so those score as empty predictions rather than erroring.
    """
    try:
        from app.cv.segmenter import FloorplanSegmenter
    except Exception as exc:
        print(f"  skip UNet (import failed): {exc}")
        return None

    try:
        seg = FloorplanSegmenter(weights_dir=weights_dir)
    except Exception as exc:
        print(f"  skip UNet (no weights?): {exc}")
        return None

    from PIL import Image

    n_classes = len(classes)
    per_class_iou: dict[str, list[float]] = {name: [] for name in classes}
    n = 0
    for img in iter_images(img_dir, limit):
        w, h = Image.open(img).size
        gt = rasterize_label_txt(lbl_dir / (img.stem + ".txt"), h, w, n_classes)
        result = seg.segment(str(img))

        # segmenter polygons are keyed by class name, in original pixel coords.
        from PIL import Image as PILImage, ImageDraw
        pred = []
        for i in range(n_classes):
            name = classes[i]
            canvas = PILImage.new("1", (w, h), 0)
            any_drawn = False
            if name and name in result.polygons:
                for poly in result.polygons[name]:
                    pts = [(x, y) for x, y in poly.outer]
                    if len(pts) >= 3:
                        ImageDraw.Draw(canvas).polygon(pts, fill=1)
                        any_drawn = True
                    for hole in poly.holes:
                        if len(hole) >= 3:
                            ImageDraw.Draw(canvas).polygon([(x, y) for x, y in hole], fill=0)
            pred.append(np.array(canvas, dtype=bool))

        for i in range(n_classes):
            name = classes[i]
            if gt[i].any() or pred[i].any():
                per_class_iou[name].append(iou(gt[i], pred[i]))
        n += 1
        print(f"  [{n}] {img.name}")

    return summarise(per_class_iou, n)


def summarise(per_class_iou: dict, n: int) -> dict:
    means = {}
    for name, vals in per_class_iou.items():
        vals = [v for v in vals if v == v]  # drop NaN
        means[name] = round(float(np.mean(vals)), 4) if vals else None
    present = [v for v in means.values() if v is not None]
    means["mean"] = round(float(np.mean(present)), 4) if present else None
    means["images"] = n
    return means


def print_table(results: dict, classes: list[str]) -> None:
    models = list(results)
    print("\nPer-class IoU (higher = better)")
    header = f"{'class':10s}" + "".join(f"{m:>14s}" for m in models)
    print(header)
    print("-" * len(header))
    for cls in classes + ["mean"]:
        row = f"{cls:10s}"
        for m in models:
            v = results[m].get(cls)
            row += f"{v if v is not None else '-':>14}"
        print(row)
    # Per-element winner (Phase 7: keep whichever is better per element type).
    if len(models) == 2:
        a, b = models
        print("\nWinner per element type:")
        for cls in classes:
            va, vb = results[a].get(cls), results[b].get(cls)
            if va is None or vb is None:
                print(f"  {cls:10s} - (insufficient data)")
            else:
                winner = a if va > vb else b if vb > va else "tie"
                print(f"  {cls:10s} {winner}  ({va} vs {vb})")


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate YOLO vs UNet segmenter")
    p.add_argument("--data", default="data/yolo/data.yaml")
    p.add_argument("--yolo", default="yolo11n-seg.pt",
                   help="YOLO weights (.pt) or a bare model name for untrained baseline")
    p.add_argument("--unet-weights", default="weights", help="UNet weights dir")
    p.add_argument("--no-unet", action="store_true")
    p.add_argument("--limit", type=int, default=50, help="max images (0 = all)")
    p.add_argument("--out", default="runs/segment/eval_report.json")
    args = p.parse_args()

    import yaml
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"dataset not found: {data_path}\nrun: python scripts/convert_cubicasa_yolo.py")
        return 1

    with data_path.open() as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", data_path.parent))
    val_rel = cfg.get("val", "images/valid")
    img_dir = root / val_rel
    if not img_dir.exists():
        print(f"val images not found: {img_dir}")
        return 1
    lbl_dir = img_dir.parent.parent / "labels" / img_dir.name
    # Class list comes from data.yaml so it always matches whatever taxonomy the
    # dataset was converted with (segmenter=4 or source=8); CLASSES is a fallback.
    names = cfg.get("names") or {}
    if isinstance(names, dict):        # yaml may give {0: 'floor', ...}
        classes = [names[k] for k in sorted(names, key=lambda x: int(x))]
    elif isinstance(names, list):      # or ['floor', ...]
        classes = [str(n) for n in names]
    else:
        classes = list(CLASSES)
    n_classes = len(classes) if classes else 4
    if not classes:
        classes = list(CLASSES)
    print(f"val images: {img_dir}  classes: {n_classes} ({', '.join(classes)})  limit: {args.limit}")

    results: dict[str, dict] = {}

    print(f"\n== YOLO: {args.yolo} ==")
    try:
        results["yolo"] = eval_yolo(args.yolo, img_dir, lbl_dir, classes, args.limit)
        print(f"  -> {results['yolo']}")
    except Exception as exc:
        print(f"  failed: {exc}")

    if not args.no_unet:
        print(f"\n== UNet (pretrained): {args.unet_weights} ==")
        r = eval_unet(args.unet_weights, img_dir, lbl_dir, classes, args.limit)
        if r:
            results["unet"] = r
            print(f"  -> {r}")

    if not results:
        print("no model produced results")
        return 1

    print_table(results, classes)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
