"""
Fine-tune a YOLO segmentation model on the CubiCasa5K dataset.

This is the build-only training entry point — it is NOT run automatically. On a
CPU-only box (no CUDA) expect on the order of an hour per epoch for a few
thousand images, so start small: a handful of epochs, or a `--limit` slice of the
dataset, before committing to a full run.

Model: yolo11n-seg (smallest YOLO11 seg model, pretrained on COCO — the seg head
transfers to our 4 classes). WORKPLAN Phase 7 also allows YOLOv8n-seg; pass
--model to switch. (YOLOv10 has no official segmentation variant in Ultralytics.)

Prep first:
    python scripts/convert_cubicasa_yolo.py           # writes data/yolo/

Run:
    python scripts/train_yolo.py --dry-run             # validate config, no training
    python scripts/train_yolo.py --epochs 3 --limit 50 # quick CPU sanity run
    python scripts/train_yolo.py --epochs 100          # real run (GPU recommended)
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_DATA = "data/yolo/data.yaml"
DEFAULT_MODEL = "yolo11n-seg.pt"  # pretrained COCO weights, auto-downloaded
DEFAULT_PROJECT = "runs/segment"
DEFAULT_NAME = "cubicasa"


def build_kwargs(args) -> dict:
    kw = dict(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        patience=args.patience,
        # CPU is the default here;AMP is fine on CPU but adds no speed, and
        # cache='disk' avoids holding every decoded image in RAM.
        cache=False,
        amp=not args.no_amp,
        verbose=True,
        seed=args.seed,
    )
    if args.limit:
        # Ultralytics has no built-in image cap, so subset the dataset up front.
        kw["data"] = make_subset_yaml(args)
    return kw


def make_subset_yaml(args) -> str:
    """Write a data.yaml pointing at only the first --limit images, so a quick
    CPU run doesn't have to chew through 4k images."""
    import yaml

    root = Path(args.data).resolve().parent
    src = {}
    with open(args.data) as f:
        src = yaml.safe_load(f)

    limit = args.limit
    subset_root = root / f"subset_{limit}"
    made = {}
    for split in ("train", "val"):
        img_dir = root / src.get(split, f"images/{split}")
        if not img_dir.exists():
            continue
        keep = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in
                      {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})[:limit]
        sub_img = subset_root / "images" / split
        sub_lbl = subset_root / "labels" / split
        sub_img.mkdir(parents=True, exist_ok=True)
        sub_lbl.mkdir(parents=True, exist_ok=True)
        # Labels mirror images/ -> labels/ by convention.
        src_lbl_dir = img_dir.parent.parent / "labels" / img_dir.name
        for p in keep:
            (sub_img / p.name).write_bytes(p.read_bytes())
            src_lbl = src_lbl_dir / (p.stem + ".txt")
            if src_lbl.exists():
                (sub_lbl / (p.stem + ".txt")).write_bytes(src_lbl.read_bytes())
        made[split] = f"images/{split}"

    cfg = {
        "path": str(subset_root.resolve()),
        "train": made.get("train", made.get("val")),
        "val": made.get("val", made.get("train")),
        "names": src.get("names", {}),
    }
    subset_root.mkdir(parents=True, exist_ok=True)
    dest = subset_root / "data.yaml"
    with dest.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"subset data.yaml -> {dest} (limit={limit}/split)")
    return str(dest)


def main() -> int:
    p = argparse.ArgumentParser(description="Fine-tune YOLO seg on CubiCasa5K")
    p.add_argument("--data", default=DEFAULT_DATA, help="dataset yaml from convert_cubicasa_yolo.py")
    p.add_argument("--model", default=DEFAULT_MODEL, help="base model (yolo11n-seg.pt / yolov8n-seg.pt)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8, help="small default for CPU RAM")
    p.add_argument("--device", default="cpu", help="'cpu', '0', or '0,1'")
    p.add_argument("--workers", type=int, default=2, help="Windows: keep low; 0 if dataloader hangs")
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--patience", type=int, default=20, help="early-stopping patience (epochs)")
    p.add_argument("--limit", type=int, default=0, help="train on first N images per split (0=all)")
    p.add_argument("--imgsz-note", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exist-ok", action="store_true")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--dry-run", action="store_true", help="build config + verify model/data, then exit")
    args = p.parse_args()

    # --- dependency / data checks -------------------------------------------
    try:
        import ultralytics
    except ImportError:
        print("missing: pip install ultralytics")
        return 1

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"dataset not found: {data_path}")
        print("run first:  python scripts/convert_cubicasa_yolo.py")
        return 1

    print(f"ultralytics {ultralytics.__version__}")
    print(f"model      {args.model}")
    print(f"data       {args.data}")
    print(f"device     {args.device}   epochs={args.epochs}  imgsz={args.imgsz}  batch={args.batch}")
    if str(args.device).lower() == "cpu":
        print("! CPU-only: expect ~1h+/epoch on the full set. Use --limit/--epochs to start small.")

    kw = build_kwargs(args)

    if args.dry_run:
        print("\n[dry-run] config OK, not training. Would call:")
        print(f"  YOLO('{args.model}').train(data=..., epochs={args.epochs}, ...)")
        return 0

    from ultralytics import YOLO

    model = YOLO(args.model)  # downloads pretrained weights on first use
    results = model.train(**kw)

    print("\ntraining complete.")
    try:
        best = Path(results.save_dir) / "weights" / "best.pt"
        if best.exists():
            print(f"best weights: {best}")
            # Quick validation readout from the run's results.
            for r in results.results_dict.items():
                print(f"  {r}")
    except Exception as exc:
        print(f"(could not summarise results: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
