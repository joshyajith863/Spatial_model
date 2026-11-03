"""
Convert CubiCasa5K (COCO/Parquet) -> Ultralytics YOLO-segmentation format.

Source: https://huggingface.co/datasets/phungpx/cubicassa5k-coco
    4,228 train / 748 valid floor-plan images with polygon masks, embedded in
    Parquet. 9 COCO category ids (0-8); id 0 ('objects') is an empty placeholder,
    the 8 real classes are 1-8:

        1 bathroom  2 bed  3 door  4 kitchen
        5 room      6 stairs  7 wall  8 window

There is NO 'floor' class in the source, and the source taxonomy does not match
app/cv/segmenter.py (floor/wall/door/window). TAXONOMY below bridges the two.

Output layout:
    data/yolo/images/<split>/*      original image bytes, unchanged
    data/yolo/labels/<split>/*.txt  YOLO-seg lines: cls x1 y1 x2 y2 ... (0..1)
    data/yolo/data.yaml             Ultralytics dataset descriptor

Usage:
    python scripts/convert_cubicasa_yolo.py                  # both splits, all rows
    python scripts/convert_cubicasa_yolo.py --split valid --limit 20   # smoke test
    python scripts/convert_cubicasa_yolo.py --taxonomy source          # keep 8 classes

Notes:
    - Images are written as-is (no re-encode) to preserve fidelity; the file
      extension is sniffed from magic bytes so OpenCV/cv2.imread can still open
      them (cv2 keys off the extension, so a PNG named .jpg would fail).
    - Images that end up with zero valid polygons after mapping are skipped:
      Ultralytics treats a missing/empty label as "no objects", which would
      silently teach the model that this plan is background.
"""

from __future__ import annotations

import argparse
import io
import sys
from collections import Counter
from pathlib import Path

REPO_ID = "phungpx/cubicassa5k-coco"
SPLIT_FILES = {
    "train": [f"data/train-{i:05d}-of-00004.parquet" for i in range(4)],
    "valid": ["data/valid-00000-of-00001.parquet"],
}
SPLIT_ALIASES = {"val": "valid", "validation": "valid"}

# --- Class taxonomies -------------------------------------------------------
# Source COCO ids (excluding placeholder 0) -> source names, in id order.
SOURCE_CLASSES = {
    1: "bathroom",
    2: "bed",
    3: "door",
    4: "kitchen",
    5: "room",
    6: "stairs",
    7: "wall",
    8: "window",
}

# Target taxonomy matching app/cv/segmenter.py: floor(0) wall(1) door(2) window(3).
SEGMENTER_CLASSES = ["floor", "wall", "door", "window"]

# Map source id -> target class name.
#   wall/door/window map straight across; the room-ish categories (room,
#   bathroom, kitchen, bed) and stairs all denote floor area, so they collapse
#   onto 'floor' — the same region the segmenter calls floor but does not
#   extract (EXTRACT_CLASSES there is wall/door/window only).
SEGMENTER_MAP = {
    1: "floor",   # bathroom
    2: "floor",   # bed
    3: "door",
    4: "floor",   # kitchen
    5: "floor",   # room
    6: "floor",   # stairs
    7: "wall",
    8: "window",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sniff_ext(data: bytes, fallback: str) -> str:
    """Choose an extension from magic bytes; cv2.imread needs a correct one."""
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:2] in (b"II", b"MM"):
        return ".tif"
    if data[:4] == b"BM":
        return ".bmp"
    return fallback if fallback in IMAGE_EXTS else ".jpg"


def poly_area(pts: list[tuple[float, float]]) -> float:
    """Shoelace area — used to drop degenerate polygons."""
    area = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def image_bytes(value) -> bytes | None:
    """Pull raw bytes out of the Parquet image cell (struct or PIL or raw)."""
    if isinstance(value, dict):
        b = value.get("bytes")
        if b:
            return bytes(b)
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    # PIL.Image fallback: re-encode rather than lose the row.
    if hasattr(value, "save"):
        buf = io.BytesIO()
        value.save(buf, format="PNG")
        return buf.getvalue()
    return None


def build_names(taxonomy: str) -> dict[int, str]:
    if taxonomy == "segmenter":
        return dict(enumerate(SEGMENTER_CLASSES))
    return {i: name for i, (_, name) in enumerate(sorted(SOURCE_CLASSES.items()))}


def mapping_for(taxonomy: str) -> tuple[dict[int, str], dict[str, int]]:
    names = build_names(taxonomy)
    class_to_idx = {v: k for k, v in names.items()}
    if taxonomy == "segmenter":
        return SEGMENTER_MAP, class_to_idx
    # 'source': keep the 8 real classes, reindexed densely (0..7).
    source_to_name = {cid: SOURCE_CLASSES[cid] for cid in SOURCE_CLASSES}
    return source_to_name, class_to_idx


def convert(args) -> int:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    splits = [args.split] if args.split != "both" else ["train", "valid"]
    out = Path(args.out)
    taxonomy = args.taxonomy
    mapping, class_to_idx = mapping_for(taxonomy)
    names = build_names(taxonomy)

    total_written = 0
    total_skipped = 0
    grand_tally: Counter = Counter()

    for split in splits:
        split = SPLIT_ALIASES.get(split, split)
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        written = skipped = 0
        tally: Counter = Counter()
        processed = 0

        for rel in SPLIT_FILES[split]:
            if args.limit and processed >= args.limit:
                break
            path = hf_hub_download(REPO_ID, rel, repo_type="dataset")
            df = pd.read_parquet(path)
            print(f"  read {rel} ({len(df)} rows)")

            for _, row in df.iterrows():
                if args.limit and processed >= args.limit:
                    break
                processed += 1

                raw = image_bytes(row["image"])
                if not raw:
                    skipped += 1
                    continue

                width, height = int(row["width"]), int(row["height"])
                anns = row["annotations"]
                # HF stores annotations columnar (dict of parallel lists); other
                # exports may be rowwise (list of COCO dicts). Handle both.
                if isinstance(anns, dict):
                    lines, row_tally = _convert_columnar(
                        anns, width, height, mapping, class_to_idx
                    )
                else:
                    lines, row_tally = _convert_rowwise(
                        anns, width, height, mapping, class_to_idx
                    )

                if not lines:
                    skipped += 1
                    continue

                # Unique stem per image (image_id disambiguates file_name collisions).
                suffix = sniff_ext(raw, Path(row["file_name"]).suffix.lower())
                stem = f"{row['image_id']}_{Path(row['file_name']).stem}"
                (img_dir / f"{stem}{suffix}").write_bytes(raw)
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")

                written += 1
                tally.update(row_tally)

        total_written += written
        total_skipped += skipped
        grand_tally.update(tally)
        print(f"[{split}] wrote {written}, skipped {skipped} (no usable labels)")
        for name in names.values():
            if tally.get(name):
                print(f"    {name:10s} {tally[name]:,}")

    write_data_yaml(out, names, splits)
    print(f"\ntotal images: {total_written} (skipped {total_skipped})")
    print("per-class instances:")
    for name in names.values():
        print(f"  {name:10s} {grand_tally.get(name, 0):,}")
    print(f"\ndataset: {out / 'data.yaml'}")
    return 0


def _convert_columnar(anns: dict, width: int, height: int, mapping, class_to_idx):
    """Columnar layout: parallel lists per subfield (the HF dataset's storage)."""
    from collections import Counter as _C

    lines: list[str] = []
    tally: _C = _C()
    n = len(anns["id"])
    segs_all = anns["segmentation"]
    cats = anns["category_id"]
    crowds = anns.get("iscrowd")

    for i in range(n):
        cls_name = mapping.get(int(cats[i]))
        if cls_name is None or (crowds is not None and int(crowds[i])):
            continue
        cls_idx = class_to_idx[cls_name]
        # segs_all[i] may be a numpy array (pandas/pyarrow) — never test array
        # truthiness with `or []`, it raises on multi-element arrays.
        segs = segs_all[i]
        if segs is None:
            segs = []
        w, h = float(width), float(height)
        if w <= 0 or h <= 0:
            continue
        for flat in segs:
            # `flat` may be a numpy array — avoid `not flat`, use len().
            if flat is None or len(flat) < 6:
                continue
            pts = [(float(flat[j]), float(flat[j + 1])) for j in range(0, len(flat) - 1, 2)]
            if len(pts) < 3 or poly_area(pts) < 1.0:
                continue
            xs = [min(max(x / w, 0.0), 1.0) for x, _ in pts]
            ys = [min(max(y / h, 0.0), 1.0) for _, y in pts]
            if max(xs) - min(xs) < 1e-4 or max(ys) - min(ys) < 1e-4:
                continue
            coords = " ".join(f"{v:.6f}" for pair in zip(xs, ys) for v in pair)
            lines.append(f"{cls_idx} {coords}")
            tally[cls_name] += 1
    return lines, tally


def _convert_rowwise(anns: list, width: int, height: int, mapping, class_to_idx):
    """Rowwise layout: list of COCO annotation dicts."""
    from collections import Counter as _C

    lines: list[str] = []
    tally: _C = _C()
    for ann in anns or []:
        cls_name = mapping.get(int(ann.get("category_id", -1)))
        if cls_name is None or int(ann.get("iscrowd", 0)):
            continue
        cls_idx = class_to_idx[cls_name]
        w, h = float(width), float(height)
        if w <= 0 or h <= 0:
            continue
        segs = ann.get("segmentation")
        if segs is None:
            segs = []
        for flat in segs:
            # `flat` may be a numpy array — avoid `not flat`, use len().
            if flat is None or len(flat) < 6:
                continue
            pts = [(float(flat[j]), float(flat[j + 1])) for j in range(0, len(flat) - 1, 2)]
            if len(pts) < 3 or poly_area(pts) < 1.0:
                continue
            xs = [min(max(x / w, 0.0), 1.0) for x, _ in pts]
            ys = [min(max(y / h, 0.0), 1.0) for _, y in pts]
            if max(xs) - min(xs) < 1e-4 or max(ys) - min(ys) < 1e-4:
                continue
            coords = " ".join(f"{v:.6f}" for pair in zip(xs, ys) for v in pair)
            lines.append(f"{cls_idx} {coords}")
            tally[cls_name] += 1
    return lines, tally


def write_data_yaml(out: Path, names: dict[int, str], splits: list[str]) -> None:
    import yaml

    present = [s for s in splits if (out / "images" / SPLIT_ALIASES.get(s, s)).exists()]
    if not present:
        present = [s for s in ("train", "valid") if (out / "images" / s).exists()]

    # Ultralytics wants train + val; if only one split was converted, point both
    # at it so `yolo train` doesn't hard-fail (warn, don't silently pretend).
    train_split = "train" if "train" in present else present[0]
    val_split = "valid" if "valid" in present else train_split
    if "train" not in present or "valid" not in present:
        print(f"WARNING: only {present} converted; train='{train_split}' val='{val_split}'")

    cfg = {
        "path": str(out.resolve()),
        "train": f"images/{train_split}",
        "val": f"images/{val_split}",
        "names": names,
    }
    with (out / "data.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main() -> int:
    p = argparse.ArgumentParser(description="CubiCasa5K COCO -> YOLO-seg converter")
    p.add_argument("--out", default="data/yolo", help="output dataset root")
    p.add_argument("--split", default="both", choices=["train", "valid", "both", "val"])
    p.add_argument("--limit", type=int, default=0, help="max images per split (0 = all)")
    p.add_argument("--taxonomy", default="segmenter", choices=["segmenter", "source"],
                   help="'segmenter' = 4 classes matching app/cv/segmenter.py (default); "
                        "'source' = the 8 CubiCasa classes reindexed 0..7")
    args = p.parse_args()

    try:
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
        import huggingface_hub  # noqa: F401
    except ImportError as exc:
        print(f"missing dependency: {exc.name}  (pip install pandas pyarrow huggingface_hub)")
        return 1

    return convert(args)


if __name__ == "__main__":
    raise SystemExit(main())
