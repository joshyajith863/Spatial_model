"""
Smoke test for the pretrained floorplan segmenter (WORKPLAN perception step).

Loads the CubiCasa5K-trained ResNet34-UNet and runs it on a raster blueprint,
printing per-class polygon counts and writing a debug overlay PNG.

Usage:
    python scripts/test_segmenter.py [image_path]
"""

import sys
import time
from pathlib import Path


def main() -> int:
    default_img = Path(
        "data/uploads/375f6cc2-b7bf-44b2-b0c7-423e5fc4727f_b9ea330f65ca8a9f10be077a6af1bde7.jpg"
    )
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_img
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}")
        return 1

    print(f"Image: {image_path}")

    from app.cv.segmenter import CLASS_COLORS, FloorplanSegmenter

    t0 = time.time()
    seg = FloorplanSegmenter(weights_dir="weights", device="auto")
    print(f"Model loaded on {seg.device} in {time.time() - t0:.1f}s")
    print(f"Input size: {seg.image_size}  letterbox={seg.letterbox}  normalize={seg.normalize}")

    t0 = time.time()
    result = seg.segment(image_path)
    dt = time.time() - t0

    print(f"\nSegmentation took {dt:.2f}s (CPU)")
    print(f"Original size: {result.original_size}")
    print(f"Polygons per class: {result.counts()}")

    # Write a debug overlay: input image + class-colored polygon fills.
    try:
        from PIL import Image, ImageDraw

        base = Image.open(image_path).convert("RGB").resize(result.original_size)
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for cls, polys in result.polygons.items():
            r, g, b = CLASS_COLORS[cls]
            for p in polys:
                if len(p.outer) >= 3:
                    draw.polygon([(x, y) for x, y in p.outer], fill=(r, g, b, 90),
                                 outline=(r, g, b, 255))
        out = Path("data/segmenter_debug.png")
        Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB").save(out)
        print(f"Wrote debug overlay: {out}")
    except Exception as exc:
        print(f"(overlay render skipped: {exc})")

    total = sum(result.counts().values())
    print(f"\nTotal polygons: {total}")
    if total == 0:
        print("WARNING: no polygons detected — check image contrast / try another blueprint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
