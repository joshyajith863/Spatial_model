"""
Download the pretrained floorplan segmenter weights.

Fetches the MIT-licensed CubiCasa5K-trained ResNet34-UNet from HuggingFace
into weights/ (gitignored — too large to commit).

Usage:
    python scripts/download_weights.py
"""

from pathlib import Path
import urllib.request

BASE = "https://huggingface.co/Yytsi/floorplan-to-3d-walls/resolve/main"
FILES = ["best.safetensors", "config.yaml"]


def main() -> int:
    out_dir = Path("weights")
    out_dir.mkdir(exist_ok=True)
    for name in FILES:
        dest = out_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"skip (exists): {dest} ({dest.stat().st_size} bytes)")
            continue
        url = f"{BASE}/{name}"
        print(f"downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        print(f"  done ({dest.stat().st_size} bytes)")
    print("\nWeights ready in weights/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
