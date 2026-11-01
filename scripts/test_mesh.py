"""Smoke test for the mesh builder: sample floorplan -> GLB bytes.

Run:  python scripts/test_mesh.py
Exits non-zero if anything throws or the GLB is empty.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mesh.builder import MeshOptions, build_glb, sample_floorplan


def main() -> int:
    fp = sample_floorplan()
    result = build_glb(fp, MeshOptions())

    assert len(result.glb) > 1000, f"GLB too small: {len(result.glb)} bytes"
    assert result.glb[:4] == b"glTF", "GLB missing glTF magic header"
    assert result.triangle_count > 0, "No triangles generated"

    minx, miny, minz, maxx, maxy, maxz = result.bounds
    print(f"OK  GLB {len(result.glb):,} bytes, {result.triangle_count:,} tris")
    print(f"    bounds x[{minx:.2f},{maxx:.2f}] y[{miny:.2f},{maxy:.2f}] z[{minz:.2f},{maxz:.2f}]")
    print(f"    wall_area={result.wall_area_m2} m2  floor_area={result.floor_area_m2} m2")
    if result.warnings:
        print(f"    warnings: {result.warnings}")

    # Floor should sit at y=0 and walls should reach ~wall height.
    assert abs(miny) < 1e-6, f"Floor not at y=0 (miny={miny})"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
