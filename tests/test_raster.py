"""Phase 1 raster ingestion tests (app/cv/raster.py + /api/blueprints/upload)."""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("shapely")

import cv2

from app.cv.raster import isolate_walls, parse_raster, preprocess


def make_plan_png(path, width=800, height=600, wall=14) -> str:
    """A simple 3-room plan: thick wall rectangles, plus thin 'furniture' lines
    that must NOT survive wall isolation."""
    img = np.full((height, width, 3), 255, np.uint8)
    cv2.rectangle(img, (40, 40), (width - 40, height - 40), (0, 0, 0), wall)
    cv2.line(img, (width // 2, 40), (width // 2, int(height * 0.6)), (0, 0, 0), wall)
    cv2.line(img, (width // 2, int(height * 0.6)), (width - 40, int(height * 0.6)),
             (0, 0, 0), wall)
    # thin strokes: furniture / dimension line (must be rejected)
    cv2.line(img, (100, 120), (260, 120), (0, 0, 0), 2)
    cv2.circle(img, (160, 200), 18, (0, 0, 0), 2)
    assert cv2.imwrite(str(path), img)
    return str(path)


def test_wall_isolation_keeps_walls_drops_thin_lines(tmp_path):
    path = make_plan_png(tmp_path / "plan.png")
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    ink, meta = preprocess(rgb)
    wall, thickness = isolate_walls(ink)

    assert wall.any(), "no walls detected"
    # A 14 px stroke has half-thickness 7; thin 2 px strokes must not survive.
    assert thickness >= 5.0
    assert (ink & ~wall).sum() > 0, "thin strokes should be filtered out"
    # The thin furniture line (rows 119-121, cols 100-260) must not be wall.
    assert wall[119:122, 110:250].sum() == 0


def test_parse_raster_geometry_and_scale(tmp_path):
    path = make_plan_png(tmp_path / "plan.png")
    # 50 px per metre -> the 720 px wide drawing is ~14.4 m across.
    fp = parse_raster(path, pixels_per_meter=50)

    assert fp.source_format == "raster"
    assert fp.units == "m"
    assert len(fp.walls) >= 6, f"expected wall segments, got {len(fp.walls)}"
    assert len(fp.rooms) >= 2, f"expected enclosed rooms, got {len(fp.rooms)}"

    minx, miny, maxx, maxy = fp.bounds
    assert maxx - minx == pytest.approx(14.4, abs=1.5)
    assert maxy - miny == pytest.approx(10.4, abs=1.5)

    # y-up plan space: the top of the image must map to the highest plan y.
    top_wall_y = max(max(w.start.y, w.end.y) for w in fp.walls)
    bottom_wall_y = min(min(w.start.y, w.end.y) for w in fp.walls)
    assert top_wall_y > bottom_wall_y
    assert top_wall_y == pytest.approx(maxy, abs=0.6)

    # Scale was supplied, so no uncalibrated-scale warning should appear.
    assert not any("uncalibrated" in w.lower() for w in fp.warnings)


def test_parse_raster_estimates_scale_when_unsupplied(tmp_path):
    path = make_plan_png(tmp_path / "plan.png")
    fp = parse_raster(path)
    assert any("uncalibrated" in w.lower() for w in fp.warnings)
    assert all(w.area_m2 > 0 for w in fp.rooms)


def test_raster_floorplan_builds_a_mesh(tmp_path):
    pytest.importorskip("trimesh")
    from app.mesh.builder import build_glb

    path = make_plan_png(tmp_path / "plan.png")
    fp = parse_raster(path, pixels_per_meter=50)
    result = build_glb(fp)
    assert result.triangle_count > 0
    assert len(result.glb) > 0
    # bounds are world metres: footprint must match the drawing, not be absurd
    width = result.bounds[3] - result.bounds[0]
    assert 5.0 < width < 40.0, f"implausible footprint width {width} m"


def test_api_accepts_raster_upload_and_confirms(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    path = make_plan_png(tmp_path / "api_plan.png")
    png = open(path, "rb").read()

    r = client.post("/api/blueprints/upload", files={"file": ("plan.png", png)},
                    params={"pixels_per_meter": 50})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "draft_ready"
    assert body["floorplan"]["source_format"] == "raster"
    assert len(body["floorplan"]["walls"]) >= 6

    # confirming must build a queryable graph from the raster rooms
    r2 = client.post(f"/api/blueprints/{body['id']}/confirm")
    assert r2.status_code == 200, r2.text


def test_pdf_without_pymupdf_gives_clear_error(tmp_path):
    pytest.importorskip("app.cv.raster")
    import importlib.util

    if importlib.util.find_spec("fitz") is not None:
        pytest.skip("pymupdf installed; covered by the real PDF path instead")
    from app.cv.raster import load_raster

    pdf = tmp_path / "plan.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    with pytest.raises(RuntimeError, match="pymupdf"):
        load_raster(pdf)
