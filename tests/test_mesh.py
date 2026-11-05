import pytest

pytest.importorskip("trimesh")
pytest.importorskip("shapely")

from app.mesh.builder import build_glb, build_scene
from app.models import NormalizedFloorplan, Point, Room, Wall


def _wall(x0, y0, x1, y1):
    return Wall(start=Point(x=x0, y=y0), end=Point(x=x1, y=y1))


def _rect(x0, y0, x1, y1):
    return [Point(x=x0, y=y0), Point(x=x1, y=y0), Point(x=x1, y=y1), Point(x=x0, y=y1)]


def _plan():
    return NormalizedFloorplan(
        source_format="test",
        bounds=(0, 0, 4, 3),
        # last wall is degenerate (zero length) — it must not break the build
        walls=[_wall(0, 0, 4, 0), _wall(4, 0, 4, 3), _wall(0, 0, 0, 0)],
        rooms=[Room(id="r1", label="Room", polygon=_rect(0, 0, 4, 3), area_m2=12.0)],
    )


def test_walls_and_floors_build():
    scene, warnings = build_scene(_plan())
    names = set(scene.geometry)
    assert "walls" in names
    assert "floors" in names

    result = build_glb(_plan())
    assert result.triangle_count > 0
    assert result.glb[:4] == b"glTF"
    assert result.wall_area_m2 > 0


def test_empty_plan_raises():
    empty = NormalizedFloorplan(source_format="test", bounds=(0, 0, 0, 0), walls=[], rooms=[])
    with pytest.raises(ValueError, match="Nothing to extrude"):
        build_scene(empty)
