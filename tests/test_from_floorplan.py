import pytest

pytest.importorskip("shapely")

from app.mesh.builder import sample_floorplan
from app.skg.engine import execute
from app.skg.from_floorplan import build_skg
from app.voice.rule_based_translator import parse


@pytest.fixture(scope="module")
def skg():
    return build_skg(sample_floorplan())


def test_rooms_become_nodes(skg):
    assert set(skg.room_geometry) == {"room_0", "room_1", "room_2", "room_3"}


def test_neighbouring_rooms_are_adjacent(skg):
    # room polygons that touch within the tolerance get a placeholder door
    path = execute(skg, parse("path from living to bath"))["path"]
    assert path[0] == "room_0" and path[-1] == "room_3"


def test_query_answer_comes_from_geometry(skg):
    # Kitchen = rect(3,0)-(8,4) => 20 m², not a number the translator could invent
    out = execute(skg, parse("area of the kitchen"))
    assert out["area_m2"] == pytest.approx(20.0)
