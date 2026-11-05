import pytest

shapely = pytest.importorskip("shapely")
from shapely.geometry import Polygon

from app.models import StructuredQuery
from app.skg.engine import SpatialKnowledgeGraph, execute


@pytest.fixture
def skg():
    g = SpatialKnowledgeGraph()
    g.add_room("kitchen", Polygon([(0, 0), (4, 0), (4, 3), (0, 3)]), label="Kitchen")
    g.add_room("hallway", Polygon([(4, 0), (5, 0), (5, 3), (4, 3)]), label="Hallway")
    g.add_room("bedroom", Polygon([(5, 0), (9, 0), (9, 3), (5, 3)]), label="Bedroom")
    g.add_door("kitchen", "hallway", 0.9)
    g.add_door("hallway", "bedroom", 0.9)
    return g


def test_area(skg):
    assert skg.compute_area("kitchen") == pytest.approx(12.0)


def test_path(skg):
    assert skg.find_path("the kitchen", "Bedroom") == ["kitchen", "hallway", "bedroom"]


def test_unknown_room_lists_known(skg):
    with pytest.raises(ValueError, match="Known rooms"):
        skg.compute_area("garage")


def test_no_path():
    g = SpatialKnowledgeGraph()
    g.add_room("a", Polygon([(0, 0), (1, 0), (1, 1)]))
    g.add_room("b", Polygon([(5, 5), (6, 5), (6, 6)]))
    with pytest.raises(ValueError, match="No connection"):
        g.find_path("a", "b")


def test_egress_provisional_then_explicit(skg):
    assert skg.check_egress("kitchen")["provisional_exits"] is True
    skg.mark_exit("bedroom")
    res = skg.check_egress("kitchen")
    assert res["provisional_exits"] is False
    assert res["path"][-1] == "bedroom"


def test_execute_dispatch(skg):
    out = execute(skg, StructuredQuery(intent="find_path", room_from="kitchen", room_to="hallway"))
    assert out["path"] == ["kitchen", "hallway"]
    assert out["hops"] == 1

    area = execute(skg, StructuredQuery(intent="compute_area", room="kitchen"))
    assert area == {"room": "kitchen", "area_m2": pytest.approx(12.0)}
