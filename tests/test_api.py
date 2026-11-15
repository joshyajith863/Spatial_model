import pytest

pytest.importorskip("fastapi")
pytest.importorskip("shapely")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from shapely.geometry import Polygon

from app.main import app
from app.skg import store
from app.skg.engine import SpatialKnowledgeGraph

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_static_mount_exists():
    # regression: app used to crash at import if app/static was missing
    assert client.get("/static/viewer.html").status_code == 200


def _stored_graph():
    g = SpatialKnowledgeGraph()
    g.add_room("kitchen", Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), label="Kitchen")
    g.add_room("hall", Polygon([(2, 0), (3, 0), (3, 2), (2, 2)]), label="Hall")
    g.add_door("kitchen", "hall", 0.9)
    store.save("t1", g)
    return g


def test_query_roundtrip():
    _stored_graph()

    r = client.post("/api/query/", json={"blueprint_id": "t1", "text": "area of kitchen"})
    assert r.status_code == 200 and r.json()["result"]["area_m2"] == pytest.approx(4.0)

    r = client.post("/api/query/", json={"blueprint_id": "t1", "text": "path from kitchen to hall"})
    assert r.json()["result"]["path"] == ["kitchen", "hall"]


def test_query_errors():
    assert client.post("/api/query/", json={"blueprint_id": "nope", "text": "area of kitchen"}).status_code == 404
    assert client.post("/api/query/", json={"blueprint_id": "t1", "text": "gibberish"}).status_code == 422


def test_confirm_builds_graph_for_queries():
    # end-to-end: a stored draft becomes queryable once confirmed
    from app.api.blueprints import BLUEPRINTS
    from app.mesh.builder import sample_floorplan

    bp_id = "bp-confirm-test"
    BLUEPRINTS[bp_id] = {"filename": "sample.dxf", "path": "", "status": "draft_ready",
                         "floorplan": sample_floorplan(), "graph": None}

    assert client.post(f"/api/blueprints/{bp_id}/confirm").status_code == 200
    r = client.post("/api/query/", json={"blueprint_id": bp_id, "text": "area of kitchen"})
    assert r.status_code == 200
    assert r.json()["result"]["area_m2"] == pytest.approx(20.0)


def test_confirm_without_floorplan_conflicts():
    from app.api.blueprints import BLUEPRINTS

    BLUEPRINTS["bp-empty"] = {"filename": "x.dxf", "path": "", "status": "draft_ready",
                              "floorplan": None, "graph": None}
    assert client.post("/api/blueprints/bp-empty/confirm").status_code == 409


def test_rejects_bad_extension():
    r = client.post("/api/blueprints/upload", files={"file": ("x.exe", b"abc")})
    assert r.status_code == 415


def test_upload_traversal_filename_leaves_no_file(tmp_path):
    # regression: client filename like "../../evil.png" must not escape data/uploads.
    # Raster uploads are now valid, so the endpoint answers 422 (garbage bytes)
    # instead of 415 — what matters is that no file lands outside data/uploads.
    from pathlib import Path

    r = client.post("/api/blueprints/upload", files={"file": ("../../evil.png", b"\x89PNG")})
    assert r.status_code in (415, 422)
    assert list(Path("data").glob("**/evil.png")) == []
