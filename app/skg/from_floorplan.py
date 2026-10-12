"""
NormalizedFloorplan -> SpatialKnowledgeGraph.

Bridges the mesh module (app/mesh/schema.py) and the graph engine
(app/skg/engine.py), so any input format that produces a NormalizedFloorplan
(DXF, vector PDF, eventually raster CV) also gets a queryable graph, not
just a 3D mesh.
"""

from shapely.geometry import Polygon

from app.mesh.schema import NormalizedFloorplan
from app.skg.engine import SpatialKnowledgeGraph


def build_skg(floorplan: NormalizedFloorplan) -> SpatialKnowledgeGraph:
    skg = SpatialKnowledgeGraph()

    for room in floorplan.rooms:
        if len(room.points) < 3:
            continue  # not a valid polygon, skip
        polygon = Polygon(room.points)
        skg.add_room(room.room_id, polygon, label=room.label)

    # Room adjacency (needed for find_path/check_egress) isn't derivable from
    # wall segments alone without knowing which walls are shared between
    # which two rooms. Until the parser tags that explicitly (e.g. a DXF
    # "DOORS" layer naming which rooms it connects), adjacency has to be
    # added manually or inferred by proximity as a placeholder:
    _add_proximity_adjacency(skg)

    return skg


def _add_proximity_adjacency(skg: SpatialKnowledgeGraph, touch_tolerance_m: float = 0.3) -> None:
    """
    Placeholder adjacency: connects two rooms if their polygons are within
    touch_tolerance_m of each other (i.e. likely share a wall). This is a
    rough stand-in — it will connect rooms separated by a thin wall, but
    doesn't know whether a *door* actually exists between them, so
    check_egress results from this are approximate until real door data
    (DXF "DOORS" layer, or CV-detected doors) populates add_door() instead.
    """
    room_ids = list(skg.room_geometry.keys())
    for i in range(len(room_ids)):
        for j in range(i + 1, len(room_ids)):
            a, b = room_ids[i], room_ids[j]
            poly_a = skg.room_geometry[a]
            poly_b = skg.room_geometry[b]
            if poly_a.distance(poly_b) <= touch_tolerance_m:
                skg.add_door(a, b, width_m=0.9)  # assumed standard door width
