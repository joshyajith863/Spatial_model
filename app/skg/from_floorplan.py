"""NormalizedFloorplan -> SpatialKnowledgeGraph bridge."""

from shapely.geometry import Polygon

from app.models import NormalizedFloorplan
from app.skg.engine import SpatialKnowledgeGraph


def build_skg(floorplan: NormalizedFloorplan, touch_tolerance: float = 0.3) -> SpatialKnowledgeGraph:
    """Rooms become nodes. Adjacency is a PLACEHOLDER: rooms whose polygons lie
    within `touch_tolerance` (plan units) are connected. This ignores whether a
    door really exists; replace with detected door data when available."""
    skg = SpatialKnowledgeGraph()
    for room in floorplan.rooms:
        if len(room.polygon) >= 3:
            skg.add_room(room.id, Polygon([(p.x, p.y) for p in room.polygon]), label=room.label)

    ids = list(skg.room_geometry)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if skg.room_geometry[a].distance(skg.room_geometry[b]) <= touch_tolerance:
                skg.add_door(a, b, width_m=0.9)
    return skg
