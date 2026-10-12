"""
Spatial Knowledge Graph engine.

Ground truth store + deterministic math over a confirmed blueprint graph.
Built with NetworkX (topology: rooms, doors, adjacency) and Shapely
(geometry: polygons, areas, clearances).

No LLM calls happen in this module. Every function here should be pure,
testable graph/geometry logic.
"""

import networkx as nx


class SpatialKnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        # Room polygons keyed by room id, as shapely.geometry.Polygon
        self.room_geometry: dict[str, object] = {}

    def add_room(self, room_id: str, polygon, label: str | None = None):
        self.graph.add_node(room_id, label=label)
        self.room_geometry[room_id] = polygon

    def add_door(self, room_a: str, room_b: str, width_m: float):
        self.graph.add_edge(room_a, room_b, width_m=width_m)

    def compute_area(self, room_id: str) -> float:
        polygon = self.room_geometry.get(room_id)
        if polygon is None:
            raise ValueError(f"Unknown room: {room_id}")
        return polygon.area

    def find_path(self, room_from: str, room_to: str) -> list[str]:
        return nx.shortest_path(self.graph, source=room_from, target=room_to)

    def check_egress(self, room_id: str, exit_rooms: list[str]) -> dict:
        """Shortest path from room_id to the nearest designated exit."""
        best = None
        for exit_room in exit_rooms:
            try:
                path = nx.shortest_path(self.graph, source=room_id, target=exit_room)
            except nx.NetworkXNoPath:
                continue
            if best is None or len(path) < len(best):
                best = path
        if best is None:
            raise ValueError(f"No egress path found from {room_id}")
        return {"path": best, "hops": len(best) - 1}


def execute(skg: "SpatialKnowledgeGraph", structured_query) -> dict:
    """Dispatches a StructuredQuery to the right SKG method. The caller
    (app/api/query.py) is responsible for looking up the graph by
    blueprint_id via app/skg/store.py before calling this."""
    intent = structured_query.intent

    if intent == "find_path":
        path = skg.find_path(structured_query.room_from, structured_query.room_to)
        return {"path": path}

    if intent == "compute_area":
        area = skg.compute_area(structured_query.room)
        return {"room": structured_query.room, "area_sq_m": area}

    if intent == "check_egress":
        # Exit rooms aren't tagged yet (no explicit "this room is an exit"
        # data from any parser) — until that exists, treat every room with
        # only one connection (a leaf node) as a provisional exit candidate.
        exit_rooms = [n for n in skg.graph.nodes if skg.graph.degree[n] <= 1]
        return skg.check_egress(structured_query.room, exit_rooms)

    raise ValueError(f"Unhandled intent: {intent}")
